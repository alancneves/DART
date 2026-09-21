// LQ_STAGES: INT8 GEMM pipeline depth. 5 fits the 164 KB shared memory of sm80/sm87/sm90; sm86/sm89 (99 KB per block) need 3 (build.sh sets it).
#ifndef LQ_STAGES
#define LQ_STAGES 5
#endif
// Block-level TensorRT plugins that internalize LayerNorm + quantize + INT8 GEMMs so that TensorRT only sees fp16 edges.
//   LQMlp : y = x + fc2( GELU( fc1( Q1( norm(x) ) ) ) )      x, y fp16 [.., C] (residual stream), C = 1024, hidden 4736
//           norm = LayerNorm(gamma, beta, eps) [mode 0]  or  masked RMSNorm (coordinate 0 zeroed, no affine) [mode 1, Hadamard basis]
//           Q1: per-tensor int8 (s1); fc1: int8 codes [N1,K] + per-channel scales + bias, GELU (erf or tanh) fused, requantized to int8 (s2);
//           fc2: int8 codes [C,N1] + scales + bias, residual add fused, fp16 out.  All INT8 tensors are plugin-internal (workspace).
// Compiled together with lq_plugins.cu into lq_plugins2.so.
#include <cuda.h>
typedef CUresult (*PFN_cuTensorMapEncodeTiled)(...);
typedef CUresult (*PFN_cuTensorMapEncodeIm2col)(...);
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/numeric_conversion.h>
#include "lq_attn_kernel.cuh"
#include "lq_gemm_mix.cuh"
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
using namespace nvinfer1;

// ---------------------------------------------------------------- kernels ----------------------------------------------------------------
namespace lqblk {
// one warp per row: fp16 [M, C=1024] -> normalized -> int8 (scale s).  MODE 0: LayerNorm(gamma,beta,eps); MODE 1: masked RMSNorm (Hadamard basis)
template <int MODE>
__global__ void norm_quant_kernel(const __half* __restrict__ x, int8_t* __restrict__ y, const float* __restrict__ gamma, const float* __restrict__ beta, float eps, float inv_s, int M) {
    constexpr int C = 1024; int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31; if (row >= M) return;
    const __half* xr = x + (size_t)row * C; float v[32];
    #pragma unroll
    for (int i = 0; i < 16; i++) { float2 f = __half22float2(*reinterpret_cast<const __half2*>(xr + (lane + 32 * i) * 2)); v[2 * i] = f.x; v[2 * i + 1] = f.y; }
    float mean = 0.f;
    if (MODE == 0) { float s = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; i++) s += v[i];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        mean = s / C; }
    else { if (lane == 0) v[0] = 0.f; }               // coordinate 0 (the ones-direction) is zeroed; no mean subtraction
    float ss = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; i++) { float d = v[i] - mean; ss += d * d; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    float rstd = rsqrtf(ss / C + eps);
    int8_t* yr = y + (size_t)row * C;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        int c = (lane + 32 * i) * 2; float a = (v[2 * i] - mean) * rstd, b = (v[2 * i + 1] - mean) * rstd;
        if (MODE == 0) { a = a * gamma[c] + beta[c]; b = b * gamma[c + 1] + beta[c + 1]; }
        char2 q; q.x = (int8_t)__float2int_rn(fminf(fmaxf(a * inv_s, -127.f), 127.f)); q.y = (int8_t)__float2int_rn(fminf(fmaxf(b * inv_s, -127.f), 127.f));
        *reinterpret_cast<char2*>(yr + c) = q;
    }
}

// epilogue functor: Z(fp16) = alpha*acc*s[n] + b[n] + C(residual, fp16 [M,N]);  (s,b) packed as two halves per fp32 word in V
template <int kElementsPerAccess_>
class DequantResidual {
public:
    using ElementOutput = cutlass::half_t; using ElementC = cutlass::half_t; using ElementAccumulator = int32_t; using ElementCompute = float;
    using ElementZ = cutlass::half_t; using ElementT = cutlass::half_t; using ElementVector = float;
    static int const kElementsPerAccess = kElementsPerAccess_; static int const kCount = kElementsPerAccess;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>; using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
    using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>; using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>; using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;
    using FragmentOutput = FragmentZ;
    static bool const kIsHeavy = false; static bool const kStoreZ = true; static bool const kStoreT = false; static bool const kIsSingleSource = true;
    struct Params { ElementCompute alpha; ElementCompute beta; ElementCompute const* alpha_ptr; ElementCompute const* beta_ptr;
        CUTLASS_HOST_DEVICE Params() : alpha(1), beta(1), alpha_ptr(nullptr), beta_ptr(nullptr) {}
        CUTLASS_HOST_DEVICE Params(ElementCompute a, ElementCompute b) : alpha(a), beta(b), alpha_ptr(nullptr), beta_ptr(nullptr) {} };
private: ElementCompute alpha_;
public:
    CUTLASS_HOST_DEVICE DequantResidual(Params const& p) : alpha_(p.alpha) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
    CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
    CUTLASS_DEVICE void operator()(FragmentZ& frag_Z, FragmentT&, FragmentAccumulator const& AB, FragmentC const& frag_C, FragmentCompute const& V) const {
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess> acc2f; cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess> c2f;
        FragmentCompute a = acc2f(AB), c = c2f(frag_C), z;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) { unsigned u = __float_as_uint(V[i]); float s = __half2float(__ushort_as_half((unsigned short)(u & 0xffffu))); float b = __half2float(__ushort_as_half((unsigned short)(u >> 16))); z[i] = alpha_ * a[i] * s + b + c[i]; }
        cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> f2z; frag_Z = f2z(z);
    }
    CUTLASS_DEVICE void operator()(FragmentZ& frag_Z, FragmentT& t, FragmentAccumulator const& AB, FragmentCompute const& V) const { FragmentC zero; zero.clear(); (*this)(frag_Z, t, AB, zero, V); }
};
// epilogue functor: Z(int8) = q( act(alpha*acc*s + b) / oscale )   (same as lq_plugins.cu's GeluDequantQ, duplicated here to keep the TU self-contained)
template <typename ElementZ_, typename Act, int kElementsPerAccess_>
class GeluDequantQ {
public:
    using ElementOutput = ElementZ_; using ElementC = ElementZ_; using ElementAccumulator = int32_t; using ElementCompute = float;
    using ElementZ = ElementZ_; using ElementT = ElementZ_; using ElementVector = float;
    static int const kElementsPerAccess = kElementsPerAccess_; static int const kCount = kElementsPerAccess;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>; using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
    using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>; using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>; using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;
    using FragmentOutput = FragmentZ;
    static bool const kIsHeavy = true; static bool const kStoreZ = true; static bool const kStoreT = false; static bool const kIsSingleSource = true;
    struct Params { ElementCompute alpha; ElementCompute beta; ElementCompute const* alpha_ptr; ElementCompute const* beta_ptr; ElementCompute inv_oscale;
        CUTLASS_HOST_DEVICE Params() : alpha(1), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr), inv_oscale(0) {}
        CUTLASS_HOST_DEVICE Params(ElementCompute a, ElementCompute inv_os) : alpha(a), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr), inv_oscale(inv_os) {} };
private: ElementCompute alpha_, inv_oscale_;
public:
    CUTLASS_HOST_DEVICE GeluDequantQ(Params const& p) : alpha_(p.alpha), inv_oscale_(p.inv_oscale) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const { return false; }
    CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
    CUTLASS_DEVICE void compute(FragmentCompute& z, FragmentAccumulator const& AB, FragmentCompute const& V) const {
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess> acc2f; FragmentCompute a = acc2f(AB); Act act;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) { unsigned u = __float_as_uint(V[i]); float s = __half2float(__ushort_as_half((unsigned short)(u & 0xffffu))); float b = __half2float(__ushort_as_half((unsigned short)(u >> 16)));
            float y = act(alpha_ * a[i] * s + b); z[i] = (inv_oscale_ > 0.f) ? fminf(fmaxf(rintf(y * inv_oscale_), -127.f), 127.f) : y; }
    }
    CUTLASS_DEVICE void operator()(FragmentZ& frag_Z, FragmentT&, FragmentAccumulator const& AB, FragmentC const&, FragmentCompute const& V) const { FragmentCompute z; compute(z, AB, V); cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> f2z; frag_Z = f2z(z); }
    CUTLASS_DEVICE void operator()(FragmentZ& frag_Z, FragmentT&, FragmentAccumulator const& AB, FragmentCompute const& V) const { FragmentCompute z; compute(z, AB, V); cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> f2z; frag_Z = f2z(z); }
};
template <typename EpilogueOp, typename ElementC>
struct GemmT {
    using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<int8_t, cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor, ElementC, cutlass::layout::RowMajor,
        int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
        EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, LQ_STAGES>;
    static cutlass::Status run(int M, int N, int K, const int8_t* A, const int8_t* B, const void* C, int ldc, int64_t bsC, void* D, const float* V, typename EpilogueOp::Params ep, void* ws, cudaStream_t st) {
        typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, ep, A, B, C, D, (void*)V, nullptr, (int64_t)M * K, (int64_t)N * K, bsC, (int64_t)M * N, (int64_t)0, (int64_t)0, K, K, ldc, N, 0, 0);
        Gemm gemm; cutlass::Status s = gemm.initialize(args, ws, st); if (s != cutlass::Status::kSuccess) return s; return gemm(st);
    }
    static size_t workspace(int M, int N, int K) {
        typename EpilogueOp::Params ep;
        typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, ep, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, (int64_t)M * K, (int64_t)N * K, (int64_t)0, (int64_t)M * N, (int64_t)0, (int64_t)0, K, K, 0, N, 0, 0);
        return Gemm::get_workspace_size(args);
    }
    // batched over windows: problem [Mb, N, K] x batch, A/D strided per batch, B and C (table) and V shared (stride 0)
    static cutlass::Status run_batched(int Mb, int batch, int N, int K, const int8_t* A, const int8_t* B, const void* Ctab, void* D, const float* V, typename EpilogueOp::Params ep, void* ws, cudaStream_t st) {
        typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kBatched, {Mb, N, K}, batch, ep, A, B, Ctab, D, (void*)V, nullptr, (int64_t)Mb * K, (int64_t)0, (int64_t)0, (int64_t)Mb * N, (int64_t)0, (int64_t)0, K, K, N, N, 0, 0);
        Gemm gemm; cutlass::Status s = gemm.initialize(args, ws, st); if (s != cutlass::Status::kSuccess) return s; return gemm(st);
    }
    static size_t workspace_batched(int Mb, int batch, int N, int K) {
        typename EpilogueOp::Params ep;
        typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kBatched, {Mb, N, K}, batch, ep, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, (int64_t)Mb * K, (int64_t)0, (int64_t)0, (int64_t)Mb * N, (int64_t)0, (int64_t)0, K, K, N, N, 0, 0);
        return Gemm::get_workspace_size(args);
    }
};
using Fc1Erf = GemmT<GeluDequantQ<int8_t, cutlass::epilogue::thread::GELU<float>, 8>, int8_t>;
using Fc1Tanh = GemmT<GeluDequantQ<int8_t, cutlass::epilogue::thread::GELU_taylor<float>, 8>, int8_t>;
// ---- fc1 with per-column requantization (EVT epilogue): q = sat_int8( round( GELU(acc*s_k + b_k) * t_k ) ), t_k = 1/(s2 * c_k) -----------
}  // namespace lqblk (pause: EVT headers pull cute into scope)
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/device_kernel.h"
namespace lqblk {
template <class T> struct GeluReqTanh;
template <class T> struct GeluReqErf;
template <int N> struct GeluReqTanh<cutlass::Array<float, N>> {
    CUTLASS_HOST_DEVICE cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& acc, cutlass::Array<float, N> const& sv, cutlass::Array<float, N> const& bv, cutlass::Array<float, N> const& tv) const {
        cutlass::Array<float, N> r; cutlass::epilogue::thread::GELU_taylor<float> g;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; i++) { float y = g(acc[i] * sv[i] + bv[i]) * tv[i]; r[i] = fminf(fmaxf(rintf(y), -127.f), 127.f); }
        return r; } };
template <int N> struct GeluReqErf<cutlass::Array<float, N>> {
    CUTLASS_HOST_DEVICE cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& acc, cutlass::Array<float, N> const& sv, cutlass::Array<float, N> const& bv, cutlass::Array<float, N> const& tv) const {
        cutlass::Array<float, N> r; cutlass::epilogue::thread::GELU<float> g;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; i++) { float y = g(acc[i] * sv[i] + bv[i]) * tv[i]; r[i] = fminf(fmaxf(rintf(y), -127.f), 127.f); }
        return r; } };
namespace evt {
namespace tbe = cutlass::epilogue::threadblock;
using TB = cutlass::gemm::GemmShape<128, 256, 64>; using WS = cutlass::gemm::GemmShape<64, 64, 64>; using IS = cutlass::gemm::GemmShape<16, 8, 32>;
using TMap = tbe::OutputTileThreadLayout<TB, WS, int8_t, 16, 1>;
using RowV = tbe::VisitorRowBroadcast<TMap, float, cute::Stride<cute::_0, cute::_1, int64_t>>;
using StoreI8 = tbe::VisitorAuxStore<TMap, int8_t, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
template <template <class> class Fn> using Tree = tbe::Sm80EVT<StoreI8, tbe::Sm80EVT<tbe::VisitorCompute<Fn, float, float, cutlass::FloatRoundStyle::round_to_nearest>, tbe::VisitorAccFetch, RowV, RowV, RowV>>;
template <template <class> class Fn> using Kern = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16, int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
    int8_t, cutlass::layout::RowMajor, 16, int32_t, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS, IS, Tree<Fn>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, LQ_STAGES, cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
template <template <class> class Fn>
static cudaError_t run_fc1(int M, int N, int K, const int8_t* A, const int8_t* B, int8_t* D, const float* sv, const float* bv, const float* tv, cudaStream_t st) {
    using K_ = Kern<Fn>; static int sms = 0; if (!sms) { int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev); cudaFuncSetAttribute(cutlass::Kernel2<K_>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(typename K_::SharedStorage)); }
    using cute::_0; using cute::_1;
    typename Tree<Fn>::Arguments ep{ { {}, {sv, 0.f, {_0{}, _1{}, int64_t(N)}}, {bv, 0.f, {_0{}, _1{}, int64_t(N)}}, {tv, 0.f, {_0{}, _1{}, int64_t(N)}}, {} }, {D, {int64_t(N), _1{}, int64_t(M) * N}} };
    typename K_::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, ep, A, B, nullptr, nullptr, (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);
    typename K_::Params params(args, sms, 1); dim3 grid = params.get_grid_dims();
    cutlass::Kernel2<K_><<<grid, dim3(256), sizeof(typename K_::SharedStorage), st>>>(params); return cudaGetLastError();
}
}  // namespace evt
// ---- 2:4 weight-sparse MLP: fc1 (W1 sparse) -> h^T int8 [N1, M] -> fc2 (W2 sparse) -> Y^T fp16 [C, M] -> fused transpose + residual ----
}  // namespace lqblk
#include "cutlass/gemm/kernel/default_gemm_sparse_with_visitor.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/host_tensor.h"
namespace lqblk { namespace sp {
namespace tbe = cutlass::epilogue::threadblock;
using TB1 = cutlass::gemm::GemmShape<128, 128, 128>; using TB2 = cutlass::gemm::GemmShape<256, 128, 128>; using WSp = cutlass::gemm::GemmShape<64, 64, 128>; using ISp = cutlass::gemm::GemmShape<16, 8, 64>;
template <class TB, class EO, int Al> using TMap = tbe::OutputTileThreadLayout<TB, WSp, EO, Al, 1>;
template <class TB, class EO, int Al> using ColV = tbe::VisitorColBroadcast<TMap<TB, EO, Al>, float, cute::Stride<cute::_1, cute::_0, int64_t>>;   // per output channel (= row of Y^T)
template <class T> struct GeluReq3T;  template <int N> struct GeluReq3T<cutlass::Array<float, N>> {   // acc*s + b -> GELU(tanh) -> *t -> sat int8
    CUTLASS_HOST_DEVICE cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& a, cutlass::Array<float, N> const& sv, cutlass::Array<float, N> const& bv, cutlass::Array<float, N> const& tv) const {
        cutlass::Array<float, N> r; cutlass::epilogue::thread::GELU_taylor<float> g;
        for (int i = 0; i < N; i++) { float y = g(a[i] * sv[i] + bv[i]) * tv[i]; r[i] = fminf(fmaxf(rintf(y), -127.f), 127.f); }
        return r; } };
template <class T> struct GeluReq3E;  template <int N> struct GeluReq3E<cutlass::Array<float, N>> {
    CUTLASS_HOST_DEVICE cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& a, cutlass::Array<float, N> const& sv, cutlass::Array<float, N> const& bv, cutlass::Array<float, N> const& tv) const {
        cutlass::Array<float, N> r; cutlass::epilogue::thread::GELU<float> g;
        for (int i = 0; i < N; i++) { float y = g(a[i] * sv[i] + bv[i]) * tv[i]; r[i] = fminf(fmaxf(rintf(y), -127.f), 127.f); }
        return r; } };
template <class T> struct ScaleBias;  template <int N> struct ScaleBias<cutlass::Array<float, N>> {
    CUTLASS_HOST_DEVICE cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& a, cutlass::Array<float, N> const& sv, cutlass::Array<float, N> const& bv) const {
        cutlass::Array<float, N> r;
        for (int i = 0; i < N; i++) r[i] = a[i] * sv[i] + bv[i];
        return r; } };
using St1 = tbe::VisitorAuxStore<TMap<TB1, int8_t, 16>, int8_t, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
using St2 = tbe::VisitorAuxStore<TMap<TB2, cutlass::half_t, 8>, cutlass::half_t, cutlass::FloatRoundStyle::round_to_nearest, cute::Stride<int64_t, cute::_1, int64_t>>;
template <template <class> class Fn> using Tree1 = tbe::Sm80EVT<St1, tbe::Sm80EVT<tbe::VisitorCompute<Fn, float, float, cutlass::FloatRoundStyle::round_to_nearest>, tbe::VisitorAccFetch, ColV<TB1, int8_t, 16>, ColV<TB1, int8_t, 16>, ColV<TB1, int8_t, 16>>>;
using Tree2 = tbe::Sm80EVT<St2, tbe::Sm80EVT<tbe::VisitorCompute<ScaleBias, float, float, cutlass::FloatRoundStyle::round_to_nearest>, tbe::VisitorAccFetch, ColV<TB2, cutlass::half_t, 8>, ColV<TB2, cutlass::half_t, 8>>>;
template <class TB, class EC, class Tree> using Kern = typename cutlass::gemm::kernel::DefaultSparseGemmWithVisitor<int8_t, cutlass::layout::RowMajor, 16, int8_t, cutlass::layout::ColumnMajor, 16, EC, cutlass::layout::RowMajor, int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WSp, ISp, Tree, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3, cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
using K1T = Kern<TB1, int8_t, Tree1<GeluReq3T>>; using K1E = Kern<TB1, int8_t, Tree1<GeluReq3E>>; using K2 = Kern<TB2, cutlass::half_t, Tree2>;
using ElementE = typename K2::ElementE; static_assert(K2::kSparse == 2 && K2::kElementsPerElementE == 16, "int8 2:4 metadata layout");
// host: dense 2:4 codes [N, K] (row-major) -> compressed [N, K/2] + reordered metadata; returns false if a group violates 2:4
static bool compress24(const int8_t* W, int N, int K, std::vector<int8_t>& Wc, std::vector<ElementE>& Er) {
    const int Ecols = K / 32; Wc.assign((size_t)N * K / 2, 0); cutlass::HostTensor<ElementE, typename K2::LayoutE> E({N, Ecols}), R({N, Ecols});
    for (int r = 0; r < N; r++) for (int c = 0; c < Ecols; c++) { ElementE meta = 0;
        for (int g = 0; g < 8; g++) { int base = c * 32 + g * 4; int idx[2], n = 0;
            for (int i = 0; i < 4; i++) if (W[(size_t)r * K + base + i] != 0) { if (n < 2) idx[n] = i; n++; }
            if (n > 2) return false;
            if (n < 2) { for (int i = 0; i < 4 && n < 2; i++) { bool used = false; for (int j = 0; j < n; j++) used |= (idx[j] == i); if (!used) idx[n++] = i; } }
            if (idx[0] > idx[1]) { int t = idx[0]; idx[0] = idx[1]; idx[1] = t; }
            Wc[(size_t)r * (K / 2) + (base / 4) * 2] = W[(size_t)r * K + base + idx[0]]; Wc[(size_t)r * (K / 2) + (base / 4) * 2 + 1] = W[(size_t)r * K + base + idx[1]];
            meta |= (ElementE)((idx[0] | (idx[1] << 2)) << (g * 4)); }
        E.at({r, c}) = meta; }
    cutlass::reorder_meta(R.host_ref(), E.host_ref(), {N, 1, Ecols}); Er.assign(R.host_data(), R.host_data() + (size_t)N * Ecols); return true;
}
template <class K_, class Tree>
static cudaError_t launch(int Nout, int M, int Kd, const int8_t* Wc, const ElementE* E, const int8_t* B, typename Tree::Arguments const& ep, cudaStream_t st) {
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<> swz; cutlass::gemm::GemmCoord problem(Nout, M, Kd);
    cutlass::gemm::GemmCoord tiled = swz.get_tiled_shape(problem, {K_::Mma::Shape::kM, K_::Mma::Shape::kN, K_::Mma::Shape::kK}, 1); dim3 grid = swz.get_grid_shape(tiled);
    typename K_::Params params(problem, tiled, {(int8_t*)Wc, Kd / 2}, {(int8_t*)B, Kd}, {(ElementE*)E, K_::LayoutE::packed(cutlass::MatrixCoord(Nout, Kd / 32))}, ep);
    static bool attr = false; size_t smem = sizeof(typename K_::SharedStorage); if (!attr) { cudaFuncSetAttribute(cutlass::Kernel<K_>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem); attr = true; }
    cutlass::Kernel<K_><<<grid, dim3(K_::kThreadCount), smem, st>>>(params); return cudaGetLastError();
}
// int8 transpose: out[m, k] = in[k, m]  (in [K, M] row-major -> out [M, K] row-major), 32x32 smem tiles
__global__ void transpose_i8_kernel(const int8_t* __restrict__ in, int8_t* __restrict__ out, int K, int M) {
    __shared__ int8_t tile[32][33]; int k0 = blockIdx.y * 32, m0 = blockIdx.x * 32; int tx = threadIdx.x, ty = threadIdx.y;
    for (int j = ty; j < 32; j += 8) { int k = k0 + j, m = m0 + tx; if (k < K && m < M) tile[j][tx] = in[(size_t)k * M + m]; }
    __syncthreads();
    for (int j = ty; j < 32; j += 8) { int m = m0 + j, k = k0 + tx; if (m < M && k < K) out[(size_t)m * K + k] = tile[tx][j]; }
}
// y[m, c] = x[m, c] + Yt[c, m]   (fp16; 32x32 smem tiles)
__global__ void transpose_residual_kernel(const __half* __restrict__ Yt, const __half* __restrict__ x, __half* __restrict__ y, int M, int C) {
    __shared__ __half tile[32][33]; int c0 = blockIdx.y * 32, m0 = blockIdx.x * 32; int tx = threadIdx.x, ty = threadIdx.y;   // block (32, 8)
    for (int j = ty; j < 32; j += 8) { int c = c0 + j, m = m0 + tx; if (c < C && m < M) tile[j][tx] = Yt[(size_t)c * M + m]; }
    __syncthreads();
    for (int j = ty; j < 32; j += 8) { int m = m0 + j, c = c0 + tx; if (m < M && c < C) { size_t o = (size_t)m * C + c; y[o] = __hadd(x[o], tile[tx][j]); } }
}
}}  // namespace lqblk::sp
namespace lqblk {

using Fc2Res = GemmT<DequantResidual<8>, cutlass::half_t>;
static inline unsigned pack_sb(float s, float b) { unsigned short hs = __half_as_ushort(__float2half(s)), hb = __half_as_ushort(__float2half(b)); return (unsigned)hs | ((unsigned)hb << 16); }
}  // namespace lqblk

// ---------------------------------------------------------------- LQMlp plugin -----------------------------------------------------------
static const char* kMlpName = "LQMlp"; static const char* kMlpVer = "1"; static const char* kMlpNs = "";
struct MlpParams {
    int mode = 1, gelu = 0, sparse = 0; float eps = 1e-6f, s1 = 0.f, s2 = 0.f; int C = 1024, N1 = 4736;
    std::vector<float> gamma, beta, ws1, b1, ws2, b2, chan; std::vector<int8_t> codes1, codes2;   // codes1 [N1,C], codes2 [C,N1] (row-major, K contiguous); chan [N1] per-column requant multiplier (optional)
};
class LQMlpPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    MlpParams p; int8_t* dW1 = nullptr; int8_t* dW2 = nullptr; float* dV1 = nullptr; float* dV2 = nullptr; float* dG = nullptr; float* dB = nullptr; float* dS1 = nullptr; float* dB1 = nullptr; float* dT1 = nullptr; int8_t* dW1c = nullptr; int8_t* dW2c = nullptr; void* dE1 = nullptr; void* dE2 = nullptr; float* dS2 = nullptr; float* dB2 = nullptr;
    explicit LQMlpPlugin(MlpParams pp) : p(std::move(pp)) {}
    ~LQMlpPlugin() override { for (void* q : {(void*)dW1, (void*)dW2, (void*)dV1, (void*)dV2, (void*)dG, (void*)dB, (void*)dS1, (void*)dB1, (void*)dT1, (void*)dW1c, (void*)dW2c, dE1, dE2, (void*)dS2, (void*)dB2}) if (q) cudaFree(q); }
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {
        if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
        if (t == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
        return static_cast<IPluginV3OneCore*>(this);
    }
    IPluginV3* clone() noexcept override { return new LQMlpPlugin(p); }
    char const* getPluginName() const noexcept override { return kMlpName; }
    char const* getPluginVersion() const noexcept override { return kMlpVer; }
    char const* getPluginNamespace() const noexcept override { return kMlpNs; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override { return io[pos].desc.format == TensorFormat::kLINEAR && io[pos].desc.type == DataType::kHALF; }
    int32_t getOutputDataTypes(DataType* out, int32_t, DataType const*, int32_t) const noexcept override { out[0] = DataType::kHALF; return 0; }
    int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t, DimsExprs* out, int32_t, IExprBuilder&) noexcept override { out[0] = in[0]; return 0; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override {
        int64_t M = 1; for (int i = 0; i < in[0].desc.dims.nbDims - 1; i++) M *= in[0].desc.dims.d[i] > 0 ? in[0].desc.dims.d[i] : 1;
        return (size_t)M * p.C + (size_t)M * p.N1 * 2 + (size_t)M * p.C * 2 + lqblk::Fc1Erf::workspace((int)M, p.N1, p.C) + lqblk::Fc2Res::workspace((int)M, p.C, p.N1) + (256 << 10);
    }
    int32_t getValidTactics(int32_t*, int32_t) noexcept override { return 0; }
    int32_t getNbTactics() noexcept override { return 0; }
    char const* getTimingCacheID() noexcept override { return nullptr; }
    int32_t getFormatCombinationLimit() noexcept override { return 1; }
    char const* getMetadataString() noexcept override { return nullptr; }
    int32_t setTactic(int32_t) noexcept override { return 0; }
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override { return 0; }
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        f_.clear();
        f_.emplace_back("mode", &p.mode, PluginFieldType::kINT32, 1); f_.emplace_back("gelu", &p.gelu, PluginFieldType::kINT32, 1); f_.emplace_back("sparse", &p.sparse, PluginFieldType::kINT32, 1); f_.emplace_back("eps", &p.eps, PluginFieldType::kFLOAT32, 1);
        f_.emplace_back("s1", &p.s1, PluginFieldType::kFLOAT32, 1); f_.emplace_back("s2", &p.s2, PluginFieldType::kFLOAT32, 1);
        if (!p.chan.empty()) f_.emplace_back("chan", p.chan.data(), PluginFieldType::kFLOAT32, (int32_t)p.chan.size());
        f_.emplace_back("gamma", p.gamma.data(), PluginFieldType::kFLOAT32, (int32_t)p.gamma.size()); f_.emplace_back("beta", p.beta.data(), PluginFieldType::kFLOAT32, (int32_t)p.beta.size());
        f_.emplace_back("codes1", p.codes1.data(), PluginFieldType::kINT8, (int32_t)p.codes1.size()); f_.emplace_back("ws1", p.ws1.data(), PluginFieldType::kFLOAT32, (int32_t)p.ws1.size()); f_.emplace_back("b1", p.b1.data(), PluginFieldType::kFLOAT32, (int32_t)p.b1.size());
        f_.emplace_back("codes2", p.codes2.data(), PluginFieldType::kINT8, (int32_t)p.codes2.size()); f_.emplace_back("ws2", p.ws2.data(), PluginFieldType::kFLOAT32, (int32_t)p.ws2.size()); f_.emplace_back("b2", p.b2.data(), PluginFieldType::kFLOAT32, (int32_t)p.b2.size());
        fc_.nbFields = (int32_t)f_.size(); fc_.fields = f_.data(); return &fc_;
    }
    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const*, void const* const* inputs, void* const* outputs, void* ws, cudaStream_t st) noexcept override {
        int64_t M = 1; for (int i = 0; i < in[0].dims.nbDims - 1; i++) M *= in[0].dims.d[i];
        if (!dW1) upload();
        const __half* x = (const __half*)inputs[0]; __half* y = (__half*)outputs[0];
        int8_t* x8 = (int8_t*)ws; int8_t* h8 = x8 + (size_t)M * p.C; char* gws = (char*)(h8 + (size_t)M * p.N1);
        if (p.mode == 0) lqblk::norm_quant_kernel<0><<<(unsigned)((M + 7) / 8), 256, 0, st>>>(x, x8, dG, dB, p.eps, 1.f / p.s1, (int)M);
        else lqblk::norm_quant_kernel<1><<<(unsigned)((M + 7) / 8), 256, 0, st>>>(x, x8, nullptr, nullptr, p.eps, 1.f / p.s1, (int)M);
        cutlass::Status s;
        if (p.sparse) {   // 2:4 weight-sparse MLP: bit0 = fc1 sparse (h^T + int8 transpose), bit1 = fc2 sparse (Y^T + fused transpose/residual)
            const bool sp1 = p.sparse & 1, sp2 = p.sparse & 2;
            if (!dS1) { std::vector<int8_t> wc; std::vector<lqblk::sp::ElementE> er;
                if (sp1) { if (!lqblk::sp::compress24(p.codes1.data(), p.N1, p.C, wc, er)) { fprintf(stderr, "LQMlp: fc1 codes are not 2:4\n"); return 1; }
                    cudaMalloc(&dW1c, wc.size()); cudaMemcpy(dW1c, wc.data(), wc.size(), cudaMemcpyHostToDevice); cudaMalloc(&dE1, er.size() * sizeof(lqblk::sp::ElementE)); cudaMemcpy(dE1, er.data(), er.size() * sizeof(lqblk::sp::ElementE), cudaMemcpyHostToDevice); }
                if (sp2) { if (!lqblk::sp::compress24(p.codes2.data(), p.C, p.N1, wc, er)) { fprintf(stderr, "LQMlp: fc2 codes are not 2:4\n"); return 1; }
                    cudaMalloc(&dW2c, wc.size()); cudaMemcpy(dW2c, wc.data(), wc.size(), cudaMemcpyHostToDevice); cudaMalloc(&dE2, er.size() * sizeof(lqblk::sp::ElementE)); cudaMemcpy(dE2, er.data(), er.size() * sizeof(lqblk::sp::ElementE), cudaMemcpyHostToDevice); }
                std::vector<float> sv(p.N1), tv(p.N1), s2v(p.C); for (int n = 0; n < p.N1; n++) { sv[n] = p.s1 * p.ws1[n]; tv[n] = p.chan.empty() ? 1.f / p.s2 : p.chan[n]; } for (int n = 0; n < p.C; n++) s2v[n] = p.s2 * p.ws2[n];
                cudaMalloc(&dS1, p.N1 * 4); cudaMemcpy(dS1, sv.data(), p.N1 * 4, cudaMemcpyHostToDevice); cudaMalloc(&dB1, p.N1 * 4); cudaMemcpy(dB1, p.b1.data(), p.N1 * 4, cudaMemcpyHostToDevice); cudaMalloc(&dT1, p.N1 * 4); cudaMemcpy(dT1, tv.data(), p.N1 * 4, cudaMemcpyHostToDevice);
                cudaMalloc(&dS2, p.C * 4); cudaMemcpy(dS2, s2v.data(), p.C * 4, cudaMemcpyHostToDevice); cudaMalloc(&dB2, p.C * 4); cudaMemcpy(dB2, p.b2.data(), p.C * 4, cudaMemcpyHostToDevice); }
            using cute::_0; using cute::_1;
            int8_t* hT = h8 + (size_t)M * p.N1; __half* yt = (__half*)(hT + (size_t)M * p.N1);   // workspace: x8 | h [M,N1] | h^T [N1,M] | Y^T [C,M] fp16 | cutlass ws
            char* gws2 = (char*)(yt + (size_t)M * p.C);
            if (sp1) {
                typename lqblk::sp::Tree1<lqblk::sp::GeluReq3T>::Arguments ep1{ { {}, {dS1, 0.f, {_1{}, _0{}, int64_t(p.N1)}}, {dB1, 0.f, {_1{}, _0{}, int64_t(p.N1)}}, {dT1, 0.f, {_1{}, _0{}, int64_t(p.N1)}}, {} }, {hT, {int64_t(M), _1{}, int64_t(p.N1) * M}} };
                cudaError_t ce = p.gelu ? lqblk::sp::launch<lqblk::sp::K1T, lqblk::sp::Tree1<lqblk::sp::GeluReq3T>>(p.N1, (int)M, p.C, dW1c, (const lqblk::sp::ElementE*)dE1, x8, ep1, st)
                                        : lqblk::sp::launch<lqblk::sp::K1E, lqblk::sp::Tree1<lqblk::sp::GeluReq3E>>(p.N1, (int)M, p.C, dW1c, (const lqblk::sp::ElementE*)dE1, x8, *reinterpret_cast<typename lqblk::sp::Tree1<lqblk::sp::GeluReq3E>::Arguments*>(&ep1), st);
                if (ce != cudaSuccess) return 1;
                dim3 gt((unsigned)((M + 31) / 32), (unsigned)(p.N1 / 32)); lqblk::sp::transpose_i8_kernel<<<gt, dim3(32, 8), 0, st>>>(hT, h8, p.N1, (int)M);
            } else {
                cutlass::Status s1s = p.gelu ? lqblk::Fc1Tanh::run((int)M, p.N1, p.C, x8, dW1, nullptr, 0, 0, h8, dV1, typename lqblk::Fc1Tanh::Gemm::EpilogueOutputOp::Params(p.s1, 1.f / p.s2), gws2, st)
                                             : lqblk::Fc1Erf::run((int)M, p.N1, p.C, x8, dW1, nullptr, 0, 0, h8, dV1, typename lqblk::Fc1Erf::Gemm::EpilogueOutputOp::Params(p.s1, 1.f / p.s2), gws2, st);
                if (s1s != cutlass::Status::kSuccess) return 1;
            }
            if (sp2) {
                typename lqblk::sp::Tree2::Arguments ep2{ { {}, {dS2, 0.f, {_1{}, _0{}, int64_t(p.C)}}, {dB2, 0.f, {_1{}, _0{}, int64_t(p.C)}}, {} }, {(cutlass::half_t*)yt, {int64_t(M), _1{}, int64_t(p.C) * M}} };
                if (lqblk::sp::launch<lqblk::sp::K2, lqblk::sp::Tree2>(p.C, (int)M, p.N1, dW2c, (const lqblk::sp::ElementE*)dE2, h8, ep2, st) != cudaSuccess) return 1;
                dim3 grid((unsigned)((M + 31) / 32), (unsigned)(p.C / 32)); lqblk::sp::transpose_residual_kernel<<<grid, dim3(32, 8), 0, st>>>(yt, x, y, (int)M, p.C); return cudaGetLastError() == cudaSuccess ? 0 : 1;
            }
            return lqblk::Fc2Res::run((int)M, p.C, p.N1, h8, dW2, x, p.C, (int64_t)M * p.C, y, dV2, typename lqblk::Fc2Res::Gemm::EpilogueOutputOp::Params(p.s2, 1.f), gws2, st) == cutlass::Status::kSuccess ? 0 : 1;
        }
        if (!p.chan.empty()) {   // per-column requantization (fc2 per-channel activation scales): EVT epilogue
            if (!dS1) { std::vector<float> sv(p.N1), tv(p.N1); for (int n = 0; n < p.N1; n++) { sv[n] = p.s1 * p.ws1[n]; tv[n] = p.chan[n]; }
                cudaMalloc(&dS1, p.N1 * 4); cudaMemcpy(dS1, sv.data(), p.N1 * 4, cudaMemcpyHostToDevice); cudaMalloc(&dB1, p.N1 * 4); cudaMemcpy(dB1, p.b1.data(), p.N1 * 4, cudaMemcpyHostToDevice); cudaMalloc(&dT1, p.N1 * 4); cudaMemcpy(dT1, tv.data(), p.N1 * 4, cudaMemcpyHostToDevice); }
            cudaError_t ce = p.gelu ? lqblk::evt::run_fc1<lqblk::GeluReqTanh>((int)M, p.N1, p.C, x8, dW1, h8, dS1, dB1, dT1, st) : lqblk::evt::run_fc1<lqblk::GeluReqErf>((int)M, p.N1, p.C, x8, dW1, h8, dS1, dB1, dT1, st);
            s = (ce == cudaSuccess) ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
        } else if (p.gelu) s = lqblk::Fc1Tanh::run((int)M, p.N1, p.C, x8, dW1, nullptr, 0, 0, h8, dV1, typename lqblk::Fc1Tanh::Gemm::EpilogueOutputOp::Params(p.s1, 1.f / p.s2), gws, st);
        else s = lqblk::Fc1Erf::run((int)M, p.N1, p.C, x8, dW1, nullptr, 0, 0, h8, dV1, typename lqblk::Fc1Erf::Gemm::EpilogueOutputOp::Params(p.s1, 1.f / p.s2), gws, st);
        if (s != cutlass::Status::kSuccess) return 1;
        s = lqblk::Fc2Res::run((int)M, p.C, p.N1, h8, dW2, x, p.C, (int64_t)M * p.C, y, dV2, typename lqblk::Fc2Res::Gemm::EpilogueOutputOp::Params(p.s2, 1.f), gws, st);
        return s == cutlass::Status::kSuccess ? 0 : 1;
    }
private:
    std::vector<PluginField> f_; PluginFieldCollection fc_{};
    void upload() {
        cudaMalloc(&dW1, p.codes1.size()); cudaMemcpy(dW1, p.codes1.data(), p.codes1.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&dW2, p.codes2.size()); cudaMemcpy(dW2, p.codes2.data(), p.codes2.size(), cudaMemcpyHostToDevice);
        std::vector<unsigned> v1(p.N1), v2(p.C); for (int n = 0; n < p.N1; n++) v1[n] = lqblk::pack_sb(p.ws1[n], p.b1[n]); for (int n = 0; n < p.C; n++) v2[n] = lqblk::pack_sb(p.ws2[n], p.b2[n]);
        cudaMalloc(&dV1, p.N1 * 4); cudaMemcpy(dV1, v1.data(), p.N1 * 4, cudaMemcpyHostToDevice); cudaMalloc(&dV2, p.C * 4); cudaMemcpy(dV2, v2.data(), p.C * 4, cudaMemcpyHostToDevice);
        if (p.mode == 0) { cudaMalloc(&dG, p.C * 4); cudaMemcpy(dG, p.gamma.data(), p.C * 4, cudaMemcpyHostToDevice); cudaMalloc(&dB, p.C * 4); cudaMemcpy(dB, p.beta.data(), p.C * 4, cudaMemcpyHostToDevice); }
    }
};
class LQMlpCreator : public IPluginCreatorV3One {
public:
    LQMlpCreator() {
        const char* names[] = {"mode", "gelu", "eps", "s1", "s2", "gamma", "beta", "codes1", "ws1", "b1", "codes2", "ws2", "b2", "chan", "sparse"};
        PluginFieldType types[] = {PluginFieldType::kINT32, PluginFieldType::kINT32, PluginFieldType::kFLOAT32, PluginFieldType::kFLOAT32, PluginFieldType::kFLOAT32, PluginFieldType::kFLOAT32, PluginFieldType::kFLOAT32, PluginFieldType::kINT8, PluginFieldType::kFLOAT32, PluginFieldType::kFLOAT32, PluginFieldType::kINT8, PluginFieldType::kFLOAT32, PluginFieldType::kFLOAT32, PluginFieldType::kFLOAT32, PluginFieldType::kINT32};
        for (int i = 0; i < 15; i++) f_.emplace_back(names[i], nullptr, types[i], 0);
        fc_.nbFields = 15; fc_.fields = f_.data();
    }
    char const* getPluginName() const noexcept override { return kMlpName; }
    char const* getPluginVersion() const noexcept override { return kMlpVer; }
    char const* getPluginNamespace() const noexcept override { return kMlpNs; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
        MlpParams p;
        for (int i = 0; i < fc->nbFields; i++) {
            const PluginField& f = fc->fields[i]; std::string n = f.name; const float* fd = (const float*)f.data;
            if (getenv("LQ_DEBUG_FIELDS")) fprintf(stderr, "LQMlp field: %s (type %d len %d)\n", fc->fields[i].name, (int)fc->fields[i].type, fc->fields[i].length);
            if (n == "mode") p.mode = *(const int32_t*)f.data; else if (n == "gelu") p.gelu = *(const int32_t*)f.data; else if (n == "sparse") p.sparse = *(const int32_t*)f.data; else if (n == "eps") p.eps = *fd; else if (n == "s1") p.s1 = *fd; else if (n == "s2") p.s2 = *fd; else if (n == "chan") p.chan.assign(fd, fd + f.length);
            else if (n == "gamma") p.gamma.assign(fd, fd + f.length); else if (n == "beta") p.beta.assign(fd, fd + f.length);
            else if (n == "codes1") p.codes1.assign((const int8_t*)f.data, (const int8_t*)f.data + f.length); else if (n == "ws1") p.ws1.assign(fd, fd + f.length); else if (n == "b1") p.b1.assign(fd, fd + f.length);
            else if (n == "codes2") p.codes2.assign((const int8_t*)f.data, (const int8_t*)f.data + f.length); else if (n == "ws2") p.ws2.assign(fd, fd + f.length); else if (n == "b2") p.b2.assign(fd, fd + f.length);
        }
        if (p.ws1.empty() || p.ws2.empty() || p.codes1.size() != p.ws1.size() * p.ws2.size() || p.codes2.size() != p.codes1.size() || p.s1 <= 0 || p.s2 <= 0) { fprintf(stderr, "LQMlp: bad fields\n"); return nullptr; }
        p.N1 = (int)p.ws1.size(); p.C = (int)p.ws2.size();
        if (p.mode == 0 && (p.gamma.size() != (size_t)p.C || p.beta.size() != (size_t)p.C)) { fprintf(stderr, "LQMlp: gamma/beta size\n"); return nullptr; }
        return new LQMlpPlugin(std::move(p));
    }
private: std::vector<PluginField> f_; PluginFieldCollection fc_{};
};
static LQMlpCreator gMlpCreator;
extern "C" IPluginCreatorInterface* lq_mlp_creator() { return &gMlpCreator; }

// ---------------------------------------------------------------- shared pieces from lq_plugins.cu ---------------------------------------
namespace lqblk {
// RoPE-in-epilogue functor (packed cos/sin source table [M,N]; (s,b) packed per column)
template <int kElementsPerAccess_>
class RopeDequant {
public:
    using ElementOutput = cutlass::half_t; using ElementC = cutlass::half_t; using ElementAccumulator = int32_t; using ElementCompute = float;
    using ElementZ = cutlass::half_t; using ElementT = cutlass::half_t; using ElementVector = float;
    static int const kElementsPerAccess = kElementsPerAccess_; static int const kCount = kElementsPerAccess;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>; using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
    using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>; using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>; using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;
    using FragmentOutput = FragmentZ;
    static bool const kIsHeavy = false; static bool const kStoreZ = true; static bool const kStoreT = false; static bool const kIsSingleSource = true;
    struct Params { ElementCompute alpha; ElementCompute beta; ElementCompute const* alpha_ptr; ElementCompute const* beta_ptr;
        CUTLASS_HOST_DEVICE Params() : alpha(1), beta(1), alpha_ptr(nullptr), beta_ptr(nullptr) {}
        CUTLASS_HOST_DEVICE Params(ElementCompute a, ElementCompute b) : alpha(a), beta(b), alpha_ptr(nullptr), beta_ptr(nullptr) {} };
private: ElementCompute alpha_;
public:
    CUTLASS_HOST_DEVICE RopeDequant(Params const& p) : alpha_(p.alpha) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const { return true; }
    CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
    CUTLASS_DEVICE void operator()(FragmentZ& frag_Z, FragmentT&, FragmentAccumulator const& AB, FragmentC const& frag_C, FragmentCompute const& V) const {
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess> acc2f; cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess> c2f;
        FragmentCompute a = acc2f(AB), cs = c2f(frag_C), y, z;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) { unsigned u = __float_as_uint(V[i]); float s = __half2float(__ushort_as_half((unsigned short)(u & 0xffffu))); float bb = __half2float(__ushort_as_half((unsigned short)(u >> 16))); y[i] = alpha_ * a[i] * s + bb; }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; i += 2) { float c = cs[i], sn = cs[i + 1]; z[i] = y[i] * c - y[i + 1] * sn; z[i + 1] = y[i + 1] * c + y[i] * sn; }
        cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> f2z; frag_Z = f2z(z);
    }
    CUTLASS_DEVICE void operator()(FragmentZ& frag_Z, FragmentT& t, FragmentAccumulator const& AB, FragmentCompute const& V) const { FragmentC zero; zero.clear(); (*this)(frag_Z, t, AB, zero, V); }
};
using RopeG = GemmT<RopeDequant<8>, cutlass::half_t>;
// plain dequant + bias -> fp16 (v projection)
template <int kElementsPerAccess_>
class DequantBias {
public:
    using ElementOutput = cutlass::half_t; using ElementC = cutlass::half_t; using ElementAccumulator = int32_t; using ElementCompute = float;
    using ElementZ = cutlass::half_t; using ElementT = cutlass::half_t; using ElementVector = float;
    static int const kElementsPerAccess = kElementsPerAccess_; static int const kCount = kElementsPerAccess;
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>; using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
    using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>; using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>; using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;
    using FragmentOutput = FragmentZ;
    static bool const kIsHeavy = false; static bool const kStoreZ = true; static bool const kStoreT = false; static bool const kIsSingleSource = true;
    struct Params { ElementCompute alpha; ElementCompute beta; ElementCompute const* alpha_ptr; ElementCompute const* beta_ptr;
        CUTLASS_HOST_DEVICE Params() : alpha(1), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr) {}
        CUTLASS_HOST_DEVICE Params(ElementCompute a, ElementCompute b) : alpha(a), beta(b), alpha_ptr(nullptr), beta_ptr(nullptr) {} };
private: ElementCompute alpha_;
public:
    CUTLASS_HOST_DEVICE DequantBias(Params const& p) : alpha_(p.alpha) {}
    CUTLASS_HOST_DEVICE bool is_source_needed() const { return false; }
    CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
    CUTLASS_DEVICE void compute(FragmentZ& frag_Z, FragmentAccumulator const& AB, FragmentCompute const& V) const {
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess> acc2f; FragmentCompute a = acc2f(AB), z;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kElementsPerAccess; ++i) { unsigned u = __float_as_uint(V[i]); float s = __half2float(__ushort_as_half((unsigned short)(u & 0xffffu))); float bb = __half2float(__ushort_as_half((unsigned short)(u >> 16))); z[i] = alpha_ * a[i] * s + bb; }
        cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> f2z; frag_Z = f2z(z);
    }
    CUTLASS_DEVICE void operator()(FragmentZ& frag_Z, FragmentT&, FragmentAccumulator const& AB, FragmentC const&, FragmentCompute const& V) const { compute(frag_Z, AB, V); }
    CUTLASS_DEVICE void operator()(FragmentZ& frag_Z, FragmentT&, FragmentAccumulator const& AB, FragmentCompute const& V) const { compute(frag_Z, AB, V); }
};
using VG = GemmT<DequantBias<8>, cutlass::half_t>;
// packed cos/sin table cache (row m -> token m % Ntok), shared across plugin instances
__global__ void expand_table_k(const float* cosT, const float* sinT, int Ntok, int M, int N, __half* out) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; if (idx >= (long)M * N) return; int m = (int)(idx / N), n = (int)(idx % N); int t = m % Ntok, d = n % 64;
    if (n >= 2048) { out[idx] = __float2half((n & 1) ? 0.f : 1.f); return; }          // v columns: identity rotation
    out[idx] = __float2half((n & 1) ? sinT[t * 64 + d] : cosT[t * 64 + d]);
}
static std::map<std::string, __half*> g_tabs; static std::mutex g_tabs_mu;
static __half* packed_table(const std::vector<float>& cosT, const std::vector<float>& sinT, int Ntok, int M, int N) {
    char key[256]; snprintf(key, sizeof key, "%d_%d_%d_%.6f_%.6f_%.6f", Ntok, M, N, cosT[64 + 3], sinT[64 + 3], cosT[(Ntok - 1) * 64 + 61]);
    std::lock_guard<std::mutex> lk(g_tabs_mu); auto it = g_tabs.find(key); if (it != g_tabs.end()) return it->second;
    float *dc, *ds; cudaMalloc(&dc, cosT.size() * 4); cudaMalloc(&ds, sinT.size() * 4); cudaMemcpy(dc, cosT.data(), cosT.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(ds, sinT.data(), sinT.size() * 4, cudaMemcpyHostToDevice);
    __half* out; cudaMalloc(&out, (size_t)M * N * 2); long total = (long)M * N; expand_table_k<<<(unsigned)((total + 255) / 256), 256>>>(dc, ds, Ntok, M, N, out); cudaDeviceSynchronize(); cudaFree(dc); cudaFree(ds); g_tabs[key] = out; return out;
}
struct Lin { std::vector<int8_t> codes; std::vector<float> ws, b; int8_t* dW = nullptr; float* dV = nullptr;
    void upload(int N) { cudaMalloc(&dW, codes.size()); cudaMemcpy(dW, codes.data(), codes.size(), cudaMemcpyHostToDevice); std::vector<unsigned> v(N); for (int n = 0; n < N; n++) v[n] = pack_sb(ws[n], b[n]); cudaMalloc(&dV, N * 4); cudaMemcpy(dV, v.data(), N * 4, cudaMemcpyHostToDevice); }
    void free() { if (dW) cudaFree(dW); if (dV) cudaFree(dV); dW = nullptr; dV = nullptr; } };
static void parse_lin(Lin& L, const std::string& n, const std::string& pre, const PluginField& f) {
    if (n == pre + "codes") L.codes.assign((const int8_t*)f.data, (const int8_t*)f.data + f.length); else if (n == pre + "ws") L.ws.assign((const float*)f.data, (const float*)f.data + f.length); else if (n == pre + "b") L.b.assign((const float*)f.data, (const float*)f.data + f.length);
}
}  // namespace lqblk

// ---------------------------------------------------------------- common plugin boilerplate ---------------------------------------------
#define LQ_COMMON_BUILD(NOUT)                                                                                                   \
    IPluginCapability* getCapabilityInterface(PluginCapabilityType t) noexcept override {                                     \
        if (t == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);                                  \
        if (t == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);                              \
        return static_cast<IPluginV3OneCore*>(this); }                                                                        \
    int32_t getNbOutputs() const noexcept override { return NOUT; }                                                          \
    int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override { return 0; } \
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t, int32_t) noexcept override { return io[pos].desc.format == TensorFormat::kLINEAR && io[pos].desc.type == DataType::kHALF; } \
    int32_t getOutputDataTypes(DataType* out, int32_t nb, DataType const*, int32_t) const noexcept override { for (int i = 0; i < nb; i++) out[i] = DataType::kHALF; return 0; } \
    int32_t getValidTactics(int32_t*, int32_t) noexcept override { return 0; }                                               \
    int32_t getNbTactics() noexcept override { return 0; }                                                                    \
    char const* getTimingCacheID() noexcept override { return nullptr; }                                                      \
    int32_t getFormatCombinationLimit() noexcept override { return 1; }                                                       \
    char const* getMetadataString() noexcept override { return nullptr; }                                                     \
    int32_t setTactic(int32_t) noexcept override { return 0; }                                                                \
    int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override { return 0; } \
    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }

// ---------------------------------------------------------------- LQQkvRope: x -> (q_rot, k_rot, v) ----------------------------------------
static const char* kQkvName = "LQQkvRope";
struct QkvParams { int mode = 1; float eps = 1e-6f, s = 0.f; std::vector<float> gamma, beta, cosT, sinT; lqblk::Lin q, k, v, qkv; int C = 1024, Ntok = 0; };
class LQQkvPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    QkvParams p; float* dG = nullptr; float* dB = nullptr; __half* dTab = nullptr; int tabM = 0;
    explicit LQQkvPlugin(QkvParams pp) : p(std::move(pp)) {}
    ~LQQkvPlugin() override { p.q.free(); p.k.free(); p.v.free(); if (dG) cudaFree(dG); if (dB) cudaFree(dB); /* dTab is owned by the shared cache g_tabs (many plugin instances and later reloads use it): never free it here */ }
    LQ_COMMON_BUILD(3)
    int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t, DimsExprs* out, int32_t nb, IExprBuilder&) noexcept override { for (int i = 0; i < nb; i++) out[i] = in[0]; return 0; }
    IPluginV3* clone() noexcept override { return new LQQkvPlugin(p); }
    char const* getPluginName() const noexcept override { return kQkvName; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override {
        int64_t M = 1; for (int i = 0; i < in[0].desc.dims.nbDims - 1; i++) M *= in[0].desc.dims.d[i] > 0 ? in[0].desc.dims.d[i] : 1;
        #ifdef LQ_BATCHED_TABLES
        return (size_t)M * p.C + lqblk::RopeG::workspace_batched(p.Ntok, (int)(M / p.Ntok), p.C, p.C) + lqblk::VG::workspace((int)M, p.C, p.C) + (256 << 10);
#else
        return (size_t)M * p.C + lqblk::RopeG::workspace((int)M, p.C, p.C) + lqblk::VG::workspace((int)M, p.C, p.C) + (256 << 10);
#endif
    }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        f_.clear(); f_.emplace_back("mode", &p.mode, PluginFieldType::kINT32, 1); f_.emplace_back("eps", &p.eps, PluginFieldType::kFLOAT32, 1); f_.emplace_back("s", &p.s, PluginFieldType::kFLOAT32, 1);
        f_.emplace_back("gamma", p.gamma.data(), PluginFieldType::kFLOAT32, (int32_t)p.gamma.size()); f_.emplace_back("beta", p.beta.data(), PluginFieldType::kFLOAT32, (int32_t)p.beta.size());
        f_.emplace_back("cos", p.cosT.data(), PluginFieldType::kFLOAT32, (int32_t)p.cosT.size()); f_.emplace_back("sin", p.sinT.data(), PluginFieldType::kFLOAT32, (int32_t)p.sinT.size());
        const char* pre[3] = {"q_", "k_", "v_"}; lqblk::Lin* L[3] = {&p.q, &p.k, &p.v};
        for (int i = 0; i < 3; i++) { names_[i * 3] = std::string(pre[i]) + "codes"; names_[i * 3 + 1] = std::string(pre[i]) + "ws"; names_[i * 3 + 2] = std::string(pre[i]) + "b";
            f_.emplace_back(names_[i * 3].c_str(), L[i]->codes.data(), PluginFieldType::kINT8, (int32_t)L[i]->codes.size()); f_.emplace_back(names_[i * 3 + 1].c_str(), L[i]->ws.data(), PluginFieldType::kFLOAT32, (int32_t)L[i]->ws.size()); f_.emplace_back(names_[i * 3 + 2].c_str(), L[i]->b.data(), PluginFieldType::kFLOAT32, (int32_t)L[i]->b.size()); }
        fc_.nbFields = (int32_t)f_.size(); fc_.fields = f_.data(); return &fc_;
    }
    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const*, void const* const* inputs, void* const* outputs, void* ws, cudaStream_t st) noexcept override {
        int64_t M = 1; for (int i = 0; i < in[0].dims.nbDims - 1; i++) M *= in[0].dims.d[i];
        if (!p.q.dW) { p.q.upload(p.C); p.k.upload(p.C); p.v.upload(p.C); if (p.mode == 0) { cudaMalloc(&dG, p.C * 4); cudaMemcpy(dG, p.gamma.data(), p.C * 4, cudaMemcpyHostToDevice); cudaMalloc(&dB, p.C * 4); cudaMemcpy(dB, p.beta.data(), p.C * 4, cudaMemcpyHostToDevice); } }
#ifdef LQ_BATCHED_TABLES   // variant: shared [Ntok, C] table, GEMMs batched over windows
        if (!dTab) { dTab = lqblk::packed_table(p.cosT, p.sinT, p.Ntok, p.Ntok, p.C); }
#else                      // default: full [M, C] table replicated per window, one GEMM per projection
        if (!dTab || tabM != (int)M) { dTab = lqblk::packed_table(p.cosT, p.sinT, p.Ntok, (int)M, p.C); tabM = (int)M; }      // cache-owned pointer: not freed by the plugin
#endif
        const __half* x = (const __half*)inputs[0]; int8_t* x8 = (int8_t*)ws; char* gws = (char*)(x8 + (size_t)M * p.C);
        if (p.mode == 0) lqblk::norm_quant_kernel<0><<<(unsigned)((M + 7) / 8), 256, 0, st>>>(x, x8, dG, dB, p.eps, 1.f / p.s, (int)M);
        else lqblk::norm_quant_kernel<1><<<(unsigned)((M + 7) / 8), 256, 0, st>>>(x, x8, nullptr, nullptr, p.eps, 1.f / p.s, (int)M);
        typename lqblk::RopeG::Gemm::EpilogueOutputOp::Params rp(p.s, 1.f);
#ifdef LQ_BATCHED_TABLES
        if (lqblk::RopeG::run_batched(p.Ntok, (int)(M / p.Ntok), p.C, p.C, x8, p.q.dW, dTab, outputs[0], p.q.dV, rp, gws, st) != cutlass::Status::kSuccess) return 1;
        if (lqblk::RopeG::run_batched(p.Ntok, (int)(M / p.Ntok), p.C, p.C, x8, p.k.dW, dTab, outputs[1], p.k.dV, rp, gws, st) != cutlass::Status::kSuccess) return 1;
#else
        if (lqblk::RopeG::run((int)M, p.C, p.C, x8, p.q.dW, dTab, p.C, 0, outputs[0], p.q.dV, rp, gws, st) != cutlass::Status::kSuccess) return 1;
        if (lqblk::RopeG::run((int)M, p.C, p.C, x8, p.k.dW, dTab, p.C, 0, outputs[1], p.k.dV, rp, gws, st) != cutlass::Status::kSuccess) return 1;
#endif
        typename lqblk::VG::Gemm::EpilogueOutputOp::Params vp(p.s, 0.f);
        return lqblk::VG::run((int)M, p.C, p.C, x8, p.v.dW, nullptr, 0, 0, outputs[2], p.v.dV, vp, gws, st) == cutlass::Status::kSuccess ? 0 : 1;
    }
private: std::vector<PluginField> f_; PluginFieldCollection fc_{}; std::string names_[9];
};
class LQQkvCreator : public IPluginCreatorV3One {
public:
    LQQkvCreator() { const char* n[] = {"mode", "eps", "s", "gamma", "beta", "cos", "sin", "q_codes", "q_ws", "q_b", "k_codes", "k_ws", "k_b", "v_codes", "v_ws", "v_b"};
        for (int i = 0; i < 16; i++) f_.emplace_back(n[i], nullptr, (std::string(n[i]).find("codes") != std::string::npos) ? PluginFieldType::kINT8 : (i == 0 ? PluginFieldType::kINT32 : PluginFieldType::kFLOAT32), 0);
        fc_.nbFields = 16; fc_.fields = f_.data(); }
    char const* getPluginName() const noexcept override { return kQkvName; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
        QkvParams p;
        for (int i = 0; i < fc->nbFields; i++) { const PluginField& f = fc->fields[i]; std::string n = f.name; const float* fd = (const float*)f.data;
            if (n == "mode") p.mode = *(const int32_t*)f.data; else if (n == "eps") p.eps = *fd; else if (n == "s") p.s = *fd; else if (n == "gamma") p.gamma.assign(fd, fd + f.length); else if (n == "beta") p.beta.assign(fd, fd + f.length);
            else if (n == "cos") p.cosT.assign(fd, fd + f.length); else if (n == "sin") p.sinT.assign(fd, fd + f.length);
            else { lqblk::parse_lin(p.q, n, "q_", f); lqblk::parse_lin(p.k, n, "k_", f); lqblk::parse_lin(p.v, n, "v_", f); } }
        if (p.s <= 0 || p.q.ws.size() != 1024 || p.k.ws.size() != 1024 || p.v.ws.size() != 1024 || p.cosT.size() % 64 != 0 || p.cosT.empty()) { fprintf(stderr, "LQQkvRope: bad fields\n"); return nullptr; }
        p.Ntok = (int)(p.cosT.size() / 64); return new LQQkvPlugin(std::move(p));
    }
private: std::vector<PluginField> f_; PluginFieldCollection fc_{};
};
static LQQkvCreator gQkvCreator;
extern "C" IPluginCreatorInterface* lq_qkv_creator() { return &gQkvCreator; }

// ---------------------------------------------------------------- LQAttnProj: (q_rot, k_rot, v, r) -> r + proj(attn(q,k,v)) ---------------
static const char* kApName = "LQAttnProj";
struct ApParams { int window = 576, heads = 16, tmix = 0; float scale = 0.125f, s = 0.f; lqblk::Lin proj; int C = 1024; float* dS = nullptr; float* dB = nullptr; };
class LQAttnProjPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    ApParams p;
    explicit LQAttnProjPlugin(ApParams pp) : p(std::move(pp)) {}
    ~LQAttnProjPlugin() override { p.proj.free(); }
    LQ_COMMON_BUILD(1)
    int32_t getOutputShapes(DimsExprs const* in, int32_t, DimsExprs const*, int32_t, DimsExprs* out, int32_t, IExprBuilder&) noexcept override { out[0] = in[3]; return 0; }
    IPluginV3* clone() noexcept override { return new LQAttnProjPlugin(p); }
    char const* getPluginName() const noexcept override { return kApName; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override {
        int64_t M = 1; for (int i = 0; i < in[3].desc.dims.nbDims - 1; i++) M *= in[3].desc.dims.d[i] > 0 ? in[3].desc.dims.d[i] : 1;
        return (size_t)M * p.C + lqblk::Fc2Res::workspace((int)M, p.C, p.C) + (256 << 10);
    }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        f_.clear(); f_.emplace_back("window", &p.window, PluginFieldType::kINT32, 1); f_.emplace_back("heads", &p.heads, PluginFieldType::kINT32, 1); f_.emplace_back("scale", &p.scale, PluginFieldType::kFLOAT32, 1); f_.emplace_back("s", &p.s, PluginFieldType::kFLOAT32, 1); f_.emplace_back("tmix", &p.tmix, PluginFieldType::kINT32, 1);
        f_.emplace_back("proj_codes", p.proj.codes.data(), PluginFieldType::kINT8, (int32_t)p.proj.codes.size()); f_.emplace_back("proj_ws", p.proj.ws.data(), PluginFieldType::kFLOAT32, (int32_t)p.proj.ws.size()); f_.emplace_back("proj_b", p.proj.b.data(), PluginFieldType::kFLOAT32, (int32_t)p.proj.b.size());
        fc_.nbFields = (int32_t)f_.size(); fc_.fields = f_.data(); return &fc_;
    }
    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const*, void const* const* inputs, void* const* outputs, void* ws, cudaStream_t st) noexcept override {
        int64_t M = 1; for (int i = 0; i < in[3].dims.nbDims - 1; i++) M *= in[3].dims.d[i];
        if (!p.proj.dW) p.proj.upload(p.C);
        const __half* q = (const __half*)inputs[0]; const __half* k = (const __half*)inputs[1]; const __half* v = (const __half*)inputs[2]; const __half* r = (const __half*)inputs[3];
        int8_t* a8 = (int8_t*)ws; char* gws = (char*)(a8 + (size_t)M * p.C); cudaError_t e;
        if (p.tmix == 64) {   // token-mixed proj input: attention epilogue mixes 64-token groups before INT8, the proj GEMM un-mixes its output rows
            if (!p.dS) { std::vector<float> sv(p.C); for (int n = 0; n < p.C; n++) sv[n] = p.s * p.proj.ws[n]; cudaMalloc(&p.dS, p.C * 4); cudaMemcpy(p.dS, sv.data(), p.C * 4, cudaMemcpyHostToDevice); cudaMalloc(&p.dB, p.C * 4); cudaMemcpy(p.dB, p.proj.b.data(), p.C * 4, cudaMemcpyHostToDevice); }
            e = lqattn::launch_pp<64, true, 64, false, true>(q, k, v, a8, (int)M, p.window, p.heads, p.scale, st, p.s); if (e != cudaSuccess) return 1;
            lqmix::EpiParams ep{p.dS, p.dB, nullptr, r, p.C, 1, 0, 0};
            return lqmix::launch<lqmix::kFp16Residual>((int)M, p.C, p.C, a8, p.C, p.proj.dW, p.C, outputs[0], p.C, ep, st) == cudaSuccess ? 0 : 1;
        }
        e = lqattn::launch_pp<64, true, 64>(q, k, v, a8, (int)M, p.window, p.heads, p.scale, st, p.s);   // software-pipelined kernel, INT8 output
        if (e != cudaSuccess) return 1;
        typename lqblk::Fc2Res::Gemm::EpilogueOutputOp::Params rp(p.s, 1.f);
        return lqblk::Fc2Res::run((int)M, p.C, p.C, a8, p.proj.dW, r, p.C, (int64_t)M * p.C, outputs[0], p.proj.dV, rp, gws, st) == cutlass::Status::kSuccess ? 0 : 1;
    }
private: std::vector<PluginField> f_; PluginFieldCollection fc_{};
};
class LQAttnProjCreator : public IPluginCreatorV3One {
public:
    LQAttnProjCreator() { f_ = {PluginField("window", nullptr, PluginFieldType::kINT32, 1), PluginField("heads", nullptr, PluginFieldType::kINT32, 1), PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1), PluginField("s", nullptr, PluginFieldType::kFLOAT32, 1),
                                PluginField("proj_codes", nullptr, PluginFieldType::kINT8, 0), PluginField("proj_ws", nullptr, PluginFieldType::kFLOAT32, 0), PluginField("proj_b", nullptr, PluginFieldType::kFLOAT32, 0), PluginField("tmix", nullptr, PluginFieldType::kINT32, 1)}; fc_.nbFields = 8; fc_.fields = f_.data(); }
    char const* getPluginName() const noexcept override { return kApName; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
        ApParams p;
        for (int i = 0; i < fc->nbFields; i++) { const PluginField& f = fc->fields[i]; std::string n = f.name; const float* fd = (const float*)f.data;
            if (n == "window") p.window = *(const int32_t*)f.data; else if (n == "heads") p.heads = *(const int32_t*)f.data; else if (n == "tmix") p.tmix = *(const int32_t*)f.data; else if (n == "scale") p.scale = *fd; else if (n == "s") p.s = *fd; else lqblk::parse_lin(p.proj, n, "proj_", f); }
        if (p.s <= 0 || p.proj.ws.size() != 1024 || p.proj.codes.size() != 1024 * 1024) { fprintf(stderr, "LQAttnProj: bad fields\n"); return nullptr; }
        return new LQAttnProjPlugin(std::move(p));
    }
private: std::vector<PluginField> f_; PluginFieldCollection fc_{};
};
static LQAttnProjCreator gApCreator;
extern "C" IPluginCreatorInterface* lq_attnproj_creator() { return &gApCreator; }
