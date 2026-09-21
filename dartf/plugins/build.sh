#!/usr/bin/env bash
# Build lq_plugins.so (TensorRT 10 IPluginV3 plugins) — usage: ARCH=87 ./build.sh   (80/86/87/89/90)
set -euo pipefail
cd "$(dirname "$0")"
ARCH=${ARCH:-87}
CUTLASS=${CUTLASS:-../third_party/cutlass/include}
TRT_INC=${TRT_INC:-/usr/include/aarch64-linux-gnu}       # x86: /usr/include or $TENSORRT_ROOT/include
TRT_LIB=${TRT_LIB:-/usr/lib/aarch64-linux-gnu}
NVCC=${NVCC:-$(ls /usr/local/cuda*/bin/nvcc | head -1)}
case "$ARCH" in 86|89) STAGES=${STAGES:-3};; *) STAGES=${STAGES:-5};; esac   # 99 KB smem per block on sm86/sm89
[ -f "$CUTLASS/cutlass/cutlass.h" ] || { echo "CUTLASS headers not found at $CUTLASS (see ../third_party/README.md)"; exit 1; }
cp ../kernels/lq_attn_kernel.cuh . 2>/dev/null || true
$NVCC -O3 -std=c++17 -arch=sm_$ARCH --use_fast_math --expt-relaxed-constexpr -DLQ_STAGES=$STAGES -Xcompiler -fPIC -shared -I "$CUTLASS" -I "$TRT_INC" \
  lq_plugins.cu lq_attn_plugin.cu lq_block_plugins.cu lq_ground_plugins.cu lq_memattn_plugin.cu -o lq_plugins.so -L "$TRT_LIB" -lnvinfer -lcublasLt -lcublas
echo "built plugins/lq_plugins.so for sm_$ARCH"
python3 check_plugin.py "$(pwd)/lq_plugins.so" || true

# torch-free tracker helpers (row gather / transpose kernels), loaded through ctypes by demo/sam3_track_np.py
$NVCC -O3 -arch=sm_$ARCH -Xcompiler -fPIC -shared lq_util.cu -o lq_util.so && echo "built plugins/lq_util.so"
