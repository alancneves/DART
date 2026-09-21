"""DARTF demo on a video: W8A8 vision backbone (TensorRT + LQ plugins) -> detector head (FP16, K=4 prompt bucket, emits queries and
encoder states) -> segmentation head (FP16, dynamic prompt axis, top-Q queries) -> lightweight tracker -> boxes + masks at 1920x1080.
The text encoder runs once per prompt set (cached); the detector head is image conditioned and runs per frame.
usage: LQ_PLUGINS=plugins/lq_plugins.so python run_video.py video.mp4 out.mp4 "lemon,strawberry" --assets <dir> [--heads masks|boxes] [--tracker none|light|sam3] [--pipeline]
       headless / benchmark: run_video.py "seq/img1/%06d.jpg" out.mp4 person --no-render --mot-out tracks.txt --fps-in 30 --src-size 1920x1080   (no ffmpeg or torch needed)
<dir> holds vision_int8.plan (built with --mark-outputs '^permute_4$' when --tracker sam3 is used), ground_c4m_fp16.plan (or ground_c1m for one prompt), maskhead_q32_phase_fp16.plan,
img_pos_c4.npy, text_c16.onnx(+.data), bpe_simple_vocab_16e6.txt.gz and, for --tracker sam3, trk_neck_fp16.plan / trk_init_fp16.plan (torch driver) or trk_init288_fp16.plan (torch-free driver) /
trk_step_v2_fp16.plan (+ pe_mem.npy, tpos_enc.npy from export_tracker.py; plugins/lq_util.so for the torch-free driver). Needs tensorrt, onnxruntime, scipy, pillow; rendering needs ffmpeg,
opencv and CUDA torch; --tracker sam3 works with or without torch (the torch-free driver is used automatically when torch is absent, or with --sam3-np)."""
import sys, os, json, time, subprocess, argparse, numpy as np
HR = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HR); sys.path.insert(0, os.path.join(HR, "..", "runtime"))
import tensorrt as trt
from scipy.optimize import linear_sum_assignment
from trt_util import load_engine, Runner, TRT_LOGGER, NP_DTYPE
NP_DTYPE = dict(NP_DTYPE); NP_DTYPE[trt.int64] = np.int64
from cuda_alloc import DevBuf, Stream, Event, PinnedBuf, memcpy_async, H2D, D2D
import threading, queue
from tokenize_classes import Sam3Tokenizer, CONTEXT_LENGTH, BUCKET
from PIL import Image, ImageDraw, ImageFont
try:
    import torch, torch.nn.functional as Fn; HAVE_TORCH = torch.cuda.is_available() and not os.environ.get("LQ_NO_TORCH")
except Exception: HAVE_TORCH = False
if not HAVE_TORCH:
    class _NP:                                                         # minimal stand-in: masks are numpy arrays, "device" tensors are numpy
        @staticmethod
        def as_tensor(x, device=None): return np.asarray(x)
        @staticmethod
        def from_numpy(x): return np.asarray(x)
        @staticmethod
        def stack(xs, dim=0): return np.stack([np.asarray(x) for x in xs], dim)
        @staticmethod
        def device(_): return None
        @staticmethod
        def tensor(x, device=None): return np.asarray(x, np.float32)
        @staticmethod
        def minimum(a, b): return np.minimum(a, b)
        @staticmethod
        def sigmoid(x): return 1 / (1 + np.exp(-x))
    torch = _NP()
    class Fn:
        @staticmethod
        def interpolate(x, size=None, mode="bilinear", align_corners=False):
            import cv2; a = np.asarray(x)[0, 0]; return cv2.resize(a, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)[None, None]
ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("out"); ap.add_argument("prompts"); ap.add_argument("--assets", default="assets")
ap.add_argument("--vision", default="vision_int8.plan"); ap.add_argument("--ground", default="ground_c4m_fp16.plan"); ap.add_argument("--mask", default="maskhead_q32_phase_fp16.plan")
ap.add_argument("--groundmask", default=None, help="fused grounding + segmentation engine (groundmask_cK_q32_phase_fp16.plan): replaces --ground and --mask; fpn_0/1/2 stay on the device, top-32 query selection happens in the graph")
ap.add_argument("--text-onnx", default="text_c16.onnx"); ap.add_argument("--img-pos", default="img_pos_c4.npy"); ap.add_argument("--bpe", default="bpe_simple_vocab_16e6.txt.gz")
ap.add_argument("--font-dir", default="/usr/share/fonts/truetype/lato"); ap.add_argument("--title", default="DARTF: Detect Anything in Real Time Faster"); ap.add_argument("--name", default=""); ap.add_argument("--affiliation", default="")
ap.add_argument("--max-frames", type=int, default=0); ap.add_argument("--thr", type=float, default=0.5, help="score to start a track"); ap.add_argument("--low", type=float, default=0.25, help="score to continue a track")
ap.add_argument("--nms", type=float, default=0.6); ap.add_argument("--app-w", type=float, default=0.3, help="weight of query-embedding cosine similarity in the association (0 = geometry only)"); ap.add_argument("--motion", type=int, default=1, help="constant velocity box prediction for association"); ap.add_argument("--fps-out", type=int, default=30); ap.add_argument("--gpu-name", default="RTX 4090")
ap.add_argument("--tracker", choices=["none", "light", "sam3", "hybrid"], default="light", help="none: per-frame detections only; light: lightweight tracker (mask IoU + motion + query embeddings); sam3: light plus SAM 3 native propagation for tracks the detector misses (needs the trk_* engines and a vision engine exposing permute_4)")
ap.add_argument("--heads", choices=["masks", "boxes"], default="masks", help="boxes: skip the segmentation head (boxes only, a few ms per frame less)")
ap.add_argument("--sam3-tracker", action="store_true", help="same as --tracker hybrid"); ap.add_argument("--sam3-assoc", type=float, default=0.5, help="sam3 tracker: mask IoU for matching a detection to a propagated object (SAM3 assoc_iou_thresh)"); ap.add_argument("--sam3-np", action="store_true", help="torch-free SAM 3 driver (raw CUDA buffers + lq_util kernels); automatic when torch is absent"); ap.add_argument("--util-so", default=None, help="lq_util shared library (default: next to LQ_PLUGINS)"); ap.add_argument("--trk-dir", default=f"{HR}/exp/trk"); ap.add_argument("--vision-trunk", default=f"{HR}/exp/b0a8/vision_b0a8_trunk_sm89.plan")
ap.add_argument("--refresh", type=int, default=6, help="frames between SAM 3 conditioning refreshes of a seen track")
ap.add_argument("--refresh-max", type=int, default=24, help="optional change-gated refresh (off by default): with --refresh-iou below 1, a memory is refreshed only when the detector mask moved (IoU below --refresh-iou vs the last refresh), and at least every --refresh-max frames"); ap.add_argument("--refresh-iou", type=float, default=1.0, help="1 (default) = fixed cadence of --refresh; below 1 enables the change gate"); ap.add_argument("--max-miss", type=int, default=8, help="longest gap the SAM 3 propagation bridges")
ap.add_argument("--sam3-budget", type=int, default=4, help="max objects propagated per frame"); ap.add_argument("--sam3-v1", action="store_true", help="use the first-generation trk_step engine instead of trk_step_v2 (gathered keys, default when the v2 plan exists)"); ap.add_argument("--sam3-prune", type=int, default=4, help="v2: keep memory keys within this dilation (72-grid cells) of the object mask plus a background grid; 0 = all keys"); ap.add_argument("--sam3-adaptive", action="store_true", help="3 memory frames for stable objects, 7 otherwise"); ap.add_argument("--sam3-mem", type=int, default=7, help="max memory frames per step (conditioning + recent)"); ap.add_argument("--sam3-min-hits", type=int, default=None, help="track confirmation: consecutive matched detections before a track is shown (default 3 for --tracker sam3, SAM3's masklet confirmation; 8 for hybrid)"); ap.add_argument("--sam3-min-iou", type=float, default=0.0, help="stop propagating a track once the predicted mask IoU falls below this")
ap.add_argument("--sam3-qprune", type=int, default=0, help="optional query-side pruning (off by default, changes the step): with trk_step_v3, only queries within this many cells of the object mask get the memory update"); ap.add_argument("--sam3-qgrid", type=int, default=0, help="v3 engines: background query grid stride (0 = none)"); ap.add_argument("--sam3-qfull", type=int, default=0, help="v3 engines: run a full (all-query) step every this many steps per object; its object score is carried through the pruned steps (0 = never)")
ap.add_argument("--stats", default=None); ap.add_argument("--snap", default=None); ap.add_argument("--mot-out", default=None, help="write MOT-format tracks (frame,id,x,y,w,h,conf,-1,-1,-1) at the source resolution; with --no-render no video is written")
ap.add_argument("--text-cache", default=None, help="text-embedding cache (.npz); default assets/text_<prompts>.npz. Computed once per prompt set (python demo/text_cache.py \"a,b\"), then the text encoder is never loaded"); ap.add_argument("--gpu-decode", action="store_true", help="image input (sequence pattern, directory or glob; headless, needs --no-render): decode JPEGs with nvJPEG, resize and normalize on the GPU in a prefetch thread, off the inference path"); ap.add_argument("--no-render", action="store_true"); ap.add_argument("--pipeline", action="store_true", help="two-stream frame pipeline: decode/preprocess in a thread, the backbone of frame t+1 runs on its own CUDA stream while the head, tracker and renderer process frame t"); ap.add_argument("--mot-box", choices=["det", "mask"], default="det", help="box written to --mot-out: the detector box (smoothed) or the mask extent"); ap.add_argument("--src-size", default=None, help="WxH of the source frames (image sequences)"); ap.add_argument("--fps-in", type=float, default=None, help="frame rate of an image-sequence input"); ap.add_argument("--no-lower-third", action="store_true"); ap.add_argument("--lt-slide", action="store_true", help="slide the lower third in (first clip); otherwise it fades in")
ap.add_argument("--duration", type=float, default=15.0, help="seconds of source video to process"); ap.add_argument("--no-swipe", action="store_true", help="disable the with/without wipe")
ap.add_argument("--lt-in", type=float, default=0.5); ap.add_argument("--lt-out", type=float, default=8.0)
a = ap.parse_args()
for k in ("vision", "ground", "mask", "text_onnx", "img_pos", "bpe"):
    if not os.path.isabs(getattr(a, k)): setattr(a, k, os.path.join(a.assets, getattr(a, k)))
a.trk_dir = a.trk_dir or a.assets; a.vision_trunk = a.vision_trunk or a.vision
if a.sam3_tracker: a.tracker = "sam3"
if a.sam3_tracker: a.tracker = "hybrid"
a.sam3_tracker = a.tracker in ("sam3", "hybrid")
if a.sam3_min_hits is None: a.sam3_min_hits = 3 if a.tracker == "sam3" else 8
if a.heads == "boxes" and a.sam3_tracker: raise SystemExit("--tracker sam3/hybrid needs masks (it is initialized from the detector masks); use --heads masks")
OW, OH = 1920, 1080; QSEL = 32
PAL = [(0, 200, 255), (255, 196, 0), (0, 230, 118), (255, 82, 82), (171, 71, 188), (255, 145, 0), (29, 233, 182), (236, 64, 122)]
def F(name, size):
    try: return ImageFont.truetype(os.path.join(a.font_dir, f"Lato-{name}.ttf"), size)
    except OSError: return ImageFont.load_default()
FONT_LAB = F("Semibold", 26); FONT_LAB_S = F("Semibold", 21); FONT_T1 = F("Bold", 40); FONT_T2 = F("Semibold", 30); FONT_T3 = F("Regular", 25); FONT_INFO = F("Regular", 24); FONT_INFO_B = F("Semibold", 24)
prompts = [p.strip() for p in a.prompts.split(",") if p.strip()]; K = len(prompts); assert 1 <= K <= 16

# ---- text encoder: once per prompt set (onnxruntime CPU; rows are independent, so slicing to K is exact) ----
from text_cache import text_embeddings, default_cache
tf, tm = text_embeddings(prompts, a.text_onnx, a.bpe, a.text_cache or default_cache(a.assets, prompts))
tfK = np.ascontiguousarray(tf[:, :K, :]).astype(np.float16); tmK = np.ascontiguousarray(tm[:K]).astype(np.float32)

# ---- engines ----
if a.sam3_tracker: a.vision = a.vision_trunk
vision = Runner(load_engine(a.vision)); img_pos = np.load(a.img_pos)
if a.groundmask: ground = None; fused_eng = load_engine(a.groundmask); KB = int(fused_eng.get_tensor_shape("text_feats")[1]); assert a.heads == "masks", "--groundmask computes masks"
else: ground = Runner(load_engine(a.ground)); KB = int(ground.bufs["img_feat"][1][0])   # prompt bucket of the head engine
assert K <= KB, f"{K} prompts but the head bucket is {KB}"; assert img_pos.shape[0] == KB, "img_pos does not match the head bucket"
tf4 = np.ascontiguousarray(tf[:, :KB, :]).astype(np.float16); tm4 = np.ascontiguousarray(tm[:KB]).astype(np.float32)
for i in range(K, KB): tm4[i, :] = tm[BUCKET - 1]; tf4[:, i, :] = tf[:, BUCKET - 1, :]                 # padded rows: empty prompt
fpn0_name, fpn1_name, fpn2_name = vision.alias.get("fpn_0", "fpn_0"), vision.alias.get("fpn_1", "fpn_1"), vision.alias.get("fpn_2", "fpn_2")   # engines with other output names (e.g. the DART FP16 backbone: conv2d_*)
class DynRunner:
    def __init__(self, engine):
        self.engine = engine; self.ctx = engine.create_execution_context(); self.names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.inputs = [n for n in self.names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]; self.outputs = [n for n in self.names if n not in self.inputs]; self.bufs = {}
        for n in self.inputs:
            shape = engine.get_tensor_profile_shape(n, 0)[2] if -1 in tuple(engine.get_tensor_shape(n)) else tuple(engine.get_tensor_shape(n))
            dt = NP_DTYPE[engine.get_tensor_dtype(n)]; self.bufs[n] = (DevBuf(int(np.prod(shape)) * np.dtype(dt).itemsize), dt); self.ctx.set_input_shape(n, shape)
        for n in self.outputs: shape = tuple(self.ctx.get_tensor_shape(n)); dt = NP_DTYPE[engine.get_tensor_dtype(n)]; self.bufs[n] = (DevBuf(int(np.prod(shape)) * np.dtype(dt).itemsize), dt)
        for n in self.names: self.ctx.set_tensor_address(n, self.bufs[n][0].ptr)
        self.external = {}
    def bind(self, name, ptr, shape): self.external[name] = ptr; self.ctx.set_tensor_address(name, ptr); self.ctx.set_input_shape(name, shape)
    def __call__(self, feeds):
        for n in self.inputs:
            if n in self.external: continue
            x = np.ascontiguousarray(feeds[n], dtype=self.bufs[n][1]); self.ctx.set_input_shape(n, tuple(x.shape)); self.bufs[n][0].upload(x)
        assert self.ctx.execute_async_v3(0); DevBuf.sync()
        return {n: self.bufs[n][0].download(tuple(self.ctx.get_tensor_shape(n)), self.bufs[n][1]) for n in self.outputs}
mask = None
if a.heads == "masks":
    if a.groundmask: fused = DynRunner(fused_eng)
    mask = fused if a.groundmask else DynRunner(load_engine(a.mask)); mask.bind("fpn_0", vision.bufs[fpn0_name][0].ptr, tuple(vision.bufs[fpn0_name][1])); mask.bind("fpn_1", vision.bufs[fpn1_name][0].ptr, tuple(vision.bufs[fpn1_name][1]))
    if a.groundmask: mask.bind("fpn_2", vision.bufs[fpn2_name][0].ptr, tuple(vision.bufs[fpn2_name][1]))

# ---- tracker: mask IoU Hungarian matching, low score continuation, coasting, EMA smoothing, class voting ----
dev = torch.device("cuda")
class Track:
    __slots__ = ("id", "box", "mask", "votes", "hits", "misses", "p", "emb", "vel", "det_row")
    def __init__(self, tid, det): self.id = tid; self.box = det["box"].copy(); self.mask = None if det["mask"] is None else (det["mask"].clone() if HAVE_TORCH else det["mask"].copy()); self.votes = np.zeros(K); self.votes[det["k"]] = det["p"]; self.hits = 1; self.misses = 0; self.p = det["p"]; self.emb = det["emb"].copy(); self.vel = np.zeros(4); self.det_row = det.get("row_ptr")
    def update(self, det):
        nb = det["box"]; self.vel = 0.6 * self.vel + 0.4 * (nb - self.box); self.box = 0.5 * self.box + 0.5 * nb; self.mask = None if det["mask"] is None else 0.3 * self.mask + 0.7 * det["mask"]; self.votes[det["k"]] += det["p"]; self.hits += 1; self.misses = 0; self.p = 0.7 * self.p + 0.3 * det["p"]
        e = 0.7 * self.emb + 0.3 * det["emb"]; self.emb = e / (np.linalg.norm(e) + 1e-6); self.det_row = det.get("row_ptr")
    @property
    def cls(self): return int(np.argmax(self.votes))
    @property
    def pred_box(self): return self.box + self.vel * (self.misses + 1) if a.motion else self.box
def iou_matrix(A, B):
    if len(A) == 0 or len(B) == 0 or any(x is None for x in A) or any(x is None for x in B): return np.zeros((len(A), len(B)))
    if not HAVE_TORCH:
        A = (np.stack(A) > 0).reshape(len(A), -1).astype(np.float32); B = (np.stack(B) > 0).reshape(len(B), -1).astype(np.float32); inter = A @ B.T; return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-6)
    A = (torch.stack(A) > 0).float().flatten(1); B = (torch.stack(B) > 0).float().flatten(1); inter = A @ B.t(); return (inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-6)).cpu().numpy()
def box_iou_matrix(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64); lt = np.maximum(A[:, None, :2], B[None, :, :2]); rb = np.minimum(A[:, None, 2:], B[None, :, 2:]); wh = np.clip(rb - lt, 0, None); inter = wh[..., 0] * wh[..., 1]
    ar = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); br = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]); return inter / np.maximum(ar[:, None] + br[None, :] - inter, 1e-6)
def match(tracks, dets, min_iou):
    """affinity = geometry (mask IoU, or box IoU of the motion-predicted box when the masks miss) blended with the cosine similarity of the decoder query embeddings"""
    if not tracks or not dets: return [], list(range(len(tracks))), list(range(len(dets)))
    miou = iou_matrix([t.mask for t in tracks], [d["mask"] for d in dets]); biou = box_iou_matrix([t.pred_box for t in tracks], [d["box"] for d in dets]); geo = np.maximum(miou, (0.8 if mask is not None else 1.0) * biou)
    if a.app_w > 0:
        E = np.stack([t.emb for t in tracks]); D = np.stack([d["emb"] for d in dets]); cos = np.clip(E @ D.T, 0, 1); aff = (1 - a.app_w) * geo + a.app_w * cos * (geo > 0.05)
    else: aff = geo
    r, c = linear_sum_assignment(1 - aff); pairs = [(i, j) for i, j in zip(r, c) if geo[i, j] >= min_iou * 0.6 and aff[i, j] >= min_iou]
    ti = {i for i, _ in pairs}; di = {j for _, j in pairs}; return pairs, [i for i in range(len(tracks)) if i not in ti], [j for j in range(len(dets)) if j not in di]
class Tracker:
    def __init__(self): self.tracks = []; self.next_id = 1
    def step(self, dets):
        for t in self.tracks: t.votes *= 0.92
        high = [d for d in dets if d["p"] >= a.thr]; low = [d for d in dets if d["p"] < a.thr]
        pairs, un_t, un_d = match(self.tracks, high, 0.25)
        for i, j in pairs: self.tracks[i].update(high[j])
        rest = [self.tracks[i] for i in un_t]; pairs2, un_t2, _ = match(rest, low, 0.3)
        for i, j in pairs2: rest[i].update(low[j])
        for i in un_t2: rest[i].misses += 1; rest[i].det_row = None
        for j in un_d: self.tracks.append(Track(self.next_id, high[j])); self.next_id += 1
        self.tracks = [t for t in self.tracks if t.misses <= 15]
        return [t for t in self.tracks if t.hits >= 5 and t.misses <= 8 and t.p >= 0.4]
tracker = Tracker(); trk = None
SAM3_NP = a.sam3_tracker and (a.sam3_np or not HAVE_TORCH)
if a.sam3_tracker and not SAM3_NP:
    from sam3_track import Sam3Tracker
    trk = Sam3Tracker(f"{a.trk_dir}/trk_neck_fp16.plan", f"{a.trk_dir}/trk_init_fp16.plan", f"{a.trk_dir}/trk_step_v2_fp16.plan" if (not a.sam3_v1 and os.path.exists(f"{a.trk_dir}/trk_step_v2_fp16.plan")) else f"{a.trk_dir}/trk_step_fp16.plan", vision, "permute_4",
                      pe_mem=np.load(f"{a.trk_dir}/pe_mem.npy"), tpos_enc=np.load(f"{a.trk_dir}/tpos_enc.npy") if os.path.exists(f"{a.trk_dir}/tpos_enc.npy") else None, prune_r=a.sam3_prune); trk.m_max = min(trk.m_max, a.sam3_mem); trk.adaptive = a.sam3_adaptive
elif SAM3_NP:
    from sam3_track_np import Sam3TrackerNP
    util = a.util_so or os.path.join(os.path.dirname(os.environ.get("LQ_PLUGINS", "")), "lq_util.so")
    step_plan = f"{a.trk_dir}/trk_step_v3_fp16.plan" if (a.sam3_qprune > 0 and os.path.exists(f"{a.trk_dir}/trk_step_v3_fp16.plan")) else f"{a.trk_dir}/trk_step_v2_fp16.plan"; print("SAM 3 step engine:", os.path.basename(step_plan))
    trk = Sam3TrackerNP(f"{a.trk_dir}/trk_neck_fp16.plan", f"{a.trk_dir}/trk_init288_fp16.plan", step_plan, vision.bufs["permute_4"][0].ptr, vision.bufs["permute_4"][1], np.load(f"{a.trk_dir}/pe_mem.npy"), np.load(f"{a.trk_dir}/tpos_enc.npy"), util, prune_r=a.sam3_prune, qprune_r=a.sam3_qprune, qgrid_stride=a.sam3_qgrid, qfull_every=a.sam3_qfull)
    trk.m_max = min(trk.m_max, a.sam3_mem); trk.adaptive = a.sam3_adaptive
NATIVE = {}
def sam3_native_frame(fi, dets):
    """SAM3-native tracking: every object is propagated by the memory attention each frame and the masks come from the tracker.
    Detections only associate (mask IoU >= --sam3-assoc), refresh the conditioning memory every --refresh frames, and spawn new objects;
    presence follows SAM3's rule (object score > 0); an object unmatched by any detection for --max-miss frames is removed."""
    trk.run_neck(); high = [d for d in dets if d["p"] >= a.thr and d["mask"] is not None]
    ids = sorted(NATIVE)
    if ids:
        res = trk.propagate(fi, ids)
        for oid in ids:
            t = NATIVE[oid]
            if oid in res:
                m, sc, iou = res[oid]; t.mask = torch.as_tensor(m, device=dev) if HAVE_TORCH else m; t.p = float(1 / (1 + np.exp(-sc)))
                bx = mask_box(t); t.box = np.array(bx, np.float64)
    pairs, un_t, un_d = ([], list(range(len(ids))), list(range(len(high)))) if not (ids and high) else (lambda M: (lambda r, c: ([(i, j) for i, j in zip(r, c) if M[i, j] >= a.sam3_assoc], [i for i in range(len(ids)) if i not in set(r[M[r, c] >= a.sam3_assoc])], [j for j in range(len(high)) if j not in set(c[M[r, c] >= a.sam3_assoc])]))(*linear_sum_assignment(1 - M)))(iou_matrix([NATIVE[i].mask for i in ids], [d["mask"] for d in high]))
    for i, j in pairs:
        t = NATIVE[ids[i]]; d = high[j]; t.votes[d["k"]] += d["p"]; t.hits += 1; t.misses = 0
        if fi - trk.obj(t.id).last_init >= a.refresh:
            if SAM3_NP and d.get("row_ptr") is not None: trk.refresh(fi, [t.id], [d["row_ptr"]])
            elif not SAM3_NP: trk.refresh(fi, [t.id], torch.stack([Fn.interpolate(torch.as_tensor(d["mask"], device=dev)[None, None], size=(1008, 1008), mode="bilinear", align_corners=False)[0, 0] > 0]))
    for i in un_t:
        t = NATIVE[ids[i]]; t.misses += 1
        if t.misses > a.max_miss: trk.drop(t.id); del NATIVE[t.id]
    for j in un_d:
        d = high[j]
        if SAM3_NP and d.get("row_ptr") is None: continue
        t = Track(tracker.next_id, d); tracker.next_id += 1; NATIVE[t.id] = t
        if SAM3_NP: trk.refresh(fi, [t.id], [d["row_ptr"]])
        else: trk.refresh(fi, [t.id], torch.stack([Fn.interpolate(torch.as_tensor(d["mask"], device=dev)[None, None], size=(1008, 1008), mode="bilinear", align_corners=False)[0, 0] > 0]))
    trk.prune(list(NATIVE))
    return [t for t in NATIVE.values() if t.hits >= a.sam3_min_hits and t.p > 0.5 and t.mask is not None]
REF_MASK = {}
def mask72(t):
    m = t.mask[::4, ::4] > 0; return m.cpu().numpy() if HAVE_TORCH else np.asarray(m)
def refresh_due(t, fi):
    """change-gated refresh: at least --refresh frames apart, always by --refresh-max frames, and in between only when the detector mask moved
    (IoU with the mask at the last refresh below --refresh-iou); --refresh-iou 1 restores a fixed cadence"""
    age = fi - trk.obj(t.id).last_init
    if age < a.refresh: return False
    if age >= a.refresh_max or t.id not in REF_MASK or a.refresh_iou >= 1.0: return True
    cur = mask72(t); ref = REF_MASK[t.id]; union = int((cur | ref).sum()); return union == 0 or int((cur & ref).sum()) / union < a.refresh_iou
def sam3_frame(fi, dets):
    """hybrid: the lightweight tracker associates detections; SAM 3 memories are refreshed from detector masks every --refresh frames for tracks
    the detector sees, and SAM 3 propagation runs only for confirmed tracks the detector missed this frame (their mask and box follow the propagation)"""
    trk.run_neck(); shown = tracker.step(dets); live = {t.id for t in tracker.tracks}
    seen = [t for t in tracker.tracks if t.misses == 0 and t.hits >= a.sam3_min_hits]
    lost = [t for t in tracker.tracks if 0 < t.misses <= a.max_miss and t.hits >= a.sam3_min_hits and not trk.dropped(t.id)]
    lost.sort(key=lambda t: (t.misses, -t.hits)); lost = lost[:a.sam3_budget]                     # budget: the most recently lost, longest-lived tracks first
    due = [t for t in seen if refresh_due(t, fi) and (not SAM3_NP or getattr(t, "det_row", None) is not None)]
    for t in due: REF_MASK[t.id] = mask72(t)
    if due and SAM3_NP: trk.refresh(fi, [t.id for t in due], [t.det_row for t in due])
    elif due:
        m1008 = torch.stack([Fn.interpolate(torch.as_tensor(t.mask, device=dev)[None, None], size=(1008, 1008), mode="bilinear", align_corners=False)[0, 0] > 0 for t in due])   # stays on the GPU
        trk.refresh(fi, [t.id for t in due], m1008)
    if lost:
        res = trk.propagate(fi, [t.id for t in lost])
        for t in lost:
            if t.id in res and res[t.id][1] > 0 and res[t.id][2] >= a.sam3_min_iou:   # SAM 3 still sees the object with a confident mask: carry it, keep the track visible
                m, sc, iou = res[t.id]; t.mask = torch.as_tensor(m, device=dev); t.p = max(t.p, 0.45); t.misses = min(t.misses, 1)
                if t not in shown: shown.append(t)
            elif t.id in res: trk.drop(t.id); stats["sam3_drops"] = stats.get("sam3_drops", 0) + 1     # the tracker lost it too: stop spending steps on it
    trk.prune(live)
    for k in [k for k in REF_MASK if k not in live]: del REF_MASK[k]
    return shown

# ---- overlays ----
def rounded_box(img, x0, y0, x1, y1, color, t=3, r=14):
    import cv2
    x0, y0, x1, y1 = [int(round(v)) for v in (x0, y0, x1, y1)]; r = max(2, min(r, (x1 - x0) // 3, (y1 - y0) // 3)); c = color; AA = cv2.LINE_AA
    cv2.line(img, (x0 + r, y0), (x1 - r, y0), c, t, AA); cv2.line(img, (x0 + r, y1), (x1 - r, y1), c, t, AA); cv2.line(img, (x0, y0 + r), (x0, y1 - r), c, t, AA); cv2.line(img, (x1, y0 + r), (x1, y1 - r), c, t, AA)
    cv2.ellipse(img, (x0 + r, y0 + r), (r, r), 180, 0, 90, c, t, AA); cv2.ellipse(img, (x1 - r, y0 + r), (r, r), 270, 0, 90, c, t, AA); cv2.ellipse(img, (x1 - r, y1 - r), (r, r), 0, 0, 90, c, t, AA); cv2.ellipse(img, (x0 + r, y1 - r), (r, r), 90, 0, 90, c, t, AA)
def lower_third():
    l1, l2, l3 = a.title, a.name or " ", a.affiliation or " "
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10))); w = int(max(tmp.textlength(l1, FONT_T1), tmp.textlength(l2, FONT_T2), tmp.textlength(l3, FONT_T3))) + 76; h = 150
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=14, fill=(8, 10, 14, 225)); d.rounded_rectangle([0, 0, 8, h - 1], radius=4, fill=(255, 196, 0, 255))
    d.text((30, 14), l1, font=FONT_T1, fill=(255, 255, 255, 255)); d.text((30, 66), l2, font=FONT_T2, fill=(235, 235, 235, 255)); d.text((30, 106), l3, font=FONT_T3, fill=(190, 190, 190, 255))
    return im
LT = None if a.no_lower_third else lower_third()
def sigmoid(x): return 1 / (1 + np.exp(-x.astype(np.float64)))
def ease(x): x = max(0.0, min(1.0, x)); return x * x * (3 - 2 * x)

# ---- video I/O ----
img_list = None
if a.gpu_decode:
    from gpu_frames import list_images, GpuFrameSource
    assert HAVE_TORCH, "--gpu-decode needs torch with CUDA"; assert a.no_render, "--gpu-decode is for headless image input: add --no-render"
    img_list = list_images(a.video); assert img_list, f"no images found for {a.video}"
    if not a.fps_in: a.fps_in = 30.0
    if not a.src_size: a.src_size = "x".join(str(v) for v in Image.open(img_list[0]).size)
if a.fps_in: src_fps = a.fps_in; n_frames = 0
else:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,nb_frames", "-of", "csv=p=0", a.video], capture_output=True, text=True).stdout.strip().split(",")
    num, den = (probe[0].split("/") + ["1"])[:2] if probe and "/" in probe[0] else ("1", "1"); src_fps = float(num) / max(1e-6, float(den)); n_frames = int(probe[1]) if len(probe) > 1 and probe[1].isdigit() else 0
if a.src_size: SW, SH = [int(v) for v in a.src_size.lower().split("x")]
else:
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.video], capture_output=True, text=True).stdout.strip().split(","); SW, SH = int(pr[0]), int(pr[1])
mot_f = open(a.mot_out, "w") if a.mot_out else None
class SeqReader:
    """image-sequence input (printf pattern such as dir/%06d.jpg) without ffmpeg: PIL decode + bilinear resize to 1008, frames numbered from 1"""
    def __init__(self, pattern): self.pattern = pattern; self.i = 1; self.stdout = self
    def read(self, n):
        path = self.pattern % self.i
        if not os.path.exists(path): return b""
        self.i += 1; im = Image.open(path).convert("RGB").resize((1008, 1008), Image.BILINEAR); return np.asarray(im, dtype=np.uint8).tobytes()
    def kill(self): pass
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
is_image = a.video.lower().endswith(IMG_EXT) and "%" not in a.video
if is_image: a.max_frames = 1; a.tracker = "none"; a.no_swipe = True               # one frame: raw detections, full overlay, no identities to build
is_seq = "%" in a.video and a.no_render
class ImageReader:
    """single image: one 1008x1008 frame for the model, the original resolution for the output"""
    def __init__(self, path):
        im = Image.open(path).convert("RGB"); self.orig = im; self.done = False
        self.model = np.asarray(im.resize((1008, 1008), Image.BILINEAR)).tobytes()
    class _P:
        def __init__(self, b): self.b = b
        def read(self, n): r = self.b[:n]; self.b = self.b[n:]; return r
    @property
    def stdout(self): return self._out
    def kill(self): pass
if img_list is not None: dec_m = None                                             # frames come from the GPU source below
elif is_image:
    _ir = ImageReader(a.video); dec_m = _ir; dec_m._out = ImageReader._P(_ir.model)
    OW, OH = (_ir.orig.width // 2) * 2, (_ir.orig.height // 2) * 2; src_fps = 1.0; n_frames = 1
else: dec_m = SeqReader(a.video) if is_seq else subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.video, "-vf", "scale=1008:1008:flags=bilinear", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=10 ** 8)
if is_image:
    dec_d = ImageReader._P(np.asarray(_ir.orig.resize((OW, OH), Image.BILINEAR)).tobytes()) if not a.no_render else None; dec_d = type("D", (), {"stdout": dec_d, "kill": lambda self: None})() if dec_d else None
    enc = None
else: dec_d = None if a.no_render else subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.video, "-vf", f"scale={OW}:{OH}:flags=bicubic", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=10 ** 8)
enc = None if (a.no_render or is_image) else subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-framerate", f"{src_fps:.6f}", "-i", "-", "-r", str(a.fps_out), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "medium", a.out], stdin=subprocess.PIPE)
class PipelinedVision:
    """the vision engine on a non-blocking stream with double-buffered inputs/outputs: enqueue(t+1) is issued before frame t is consumed"""
    def __init__(self, runner):
        eng = runner.engine; self.ctx = eng.create_execution_context(); self.stream = Stream(); self.in_name = runner.inputs[0]; self.outs = runner.outputs; self.shapes = {n: runner.bufs[n][1] for n in runner.names}; self.dts = {n: runner.bufs[n][2] for n in runner.names}
        nb = lambda n: int(np.prod(self.shapes[n])) * np.dtype(self.dts[n]).itemsize
        self.sets = [{n: DevBuf(nb(n)) for n in self.outs} for _ in range(2)]; self.inp = [DevBuf(nb(self.in_name)) for _ in range(2)]; self.pinned = [PinnedBuf(self.shapes[self.in_name], self.dts[self.in_name]) for _ in range(2)]; self.events = [Event(), Event()]; self.hold = [None, None]
    def enqueue(self, t, x):
        s = t % 2
        if hasattr(x, "data_ptr"):      # preprocessed frame already on the GPU: device-to-device; keep it referenced until this slot is consumed
            assert x.is_cuda and x.is_contiguous() and x.numel() * x.element_size() == self.pinned[s].nbytes; self.hold[s] = x; memcpy_async(self.inp[s].ptr, x.data_ptr(), self.pinned[s].nbytes, D2D, self.stream)
        else: np.copyto(self.pinned[s].arr, x); memcpy_async(self.inp[s].ptr, self.pinned[s].ptr, self.pinned[s].nbytes, H2D, self.stream)
        self.ctx.set_tensor_address(self.in_name, self.inp[s].ptr)
        for n in self.outs: self.ctx.set_tensor_address(n, self.sets[s][n].ptr)
        assert self.ctx.execute_async_v3(self.stream.ptr); self.events[s].record(self.stream)
    def wait(self, t): self.events[t % 2].sync(); return self.sets[t % 2]
def preprocess(bm): return np.frombuffer(bm, np.uint8).reshape(1008, 1008, 3).transpose(2, 0, 1).astype(np.float32) * (1 / 127.5) - 1.0
def reader(q):
    fi = 0
    while True:
        bm = dec_m.stdout.read(1008 * 1008 * 3); bd = dec_d.stdout.read(OW * OH * 3) if dec_d else b""
        if len(bm) < 1008 * 1008 * 3 or (dec_d and len(bd) < OW * OH * 3) or (a.max_frames and fi >= a.max_frames) or (a.duration and fi >= int(a.duration * src_fps)): q.put(None); return
        q.put((preprocess(bm), bd)); fi += 1
gsrc = GpuFrameSource(img_list, 1008, limit=a.max_frames) if img_list is not None else None       # decode + resize + normalize on the GPU, in a thread, ahead of the loop
PV = PipelinedVision(vision) if a.pipeline else None
if a.pipeline:
    if gsrc is not None: fq = gsrc.q
    else: fq = queue.Queue(maxsize=3); threading.Thread(target=reader, args=(fq,), daemon=True).start()
    nxt = fq.get()
stats = {"frames": 0, "input_ms": [], "vision_ms": [], "head_ms": [], "mask_ms": [], "tracks": [], "new_ids": [], "appear": []}; prev_shown = set()
XG = np.arange(OW)[None, :, None]
src_dur = 0.0 if (a.no_render or a.fps_in or is_image) else float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.video], capture_output=True, text=True).stdout.strip() or 0)
clip_len = min(a.duration, src_dur) if a.duration else src_dur; cycle = clip_len >= 13.0      # the off/on cycle needs 4 s and must finish 2 s before the cut
T_OFF, T_ON = clip_len - 6.0, clip_len - 4.0
def mask_box(t):
    m = np.asarray(t.mask.cpu().numpy() if HAVE_TORCH and hasattr(t.mask, "cpu") else t.mask) > 0; ys, xs = np.nonzero(m)
    if len(xs) < 4: return tuple(t.box)
    return (float(xs.min()) * OW / 288, float(ys.min()) * OH / 288, float(xs.max() + 1) * OW / 288, float(ys.max() + 1) * OH / 288)
def wipe_boundary(t):
    """x boundary of the with/without wipe: left of it the overlay is shown. None = full overlay, 0 = raw.
    0 to 0.5 s raw; 0.5 to 4.5 s the overlay sweeps in from the left; for clips of 13 s or more it sweeps out over [L-6, L-4] and back in over [L-4, L-2]."""
    if a.no_swipe: return None
    if t < 0.5: return 0
    if t < 4.5: return int(OW * ease((t - 0.5) / 4.0)) + 1
    if not cycle or t < T_OFF or t >= T_ON + 2.0: return None
    if t < T_ON: return int(OW * (1 - ease((t - T_OFF) / 2.0)))       # overlay retreats to the left
    return int(OW * ease((t - T_ON) / 2.0)) + 1
t_start = time.time(); fi = 0
while True:
    if a.pipeline:
        if nxt is None: break
        x, bd = nxt
        if fi == 0: PV.enqueue(0, x[None])
        t0 = time.perf_counter(); nxt = fq.get(); t_in = time.perf_counter() - t0                  # time the loop waits for the next frame (decode is prefetched)
        if nxt is not None: PV.enqueue(fi + 1, nxt[0][None])                       # backbone of the next frame starts now, on its own stream
        bufset = PV.wait(fi); v = {} if a.groundmask else {"fpn_2": bufset["fpn_2"].download(PV.shapes["fpn_2"], PV.dts["fpn_2"])}; t1 = time.perf_counter()
        if mask is not None: mask.bind("fpn_0", bufset[fpn0_name].ptr, PV.shapes[fpn0_name]); mask.bind("fpn_1", bufset[fpn1_name].ptr, PV.shapes[fpn1_name])
        if a.groundmask: mask.bind("fpn_2", bufset[fpn2_name].ptr, PV.shapes[fpn2_name])
        if trk is not None: trk.neck.bind("trunk", bufset["permute_4"].ptr, PV.shapes["permute_4"])
    else:
        ti = time.perf_counter()
        if gsrc is not None:
            item = gsrc.get()
            if item is None: break
            x, bd = item
        else:
            bm = dec_m.stdout.read(1008 * 1008 * 3); bd = dec_d.stdout.read(OW * OH * 3) if dec_d else b""
            if len(bm) < 1008 * 1008 * 3 or (dec_d and len(bd) < OW * OH * 3) or (a.max_frames and fi >= a.max_frames) or (a.duration and fi >= int(a.duration * src_fps)): break
            x = preprocess(bm)
        t_in = time.perf_counter() - ti
        t0 = time.perf_counter(); v = vision({"images": x[None]}, want=[] if a.groundmask else ["fpn_2"]); DevBuf.sync(); t1 = time.perf_counter()
    g = fused({"text_feats": tf4, "text_mask": tm4}) if a.groundmask else ground({"img_feat": np.repeat(v["fpn_2"].astype(np.float16), KB, axis=0), "img_pos": img_pos, "text_feats": tf4, "text_mask": tm4}); DevBuf.sync(); t2 = time.perf_counter()
    probs = sigmoid(g["scores"][:K, :, 0]) * sigmoid(g["presence"][:K, 0])[:, None]
    active = [k for k in range(K) if (probs[k] > a.low).any()]; sel = {k: (g["sel"][k] if a.groundmask else np.argsort(-probs[k])[:QSEL]) for k in active}; dets = []
    if active:
        cand = sorted([(float(probs[k, q]), k, q, (k if a.groundmask else ai), qi) for ai, k in enumerate(active) for qi, q in enumerate(sel[k]) if probs[k, q] > a.low], key=lambda c: -c[0])   # ai: row of the mask array (the prompt itself for the fused engine)
        if mask is not None:
            if a.groundmask: m = g["masks"]
            else: hs_sel = np.stack([g["hs"][k, sel[k]] for k in active]).astype(np.float16); m = mask({"enc": np.ascontiguousarray(g["enc"][:, active, :]), "text_feats": np.ascontiguousarray(tfK[:, active, :]), "text_mask": np.ascontiguousarray(tmK[active]), "hs_sel": hs_sel})["masks"]
            mt = torch.from_numpy(m.astype(np.float32)).to(dev) if HAVE_TORCH else m.astype(np.float32)
        if cand and mask is not None:
            if HAVE_TORCH:
                M = (torch.stack([mt[c[3], c[4]] for c in cand]) > 0).float().flatten(1); inter = (M @ M.t()); area = M.sum(1)
                iou = (inter / (area[:, None] + area[None, :] - inter + 1e-6)).cpu().numpy(); cont = (inter / (torch.minimum(area[:, None], area[None, :]) + 1e-6)).cpu().numpy()
                ins = (inter / (area[:, None] + 1e-6)).cpu().numpy(); ar = area.cpu().numpy()
            else:
                M = (np.stack([mt[c[3], c[4]] for c in cand]) > 0).reshape(len(cand), -1).astype(np.float32); inter = M @ M.T; ar = M.sum(1)
                iou = inter / (ar[:, None] + ar[None, :] - inter + 1e-6); cont = inter / (np.minimum(ar[:, None], ar[None, :]) + 1e-6); ins = inter / (ar[:, None] + 1e-6)
            keep = []; n = len(cand)
            groups = {j for j in range(n) if sum(1 for i in range(n) if i != j and ins[i, j] > 0.7 and ar[i] < 0.5 * ar[j] and cand[i][0] >= a.thr) >= 2}   # contains two confident objects: a group, prefer the parts
            for i in range(n):              # highest score first: drop groups, duplicates (mask IoU) and nested parts of an already kept object (containment)
                if i in groups: continue
                if a.nms < 1 and any(iou[i, j] > a.nms or cont[i, j] > 0.85 for j in keep): continue
                keep.append(i)
            for i in keep:
                p, k, q, ai, qi = cand[i]; cx, cy, w, h = g["boxes"][k, q].astype(np.float64)
                e = g["hs"][k, q].astype(np.float32); e = e / (np.linalg.norm(e) + 1e-6)
                dets.append({"p": p, "k": k, "mask": mt[ai, qi], "emb": e, "box": np.array([(cx - w / 2) * OW, (cy - h / 2) * OH, (cx + w / 2) * OW, (cy + h / 2) * OH]), "row_ptr": mask.bufs["masks"][0].ptr + (ai * QSEL + qi) * 288 * 288 * 2})
        elif cand:                                                                   # boxes only: the same suppression on box geometry
            bx = np.array([[(g["boxes"][k, q][0] - g["boxes"][k, q][2] / 2) * OW, (g["boxes"][k, q][1] - g["boxes"][k, q][3] / 2) * OH, (g["boxes"][k, q][0] + g["boxes"][k, q][2] / 2) * OW, (g["boxes"][k, q][1] + g["boxes"][k, q][3] / 2) * OH] for _, k, q, _, _ in cand], np.float64)
            iou = box_iou_matrix(bx, bx); ar = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]); lt = np.maximum(bx[:, None, :2], bx[None, :, :2]); rb = np.minimum(bx[:, None, 2:], bx[None, :, 2:]); wh = np.clip(rb - lt, 0, None); inter = wh[..., 0] * wh[..., 1]
            ins = inter / (ar[:, None] + 1e-6); cont = inter / (np.minimum(ar[:, None], ar[None, :]) + 1e-6); n = len(cand); keep = []
            groups = {j for j in range(n) if sum(1 for i in range(n) if i != j and ins[i, j] > 0.7 and ar[i] < 0.5 * ar[j] and cand[i][0] >= a.thr) >= 2}
            for i in range(n):
                if i in groups: continue
                if a.nms < 1 and any(iou[i, j] > a.nms or cont[i, j] > 0.85 for j in keep): continue
                keep.append(i)
            for i in keep:
                p, k, q, ai, qi = cand[i]; e = g["hs"][k, q].astype(np.float32); e = e / (np.linalg.norm(e) + 1e-6)
                dets.append({"p": p, "k": k, "mask": None, "emb": e, "box": bx[i]})
    t3 = time.perf_counter(); n0 = tracker.next_id
    if a.tracker == "none":
        shown = [Track(i + 1, d) for i, d in enumerate(dets)]
        for t in shown: t.hits = 99
    else: shown = ((sam3_native_frame(fi, dets) if a.tracker == "sam3" else sam3_frame(fi, dets)) if trk else tracker.step(dets))
    if mot_f is not None:
        for t in shown:
            x0, y0, x1, y1 = (mask_box(t) if (t.mask is not None and a.mot_box == "mask") else tuple(t.box)); mot_f.write(f"{fi + 1},{t.id},{x0 * SW / OW:.2f},{y0 * SH / OH:.2f},{(x1 - x0) * SW / OW:.2f},{(y1 - y0) * SH / OH:.2f},{t.p:.3f},-1,-1,-1\n")
    if a.no_render:
        stats["vision_ms"].append((t1 - t0) * 1000); stats["head_ms"].append((t2 - t1) * 1000); stats["mask_ms"].append((t3 - t2) * 1000); stats["tracks"].append(len(shown)); stats["input_ms"].append(t_in * 1000); stats["frames"] += 1; fi += 1
        if fi % 200 == 0: print(f"{fi}/{n_frames}  backbone {np.mean(stats['vision_ms'][-200:]):.1f}  head {np.mean(stats['head_ms'][-200:]):.1f}  masks {np.mean(stats['mask_ms'][-200:]):.1f} ms  tracks {len(shown)}  {time.time() - t_start:.0f}s", flush=True)
        stats["new_ids"].append(tracker.next_id - n0); ids = {t.id for t in shown}; stats["appear"].append(len(ids - prev_shown)); prev_shown = ids
        continue
    stats["new_ids"].append(tracker.next_id - n0); ids = {t.id for t in shown}; stats["appear"].append(len(ids - prev_shown)); prev_shown = ids
    # ---- render: soft masks, anti-aliased contours and rounded boxes, name labels, lower third ----
    frame = torch.from_numpy(np.frombuffer(bd, np.uint8).reshape(OH, OW, 3).copy()).to(dev).float(); contours = []
    dense = len(shown) > 15; boxes = {}
    for t in shown:
        if t.mask is None: boxes[t.id] = tuple(t.box); continue
        col = torch.tensor(PAL[t.cls % len(PAL)], device=dev).float(); tm_ = torch.as_tensor(t.mask, device=dev).float(); soft = torch.sigmoid(Fn.interpolate(tm_[None, None], size=(OH, OW), mode="bilinear", align_corners=False)[0, 0] * 1.5)
        al = (soft * 0.30)[..., None]; frame = frame * (1 - al) + col * al; b = (soft > 0.5).byte().cpu().numpy(); contours.append((b, PAL[t.cls % len(PAL)]))
        ys, xs = np.where(b)
        boxes[t.id] = (xs.min() - 3, ys.min() - 3, xs.max() + 3, ys.max() + 3) if len(xs) > 30 else tuple(t.box)     # box from the smoothed mask: consistent with what is drawn
    import cv2
    img = np.ascontiguousarray(frame.clamp(0, 255).byte().cpu().numpy())
    for bm_, c in contours:
        cs, _ = cv2.findContours(bm_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cs = [cc for cc in cs if cv2.contourArea(cc) > 150]
        if cs: cv2.polylines(img, cs, True, c, 2, cv2.LINE_AA)
    for t in shown: rounded_box(img, *boxes[t.id], PAL[t.cls % len(PAL)], t=2 if dense else 3, r=10 if dense else 14)
    pil = Image.fromarray(img); d = ImageDraw.Draw(pil, "RGBA"); fl = FONT_LAB_S if dense else FONT_LAB; lh = 26 if dense else 32
    for t in shown:
        c = PAL[t.cls % len(PAL)]; lab = prompts[t.cls]; tw = d.textlength(lab, font=fl); x0, y0 = boxes[t.id][0], boxes[t.id][1]; ly = y0 - lh - 4 if y0 - lh - 4 > 4 else y0 + 3; lx = max(4, min(x0, OW - tw - 20))
        d.rounded_rectangle([lx, ly, lx + tw + 14, ly + lh], radius=7, fill=c + (255,)); d.text((lx + 7, ly + 1), lab, font=fl, fill=(10, 10, 10, 255))
    tsec = fi / src_fps; xb = wipe_boundary(tsec); raw = np.frombuffer(bd, np.uint8).reshape(OH, OW, 3)
    if xb is not None:                                   # with/without wipe: overlay left of the boundary, raw video right of it
        ov = np.asarray(pil.convert("RGB")); comp = np.where(XG < xb, ov, raw); pil = Image.fromarray(np.ascontiguousarray(comp)); d = ImageDraw.Draw(pil, "RGBA")
        if 0 < xb < OW: d.rectangle([xb - 5, 0, xb + 5, OH], fill=(255, 196, 0, 70)); d.rectangle([xb - 2, 0, xb + 1, OH], fill=(255, 196, 0, 255))
    vm = np.mean(stats["vision_ms"][-30:]) if stats["vision_ms"] else (t1 - t0) * 1000; pa = ease((tsec - 0.5) / 1.0)
    if pa > 0:
        info1 = "prompts: " + ", ".join(prompts); info2 = f"SAM 3 ViT H, W8A8 in TensorRT, 1008 px; backbone {vm:.0f} ms per frame on an {a.gpu_name}" + ("; boxes only" if a.heads == "boxes" else "") + ("; SAM 3 propagation" if a.tracker == "sam3" else "; no tracker" if a.tracker == "none" else "")
        w1, w2 = d.textlength(info1, font=FONT_INFO_B), d.textlength(info2, font=FONT_INFO); pw = int(max(w1, w2)) + 40; A = lambda v: int(v * pa)
        d.rounded_rectangle([OW - pw - 40, 40, OW - 40, 118], radius=12, fill=(8, 10, 14, A(215))); d.text((OW - pw - 20, 50), info1, font=FONT_INFO_B, fill=(255, 196, 0, A(255))); d.text((OW - pw - 20, 84), info2, font=FONT_INFO, fill=(220, 220, 220, A(255)))
    if LT is not None:
        e = ease((tsec - a.lt_in) / 0.8) * (1 - ease((tsec - (a.lt_out - 1.0)) / 1.0))     # in after lt_in, gone by lt_out
        if e > 0:
            lt = LT if e >= 1 else LT.copy()
            if e < 1: lt.putalpha(lt.getchannel("A").point(lambda v: int(v * e)))
            slide = int((1 - ease((tsec - a.lt_in) / 0.8)) * 60) if a.lt_slide else 0
            pil.paste(lt, (40 - slide, OH - LT.height - 40), lt)
    if is_image: pil.convert("RGB").save(a.out)
    else: out = np.asarray(pil.convert("RGB")); enc.stdin.write(out.tobytes())
    if a.snap and fi == int(a.snap): pil.convert("RGB").save(a.out + f".frame{fi}.png")
    stats["vision_ms"].append((t1 - t0) * 1000); stats["head_ms"].append((t2 - t1) * 1000); stats["mask_ms"].append((t3 - t2) * 1000); stats["tracks"].append(len(shown)); stats["input_ms"].append(t_in * 1000); stats["frames"] += 1; fi += 1
    if fi % 50 == 0: print(f"{fi}/{n_frames}  backbone {np.mean(stats['vision_ms'][-50:]):.1f}  head {np.mean(stats['head_ms'][-50:]):.1f}  masks {np.mean(stats['mask_ms'][-50:]):.1f} ms  tracks {len(shown)}  {time.time() - t_start:.0f}s", flush=True)
if enc: enc.stdin.close(); enc.wait()
if dec_m is not None: dec_m.kill()
if dec_d: dec_d.kill()
if mot_f: mot_f.close()
summary = {"video": a.video, "prompts": prompts, "tracker": a.tracker, "heads": a.heads, "pipeline": a.pipeline, "frames": stats["frames"], "wall_ms_per_frame": (time.time() - t_start) * 1000 / max(1, stats["frames"]), "backbone_ms": float(np.mean(stats["vision_ms"])), "head_ms": float(np.mean(stats["head_ms"])), "mask_ms": float(np.mean(stats["mask_ms"])), **({"sam3_tracker_ms_per_frame": {k: v / max(1, stats["frames"]) for k, v in trk.t_ms.items()}, "sam3_refreshes": trk.n_init, "sam3_steps": trk.n_step, "sam3_step_calls": trk.n_step_calls, "sam3_drops": stats.get("sam3_drops", 0), **({"sam3_prof_ms_per_call": {k: v * 1000 / max(1, trk.n_step_calls) for k, v in trk.prof.items()}} if getattr(trk, "prof", None) else {}), "sam3_keys_fraction": trk.keys_used / max(1, trk.keys_full), "sam3_queries_fraction": (trk.q_used / max(1, trk.q_full)) if getattr(trk, "q_full", 0) else None} if trk else {}), "tracks_mean": float(np.mean(stats["tracks"])), "new_ids_per_frame_after_warmup": float(np.mean(stats["new_ids"][10:])) if len(stats["new_ids"]) > 10 else None, "tracks_appearing_per_frame": float(np.mean(stats["appear"][10:])) if len(stats["appear"]) > 10 else None, "src_fps": src_fps}
summary["input_wait_ms"] = float(np.mean(stats["input_ms"])) if stats["input_ms"] else 0.0
if gsrc is not None: summary["gpu_decode_ms"] = float(np.mean(gsrc.decode_ms)) if gsrc.decode_ms else 0.0; summary["gpu_decode_fallbacks"] = gsrc.fallbacks
print(json.dumps(summary)); json.dump(summary, open(a.stats or a.out + ".json", "w"), indent=1)
