"""Hadamard basis + window-major (no-window) trunk layout, combined (both exact). usage: export_hf_had_nowin.py <dart_root> <out_dir> <imgsz> <check_image>"""
import sys, os
dart, out, imgsz, img = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, dart); sys.path.insert(0, os.path.join(dart, "scripts")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
_orig_arange = torch.arange
def _arange(*a, **k):
    if str(k.get("device", "")).startswith("cuda"): k["device"] = "cpu"
    return _orig_arange(*a, **k)
torch.arange = _arange
try:
    from sam3.model.position_encoding import PositionEmbeddingSine as _PES
    _oi = _PES.__init__
    def _ci(self, *a, **k): k["precompute_resolution"] = None; _oi(self, *a, **k)
    _PES.__init__ = _ci
except Exception as e: print("PES patch skipped:", e)
import transformers.models.sam3.modeling_sam3 as M
from hadamard_basis import prepare as had_prepare
def nowin_prepare(vit):
    cfg = vit.config; ws = cfg.window_size; H = W = imgsz // cfg.patch_size; nh, nw = H // ws, W // ws
    perm = torch.arange(H * W).view(nh, ws, nw, ws).permute(0, 2, 1, 3).reshape(-1)
    for layer in vit.layers:
        if layer.window_size > 0: layer.window_size = 0
        else:
            re = layer.rotary_emb; re.rope_embeddings_cos = torch.nn.Buffer(re.rope_embeddings_cos[perm].clone(), persistent=False); re.rope_embeddings_sin = torch.nn.Buffer(re.rope_embeddings_sin[perm].clone(), persistent=False)
    vit._nowin = (nh, nw, ws); return vit
def prepare_both(vit): had_prepare(vit); nowin_prepare(vit); return vit
def forward_both(self, pixel_values, **kw):
    h = self.embeddings(pixel_values); B = h.shape[0]; H = pixel_values.shape[-2] // self.config.patch_size; W = pixel_values.shape[-1] // self.config.patch_size; C = h.shape[-1]
    g, b = self._had_pre; h = torch.nn.functional.layer_norm(h.view(B, H, W, C), (C,), g, b, self._had_eps) @ self._had     # pre-LN + rotate once
    nh, nw, ws = self._nowin; h = h.view(B, nh, ws, nw, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B * nh * nw, ws, ws, C)   # partition once
    for layer in self.layers:
        if layer.rotary_emb.rope_embeddings_cos.shape[0] == ws * ws: h = layer(h, **kw)
        else: h = layer(h.view(B, H, W, C), **kw).view(B * nh * nw, ws, ws, C)
    h = h.view(B, nh, nw, ws, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C) @ self._had.t()                          # unpartition + un-rotate once
    return M.BaseModelOutput(last_hidden_state=h.view(B, H * W, C))
import export_hf_backbone as E
from transformers import Sam3Model
from preprocess import load_image_tensor
x, _ = load_image_tensor(img, imgsz); x = torch.from_numpy(x)
m = Sam3Model.from_pretrained("facebook/sam3", attn_implementation="eager").eval()
with torch.no_grad():
    ref = m.vision_encoder.backbone(pixel_values=x).last_hidden_state
    prepare_both(m.vision_encoder.backbone); M.Sam3ViTModel.forward = forward_both
    new = m.vision_encoder.backbone(pixel_values=x).last_hidden_state
rel = ((new - ref).norm() / ref.norm()).item(); print(f"had+nowin vs original: rel={rel:.3e}", flush=True); assert rel < 1e-3
del m
_orig_fp = Sam3Model.from_pretrained.__func__
@classmethod
def _fp(cls, *a, **k):
    mm = _orig_fp(cls, *a, **k); prepare_both(mm.vision_encoder.backbone); return mm
Sam3Model.from_pretrained = _fp
_B = int(os.environ.get("LQ_BATCH", "1"))          # batch size baked into the exported graph (static shapes); the weights / calibration / GPTQ results do not depend on it
if _B > 1:
    _orig_randn = torch.randn
    def _randn(*a, **k): return _orig_randn(_B, *a[1:], **k) if len(a) == 4 and a[0] == 1 and a[1] == 3 else _orig_randn(*a, **k)
    torch.randn = _randn
os.makedirs(out, exist_ok=True); print("export_onnx returned:", E.export_onnx(out, imgsz=imgsz), "| batch", _B)
