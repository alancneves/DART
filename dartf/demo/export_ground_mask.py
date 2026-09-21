"""Export ground_cKm.onnx (grounding head that also emits the decoder queries `hs` [K,200,256] and encoder states `enc` [5184,K,256])
and maskhead_qQ.onnx (the SAM 3 segmentation head with a dynamic prompt axis K). CPU, a few seconds; needs the DART sam3 package and sam3.pt.
usage: python export_ground_mask.py --dart <DART root> --ckpt sam3.pt --out out/head [--buckets 4,1] [--qsel 32]"""
import sys, os, json, time, argparse
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--dart", required=True, help="DART repository root (its sam3 package is imported)"); ap.add_argument("--ckpt", required=True, help="official sam3.pt")
ap.add_argument("--out", required=True); ap.add_argument("--buckets", default="4,1", help="prompt buckets K to export"); ap.add_argument("--qsel", type=int, default=32, help="queries per prompt fed to the mask head")
ap.add_argument("--threads", type=int, default=16); ap.add_argument("--shared-encoder", action="store_true", help="also export groundmaskse_cK: the fused head with ONE encoder pass at batch 1 conditioned on a generic prompt (inputs gen_feats [32,1,256], gen_mask [1,32]) and the decoder per prompt on the shared memory; approximate (the encoder no longer sees each prompt), ~(K-1)/K of the encoder cost saved"); ap.add_argument("--u8-masks", action="store_true", help="also export groundmaskU8_cK: the fused head reduced to what the model-server postprocessing needs: scores [K,200,1] fp16, boxes [K,200,4] fp16 (normalised cxcywh, DETR head; only used by the boxes-only path), sel [K,Q] int64 and masks_u8 [K,Q,288,288] uint8 (mask logit > 0, i.e. sigmoid > 0.5, thresholded on the GPU); no presence / hs outputs"); ap.add_argument("--fused", action="store_true", help="also export groundmask_cK: grounding head + in-graph top-Q query selection + segmentation head in one static graph (fpn_0/1/2 bound to the backbone buffers, img_pos baked in)"); ap.add_argument("--phase-conv", action="store_true", help="exact rewrite of conv3x3(fpn + nearest_up2(x)) as conv3x3(fpn) once per image plus four 2x2 phase convolutions at half resolution"); ap.add_argument("--only-mask", action="store_true"); args = ap.parse_args()
DART = args.dart; CKPT = args.ckpt; OUT = Path(args.out); BUCKETS = [int(k) for k in args.buckets.split(",")]; QSEL = args.qsel
sys.path.insert(0, DART)
import torch
import torch.nn.functional as F
def _cpu(dev): return "cpu" if (dev is not None and str(dev).startswith("cuda")) else dev
for fn in ("arange","zeros","ones","empty","tensor","linspace","full","rand","randn"):
    orig=getattr(torch,fn)
    def make(orig):
        def w(*a,**k):
            if "device" in k: k["device"]=_cpu(k["device"])
            return orig(*a,**k)
        return w
    setattr(torch,fn,make(orig))
torch.Tensor.cuda=lambda self,*a,**k: self; torch.nn.Module.cuda=lambda self,*a,**k: self
_orig_to=torch.Tensor.to
def _to(self,*a,**k):
    a=tuple(_cpu(x) if isinstance(x,(str,torch.device)) else x for x in a)
    if "device" in k: k["device"]=_cpu(k["device"])
    return _orig_to(self,*a,**k)
torch.Tensor.to=_to
torch.set_num_threads(args.threads); torch.manual_seed(314159)
from sam3.model.position_encoding import PositionEmbeddingSine
_oi = PositionEmbeddingSine.__init__
def _ci(self, *a, **k): k["precompute_resolution"] = None; _oi(self, *a, **k)
PositionEmbeddingSine.__init__ = _ci
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import inverse_sigmoid
t0 = time.time()
model = build_sam3_image_model(checkpoint_path=CKPT, device="cpu", eval_mode=True, load_from_HF=False, enable_segmentation=True, enable_inst_interactivity=False)
PositionEmbeddingSine.__init__ = _oi
print("model loaded %.0fs" % (time.time() - t0), flush=True)
seg = model.segmentation_head; print("seg head:", type(seg).__name__, "cross_attend:", seg.cross_attend_prompt is not None, "act_ckpt:", seg.act_ckpt)
seg.act_ckpt = False

class GroundingWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder, scoring):
        super().__init__(); self.encoder = encoder; self.decoder = decoder; self.scoring = scoring
    def forward(self, img_feat, img_pos, text_feats, text_mask):
        img_feat = img_feat.to(torch.float32); img_pos = img_pos.to(torch.float32); text_feats = text_feats.to(torch.float32); tm = text_mask.to(torch.bool)
        img_seq = img_feat.flatten(2).permute(2, 0, 1); pos_seq = img_pos.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src=[img_seq], src_key_padding_mask=None, src_pos=[pos_seq], prompt=text_feats, prompt_key_padding_mask=tm, feat_sizes=[(72, 72)])
        query_embed = self.decoder.query_embed.weight; target = query_embed.unsqueeze(1).expand(-1, img_feat.shape[0], -1)
        hidden, reference_boxes, presence, _ = self.decoder(tgt=target, memory=memory["memory"], memory_key_padding_mask=memory["padding_mask"], pos=memory["pos_embed"],
            reference_boxes=None, level_start_index=memory["level_start_index"], spatial_shapes=memory["spatial_shapes"], valid_ratios=memory["valid_ratios"],
            tgt_mask=None, memory_text=text_feats, text_attention_mask=tm, apply_dac=False)
        hidden = hidden.transpose(1, 2); reference_boxes = reference_boxes.transpose(1, 2); hidden_last = hidden[-1:]
        scores = self.scoring(hidden_last, text_feats, tm)[0]; box_offsets = self.decoder.bbox_embed(hidden_last)[0]
        boxes = (inverse_sigmoid(reference_boxes[-1]) + box_offsets).sigmoid(); presence_last = presence[-1].permute(1, 0)
        return scores.to(torch.float16), boxes.to(torch.float16), presence_last.to(torch.float16), hidden_last[0].to(torch.float16), memory["memory"].to(torch.float16)

class SharedGroundingWrapper(GroundingWrapper):
    """Shared-encoder grounding: encoder once at batch 1 with a generic prompt (gen_feats/gen_mask), decoder + scoring per prompt (text_feats/text_mask [K]) on the expanded memory."""
    def forward(self, img_feat, img_pos, gen_feats, gen_mask, text_feats, text_mask):
        K = text_mask.shape[0]; img_feat = img_feat.to(torch.float32); img_pos = img_pos.to(torch.float32); gf = gen_feats.to(torch.float32); gm = gen_mask.to(torch.bool)
        tf = text_feats.to(torch.float32); tm = text_mask.to(torch.bool)
        img_seq = img_feat.flatten(2).permute(2, 0, 1); pos_seq = img_pos.flatten(2).permute(2, 0, 1)
        memory = self.encoder(src=[img_seq], src_key_padding_mask=None, src_pos=[pos_seq], prompt=gf, prompt_key_padding_mask=gm, feat_sizes=[(72, 72)])
        mem = memory["memory"].expand(-1, K, -1); pos = memory["pos_embed"].expand(-1, K, -1); vr = memory["valid_ratios"].expand(K, -1, -1)
        pm = memory["padding_mask"]; pm = pm.expand(K, -1) if pm is not None else None
        target = self.decoder.query_embed.weight.unsqueeze(1).expand(-1, K, -1)
        hidden, reference_boxes, presence, _ = self.decoder(tgt=target, memory=mem, memory_key_padding_mask=pm, pos=pos, reference_boxes=None, level_start_index=memory["level_start_index"],
            spatial_shapes=memory["spatial_shapes"], valid_ratios=vr, tgt_mask=None, memory_text=tf, text_attention_mask=tm, apply_dac=False)
        hidden = hidden.transpose(1, 2); reference_boxes = reference_boxes.transpose(1, 2); hidden_last = hidden[-1:]
        scores = self.scoring(hidden_last, tf, tm)[0]; box_offsets = self.decoder.bbox_embed(hidden_last)[0]
        boxes = (inverse_sigmoid(reference_boxes[-1]) + box_offsets).sigmoid(); presence_last = presence[-1].permute(1, 0)
        return scores.to(torch.float16), boxes.to(torch.float16), presence_last.to(torch.float16), hidden_last[0].to(torch.float16), mem.to(torch.float16)

class PhasePixelDecoder(torch.nn.Module):
    """Exact rewrite of PixelDecoder.forward. For each level: conv3x3(fpn + up2(prev)) = conv3x3(fpn) + conv3x3(up2(prev)); the second term on a
    nearest-upsampled input equals, for output phase (p, q), a 2x2 convolution of prev with the 3x3 taps merged pairwise and an asymmetric pad
    (rows {i-1, i} for p=0, {i, i+1} for p=1; same for columns), followed by a pixel shuffle. 4 taps at half resolution instead of 9 at full."""
    def __init__(s, pd):
        super().__init__(); s.pd = pd; s.phase = torch.nn.ModuleList()
        for conv in pd.conv_layers:
            W = conv.weight.detach(); C = W.shape[0]; ph = torch.nn.ModuleList()
            for p in (0, 1):
                for q in (0, 1):
                    rows = [W[:, :, 0], W[:, :, 1] + W[:, :, 2]] if p == 0 else [W[:, :, 0] + W[:, :, 1], W[:, :, 2]]      # [C, Cin, 3] each, taps along columns
                    k = torch.stack(rows, 2)                                                                                # [C, Cin, 2, 3]
                    cols = [k[..., 0], k[..., 1] + k[..., 2]] if q == 0 else [k[..., 0] + k[..., 1], k[..., 2]]
                    k = torch.stack(cols, 3)                                                                                # [C, Cin, 2, 2]
                    c = torch.nn.Conv2d(W.shape[1], C, 2, bias=False); c.weight.data.copy_(k); c.pad = (1 - q, q, 1 - p, p); ph.append(c)   # (left, right, top, bottom)
            s.phase.append(ph)
    def forward(s, backbone_feats):
        prev = backbone_feats[-1]; fpn_feats = backbone_feats[:-1]
        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            conv = s.pd.conv_layers[layer_idx]; a = F.conv2d(bb_feat, conv.weight, conv.bias, padding=1)                    # once per image (batch 1), broadcast below
            outs = [F.conv2d(F.pad(prev, c.pad), c.weight) for c in s.phase[layer_idx]]                                      # phases (0,0),(0,1),(1,0),(1,1)
            up = F.pixel_shuffle(torch.stack(outs, 2).flatten(1, 2), 2)                                                     # [B, C*4, h, w] channel-major, phase (p, q) minor -> [B, C, 2h, 2w]
            prev = F.relu(s.pd.norms[layer_idx](a + up))
        return prev
class MaskWrapper(torch.nn.Module):
    """fpn_0 [1,256,288,288] fp32, fpn_1 [1,256,144,144] fp32, enc [5184,K,256] fp16, text_feats [32,K,256] fp16, text_mask [K,32] fp32, hs_sel [K,Q,256] fp16 -> masks [K,Q,288,288] fp16"""
    def __init__(self, seg): super().__init__(); self.seg = seg
    def forward(self, fpn_0, fpn_1, enc, text_feats, text_mask, hs_sel):
        enc = enc.to(torch.float32); tf = text_feats.to(torch.float32); tm = text_mask.to(torch.bool); hs = hs_sel.to(torch.float32)
        K = enc.shape[1]; fpn_2 = torch.zeros((1, 256, 72, 72), dtype=torch.float32)
        out = self.seg(backbone_feats=[fpn_0.to(torch.float32), fpn_1.to(torch.float32), fpn_2], obj_queries=hs.unsqueeze(0), image_ids=torch.zeros((K,), dtype=torch.long),
                       encoder_hidden_states=enc, prompt=tf, prompt_mask=tm)
        return out["pred_masks"].to(torch.float16)

class FusedWrapper(torch.nn.Module):
    """fpn_0 [1,256,288,288], fpn_1 [1,256,144,144], fpn_2 [1,256,72,72] (fp32, bound to the backbone's buffers), text_feats [32,K,256] fp16, text_mask [K,32] fp32
    -> scores [K,200,1], boxes [K,200,4], presence [K,1], hs [K,200,256] (fp16), sel [K,Q] int64 (top-Q queries per prompt by sigmoid(score)*sigmoid(presence)), masks [K,Q,288,288] fp16.
    Same mathematics as ground_cKm followed by maskhead_qQ on every prompt; nothing crosses the host between the two."""
    def __init__(self, ground, seg, img_pos_k, Q): super().__init__(); self.ground = ground; self.seg = seg; self.Q = Q; self.register_buffer("img_pos", img_pos_k.to(torch.float32))
    def forward(self, fpn_0, fpn_1, fpn_2, text_feats, text_mask):
        K = text_mask.shape[0]; img_feat = fpn_2.to(torch.float32).expand(K, -1, -1, -1)
        scores, boxes, presence, hs, enc = self.ground(img_feat, self.img_pos, text_feats, text_mask)
        probs = torch.sigmoid(scores.to(torch.float32)[:, :, 0]) * torch.sigmoid(presence.to(torch.float32)[:, 0])[:, None]
        sel = probs.topk(self.Q, dim=1).indices                                                  # [K, Q]
        hs_sel = torch.gather(hs, 1, sel[..., None].expand(-1, -1, hs.shape[-1]))
        tf = text_feats.to(torch.float32); tm = text_mask.to(torch.bool)
        out = self.seg(backbone_feats=[fpn_0.to(torch.float32), fpn_1.to(torch.float32), torch.zeros((1, 256, 72, 72), dtype=torch.float32)], obj_queries=hs_sel.to(torch.float32).unsqueeze(0),
                       image_ids=torch.zeros((K,), dtype=torch.long), encoder_hidden_states=enc.to(torch.float32), prompt=tf, prompt_mask=tm)
        return scores, boxes, presence, hs, sel, out["pred_masks"].to(torch.float16)
class SharedFusedWrapper(FusedWrapper):
    """FusedWrapper with the shared encoder: fpn_0/1/2, text_feats [32,K,256], text_mask [K,32], gen_feats [32,1,256], gen_mask [1,32] -> same outputs."""
    def forward(self, fpn_0, fpn_1, fpn_2, text_feats, text_mask, gen_feats, gen_mask):
        K = text_mask.shape[0]
        scores, boxes, presence, hs, enc = self.ground(fpn_2.to(torch.float32), self.img_pos, gen_feats, gen_mask, text_feats, text_mask)
        probs = torch.sigmoid(scores.to(torch.float32)[:, :, 0]) * torch.sigmoid(presence.to(torch.float32)[:, 0])[:, None]
        sel = probs.topk(self.Q, dim=1).indices; hs_sel = torch.gather(hs, 1, sel[..., None].expand(-1, -1, hs.shape[-1]))
        tf = text_feats.to(torch.float32); tm = text_mask.to(torch.bool)
        out = self.seg(backbone_feats=[fpn_0.to(torch.float32), fpn_1.to(torch.float32), torch.zeros((1, 256, 72, 72), dtype=torch.float32)], obj_queries=hs_sel.to(torch.float32).unsqueeze(0),
                       image_ids=torch.zeros((K,), dtype=torch.long), encoder_hidden_states=enc.to(torch.float32), prompt=tf, prompt_mask=tm)
        return scores, boxes, presence, hs, sel, out["pred_masks"].to(torch.float16)
decoder = model.transformer.decoder; decoder.compile_mode = None; decoder.compiled = True
ch, cw = decoder._get_coords(72, 72, device="cpu"); decoder.compilable_cord_cache = (ch, cw); decoder.compilable_stored_size = (72, 72)
wrapper = GroundingWrapper(model.transformer.encoder, decoder, model.dot_prod_scoring).cpu().eval()
position_encoding = model.backbone.vision_backbone.position_encoding
OUT.mkdir(parents=True, exist_ok=True)
with torch.inference_mode(): img_pos1 = position_encoding(torch.zeros((1, 256, 72, 72), dtype=torch.float32))
import numpy as np
for K in ([] if args.only_mask else BUCKETS):
    img_pos = img_pos1.repeat(K, 1, 1, 1).to(torch.float16).contiguous(); np.save(OUT / f"img_pos_c{K}.npy", img_pos.numpy(), allow_pickle=False)
    inputs = (torch.zeros((K, 256, 72, 72), dtype=torch.float16), img_pos, torch.zeros((32, K, 256), dtype=torch.float16), torch.zeros((K, 32), dtype=torch.float32))
    with torch.inference_mode():
        outs = wrapper(*inputs); print("ground_c%d shapes:" % K, [tuple(o.shape) for o in outs], flush=True)
        torch.onnx.export(wrapper, inputs, str(OUT / f"ground_c{K}m.onnx"), opset_version=17, input_names=["img_feat", "img_pos", "text_feats", "text_mask"],
                          output_names=["scores", "boxes", "presence", "hs", "enc"], dynamic_axes=None, do_constant_folding=True, dynamo=False)
    print("exported ground_c%dm %.0fs" % (K, time.time() - t0), flush=True)
if args.phase_conv:
    with torch.inference_mode():
        pdx = PhasePixelDecoder(seg.pixel_decoder).eval(); feats = [torch.randn(1, 256, 288, 288), torch.randn(1, 256, 144, 144), torch.randn(3, 256, 72, 72)]
        ref = seg.pixel_decoder(feats); out = pdx(feats); print("phase decoder exactness: rel %.2e max abs %.2e" % (float((out - ref).norm() / ref.norm()), float((out - ref).abs().max())), flush=True)
    seg.pixel_decoder = pdx; suffix = "_phase"
else: suffix = ""
mw = MaskWrapper(seg).cpu().eval(); K = 2
minp = (torch.zeros((1, 256, 288, 288)), torch.zeros((1, 256, 144, 144)), torch.zeros((5184, K, 256), dtype=torch.float16), torch.zeros((32, K, 256), dtype=torch.float16), torch.zeros((K, 32)), torch.zeros((K, QSEL, 256), dtype=torch.float16))
with torch.inference_mode():
    m = mw(*minp); print("mask shapes:", tuple(m.shape), flush=True)
    torch.onnx.export(mw, minp, str(OUT / f"maskhead_q{QSEL}{suffix}.onnx"), opset_version=17, input_names=["fpn_0", "fpn_1", "enc", "text_feats", "text_mask", "hs_sel"], output_names=["masks"],
                      dynamic_axes={"enc": {1: "K"}, "text_feats": {1: "K"}, "text_mask": {0: "K"}, "hs_sel": {0: "K"}, "masks": {0: "K"}}, do_constant_folding=True, dynamo=False)
print("exported maskhead_q%d %.0fs" % (QSEL, time.time() - t0), flush=True)
if args.fused:
    for K in BUCKETS:
        fw = FusedWrapper(wrapper, seg, img_pos1.repeat(K, 1, 1, 1), QSEL).cpu().eval()
        finp = (torch.zeros((1, 256, 288, 288)), torch.zeros((1, 256, 144, 144)), torch.zeros((1, 256, 72, 72)), torch.zeros((32, K, 256), dtype=torch.float16), torch.zeros((K, 32)))
        with torch.inference_mode():
            outs = fw(*finp); print("groundmask_c%d shapes:" % K, [tuple(o.shape) for o in outs], flush=True)
            torch.onnx.export(fw, finp, str(OUT / f"groundmask_c{K}_q{QSEL}{suffix}.onnx"), opset_version=17, input_names=["fpn_0", "fpn_1", "fpn_2", "text_feats", "text_mask"],
                              output_names=["scores", "boxes", "presence", "hs", "sel", "masks"], dynamic_axes=None, do_constant_folding=True, dynamo=False)
        print("exported groundmask_c%d %.0fs" % (K, time.time() - t0), flush=True)
if args.shared_encoder:
    swrapper = SharedGroundingWrapper(model.transformer.encoder, decoder, model.dot_prod_scoring).cpu().eval()
    for K in BUCKETS:
        fw = SharedFusedWrapper(swrapper, seg, img_pos1, QSEL).cpu().eval()
        finp = (torch.zeros((1, 256, 288, 288)), torch.zeros((1, 256, 144, 144)), torch.zeros((1, 256, 72, 72)), torch.zeros((32, K, 256), dtype=torch.float16), torch.zeros((K, 32)),
                torch.zeros((32, 1, 256), dtype=torch.float16), torch.zeros((1, 32)))
        with torch.inference_mode():
            outs = fw(*finp); print("groundmaskse_c%d shapes:" % K, [tuple(o.shape) for o in outs], flush=True)
            torch.onnx.export(fw, finp, str(OUT / f"groundmaskse_c{K}_q{QSEL}{suffix}.onnx"), opset_version=17, input_names=["fpn_0", "fpn_1", "fpn_2", "text_feats", "text_mask", "gen_feats", "gen_mask"],
                              output_names=["scores", "boxes", "presence", "hs", "sel", "masks"], dynamic_axes=None, do_constant_folding=True, dynamo=False)
        print("exported groundmaskse_c%d %.0fs" % (K, time.time() - t0), flush=True)
if args.u8_masks:
    class FusedU8Wrapper(FusedWrapper):
        def forward(self, *xs):
            scores, boxes, presence, hs, sel, masks = super().forward(*xs)
            return scores, boxes, sel, (masks > 0).to(torch.uint8)
    for K in BUCKETS:
        fw = FusedU8Wrapper(wrapper, seg, img_pos1.repeat(K, 1, 1, 1), QSEL).cpu().eval()
        finp = (torch.zeros((1, 256, 288, 288)), torch.zeros((1, 256, 144, 144)), torch.zeros((1, 256, 72, 72)), torch.zeros((32, K, 256), dtype=torch.float16), torch.zeros((K, 32)))
        with torch.inference_mode():
            outs = fw(*finp); print("groundmaskU8_c%d shapes:" % K, [(tuple(o.shape), str(o.dtype)) for o in outs], flush=True)
            torch.onnx.export(fw, finp, str(OUT / f"groundmaskU8_c{K}_q{QSEL}{suffix}.onnx"), opset_version=17, input_names=["fpn_0", "fpn_1", "fpn_2", "text_feats", "text_mask"],
                              output_names=["scores", "boxes", "sel", "masks_u8"], dynamic_axes=None, do_constant_folding=True, dynamo=False)
        print("exported groundmaskU8_c%d %.0fs" % (K, time.time() - t0), flush=True)
print("EXPORT_DONE")
