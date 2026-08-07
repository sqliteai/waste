#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
deepseek_ref.py — pure-PyTorch DeepSeek-V3 / Kimi-K2, running off a WASTE container.

Companion to kimi_ref.py, which is the oracle for the Kimi-Linear family. That
one cannot serve K2: it hardcodes `linear_attn_config` (KeyError on a config
without KDA) and, more importantly, applies NO rotary at all --

    # NoPE: mla_use_nope, so no rotary is applied to the "rot" dims

which is correct for Kimi-Linear and K3 (`mla_use_nope: true`) and wrong for
every DeepSeek-V3 model. K2 sets no such flag and ships
`rope_theta: 50000` with YaRN scaling, so its `qk_rope_head_dim` dims must be
rotated. In MLA those dims are the ONLY positional signal -- the nope dims are
position-free by construction -- so omitting the rotation leaves the model
unable to order its own prompt.

Same contract as kimi_ref.py: weights come FROM THE CONTAINER, trunk
dequantized on demand and experts dequantized per use, so a diff against the C
engine measures ARITHMETIC and not quantization error. Both sides see the same
3-bit experts.

  uv run --with torch python tools/deepseek_ref.py \
      --container /data/hermes/waste_containers/kimi-k2.waste \
      --ids 163594,14062,163601,...   --top 10

`--no-rope --no-mscale` reproduces the engine exactly: layer 0's residual
stream matches its WASTE_DUMP_HIDDEN output to 0.000% rel L2 on K2. That
agreement is the reference's validation, so run it before reading any delta.

Speed: a 61-layer forward dequantizes every routed expert it touches in
Python, and the distinct experts per layer grow with the token count — minutes
for a 15-token prompt, hours for a long one. Use the shortest prompt that
reproduces the behaviour under test.
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kimi_ref import Container, rms_norm          # noqa: E402


# ------------------------------------------------------------------ yarn ---

def yarn_find_correction_dim(num_rot, dim, base, max_pos):
    return (dim * math.log(max_pos / (num_rot * 2 * math.pi))) / (2 * math.log(base))


def yarn_get_mscale(scale, mscale):
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def rope_tables(cfg, dim):
    """(inv_freq[dim/2], softmax_scale_multiplier), following
    DeepseekV3YarnRotaryEmbedding in the checkpoint's modeling_deepseek.py.

    K2 carries beta_fast = beta_slow = 1.0 rather than HF's 32/1 defaults,
    which collapses the correction range to dims 19..20: below that the
    extrapolated frequency is kept, above it the frequency is interpolated by
    1/factor. YaRN rescales inv_freq globally, so it applies at every position,
    including 0."""
    base = float(cfg.get("rope_theta", 10000.0))
    sc = cfg.get("rope_scaling")
    half = torch.arange(0, dim, 2, dtype=torch.float32) / dim
    freq_extra = 1.0 / (base ** half)
    kind = sc.get("type", sc.get("rope_type")) if sc else None
    if not sc:
        return freq_extra, 1.0
    if kind != "yarn":
        raise SystemExit(f"rope_scaling type {kind!r} is not implemented here; "
                         "the engine refuses the same shape at load")
    factor = float(sc["factor"])
    orig = float(sc.get("original_max_position_embeddings", 4096))
    bf, bs = float(sc.get("beta_fast", 32)), float(sc.get("beta_slow", 1))
    freq_inter = freq_extra / factor
    low = max(math.floor(yarn_find_correction_dim(bf, dim, base, orig)), 0)
    high = min(math.ceil(yarn_find_correction_dim(bs, dim, base, orig)), dim - 1)
    if low == high:
        high += 0.001                      # upstream's singularity guard
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    mask = 1.0 - ramp                      # 1 => extrapolate, 0 => interpolate
    inv_freq = freq_inter * (1 - mask) + freq_extra * mask
    # cos/sin carry mscale / mscale_all_dim, which is 1.0 when the two are equal
    # (K2: both 1.0). The attention scale carries mscale_all_dim squared, which
    # is 1.8133x on K2. Same name, two different factors. Unequal mscales are
    # refused rather than approximated, so this stays an oracle for exactly the
    # shapes the engine accepts.
    m_one, m_all = sc.get("mscale", 1.0), sc.get("mscale_all_dim", 0)
    if float(m_one) != float(m_all):
        raise SystemExit(f"rope_scaling mscale {m_one} != mscale_all_dim "
                         f"{m_all}; the ratio on cos/sin is not implemented "
                         "here, and the engine refuses it at load")
    att_mul = yarn_get_mscale(factor, float(m_all)) ** 2 if m_all else 1.0
    return inv_freq, att_mul


def apply_rope(x, pos, inv_freq):
    """Rotate the last dim of x [T, ..., dim] at integer positions `pos` [T].

    GPT-J / interleaved convention: pair j is (x[2j], x[2j+1]). Upstream
    de-interleaves before its half-split rotate,

        q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    and the two compose to exactly this. The LLaMA half-split form applied
    directly to these weights pairs the wrong dims and still yields finite,
    weight-shaped output."""
    ang = pos.float().unsqueeze(-1) * inv_freq                    # [T, dim/2]
    cos, sin = ang.cos(), ang.sin()
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [inv_freq.numel()]
    cos, sin = cos.view(shape), sin.view(shape)
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


# ------------------------------------------------- row-subset dequant ------

def deq_rows(c, name, rows):
    """Dequantize only `rows` of a trunk tensor.

    embed_tokens and lm_head are 163840 x 7168 on K2; materializing either in
    f32 is 4.7 GB and the Q4G/Q8G unpack needs several times that transiently.
    The forward needs a handful of embedding rows and can chunk the head, so
    neither is ever built whole."""
    e = c._meta[name]
    blob, shape = c._blob, e["shape"]
    N = shape[-1]
    if e["fmt"] == 0:
        out = torch.empty(len(rows), N)
        for i, r in enumerate(rows):
            o = e["off"] + r * N * 4
            out[i] = torch.frombuffer(bytearray(blob[o:o + N * 4]), dtype=torch.float32)
        return out
    g = e["group"]
    ng = (N + g - 1) // g
    q4 = e["fmt"] == 3
    rb = ng * g // 2 if q4 else ng * g
    out = torch.empty(len(rows), N)
    for i, r in enumerate(rows):
        o = e["off"] + r * rb
        raw = bytearray(blob[o:o + rb])
        if q4:
            b = torch.frombuffer(raw, dtype=torch.uint8).int()
            v = (torch.stack([b & 0x0F, b >> 4], -1).view(ng, g) - 8).float()
        else:
            v = torch.frombuffer(raw, dtype=torch.int8).view(ng, g).float()
        so = e["scale_off"] + r * ng * 2
        sc = torch.frombuffer(bytearray(blob[so:so + ng * 2]),
                              dtype=torch.float16).float().view(ng, 1)
        out[i] = (v * sc).view(-1)[:N]
    return out


# ----------------------------------------------------------------- model ---

class DeepseekRef:
    def __init__(self, c: Container, rope=True, verbose=False):
        self.c, self.t, self.cfg = c, c.t, c.cfg
        self.p = c.prefix
        self.eps = self.cfg["rms_norm_eps"]
        self.n_layers = self.cfg["num_hidden_layers"]
        self.first_dense = self.cfg.get("first_k_dense_replace", 0)
        self.use_rope = rope
        self.verbose = verbose
        self.qk_n = self.cfg["qk_nope_head_dim"]
        self.qk_r = self.cfg["qk_rope_head_dim"]
        self.inv_freq, self.att_mul = rope_tables(self.cfg, self.qk_r)
        # A container this size makes the trunk cache the memory ceiling: one
        # K2 layer is ~600 MB of f32 weights, so the 64 kimi_ref defaults to
        # would hold tens of gigabytes.
        c.t.cap = 6
        # No expert cache: one dequantized expert is three f32 matrices, 176 MB,
        # and experts are never reused across layers because each layer has its
        # own bank. A cache keyed on (layer, expert) would only grow — 512
        # entries want 90 GB. The grouping in moe() removes the redundant work
        # instead, decoding each distinct expert once per layer.

    def mla(self, L, x, pos):
        p = f"{self.p}model.layers.{L}.self_attn."
        cfg, T = self.cfg, x.shape[0]
        nh, qk_n, qk_r = cfg["num_attention_heads"], self.qk_n, self.qk_r
        vh, qd = cfg["v_head_dim"], self.qk_n + self.qk_r
        if cfg.get("q_lora_rank"):
            qa = rms_norm(x @ self.t[p + "q_a_proj.weight"].T,
                          self.t[p + "q_a_layernorm.weight"], self.eps)
            q = (qa @ self.t[p + "q_b_proj.weight"].T).view(T, nh, qd)
        else:
            q = (x @ self.t[p + "q_proj.weight"].T).view(T, nh, qd)
        ckv = x @ self.t[p + "kv_a_proj_with_mqa.weight"].T
        kpass, krot = ckv.split([cfg["kv_lora_rank"], qk_r], dim=-1)
        kpass = rms_norm(kpass, self.t[p + "kv_a_layernorm.weight"], self.eps)
        kb = (kpass @ self.t[p + "kv_b_proj.weight"].T).view(T, nh, qk_n + vh)
        knope, val = kb.split([qk_n, vh], dim=-1)

        if self.use_rope:
            qn, qr = q.split([qk_n, qk_r], dim=-1)
            qr = apply_rope(qr, pos, self.inv_freq)
            q = torch.cat([qn, qr], -1)
            krot = apply_rope(krot, pos, self.inv_freq)
        k = torch.cat([knope, krot.view(T, 1, qk_r).expand(T, nh, qk_r)], -1)

        scale = (qd ** -0.5) * self.att_mul
        att = torch.einsum("thd,shd->hts", q, k) * scale
        att = (att + torch.full((T, T), float("-inf")).triu(1)).softmax(-1)
        o = torch.einsum("hts,shd->thd", att, val).reshape(T, nh * vh)
        return o @ self.t[p + "o_proj.weight"].T

    def moe(self, L, x):
        p = f"{self.p}model.layers.{L}.block_sparse_moe."
        cfg, T = self.cfg, x.shape[0]
        scores = torch.sigmoid(x.float() @ self.t[p + "gate.weight"].float().T)
        choice = scores + self.t[p + "gate.e_score_correction_bias"].unsqueeze(0)
        k = cfg["num_experts_per_token"]
        idx = torch.topk(choice, k=k, dim=-1, sorted=False)[1]
        w = scores.gather(1, idx)
        if cfg.get("moe_renormalize", True):
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = w * cfg["routed_scaling_factor"]

        # Group tokens by expert so each distinct expert is decoded ONCE per
        # layer rather than once per (token, slot). On a 15-token prompt that
        # is ~90 decodes instead of 120, and the gap widens with length.
        jobs = {}
        for t in range(T):
            for j in range(k):
                jobs.setdefault(int(idx[t, j]), []).append((t, w[t, j]))
        y = torch.zeros_like(x)
        for eid, hits in jobs.items():
            E = self.c.expert(L, eid)
            ts = [t for t, _ in hits]
            xi = x[ts]
            h = F.silu(xi @ E["gate"].T) * (xi @ E["up"].T)
            o = h @ E["down"].T
            for r, (t, wt) in enumerate(hits):
                y[t] += wt * o[r]
        sg, su, sd = (self.t[p + f"shared_experts.{n}.weight"]
                      for n in ("gate_proj", "up_proj", "down_proj"))
        sh = F.silu(x @ sg.T) * (x @ su.T)
        return y + sh @ sd.T

    def dense_mlp(self, L, x):
        p = f"{self.p}model.layers.{L}.mlp."
        h = F.silu(x @ self.t[p + "gate_proj.weight"].T) * (x @ self.t[p + "up_proj.weight"].T)
        return h @ self.t[p + "down_proj.weight"].T

    def forward(self, ids, dump=None, upto=None):
        pos = torch.arange(len(ids))
        x = deq_rows(self.c, self.p + "model.embed_tokens.weight", ids)
        n = self.n_layers if upto is None else min(upto, self.n_layers)
        for L in range(n):
            pre = f"{self.p}model.layers.{L}."
            t0 = time.time()
            x = x + self.mla(L, rms_norm(x, self.t[pre + "input_layernorm.weight"], self.eps), pos)
            h = rms_norm(x, self.t[pre + "post_attention_layernorm.weight"], self.eps)
            x = x + (self.dense_mlp(L, h) if L < self.first_dense else self.moe(L, h))
            if dump:
                with open(dump, "ab" if L else "wb") as f:
                    v = x[-1].float().tolist()
                    f.write(struct.pack(f"<{len(v)}f", *v))
            if self.verbose:
                print(f"  layer {L:>3}/{n}  {time.time()-t0:6.1f}s  "
                      f"|x|={x[-1].norm():.3f}",
                      flush=True)
        if upto is not None:
            return None
        x = rms_norm(x, self.t[self.p + "model.norm.weight"], self.eps)[-1]
        # lm_head in row blocks: 163840 x 7168 is 4.7 GB in f32 and we only
        # need the resulting vector of logits.
        name = self.p + "lm_head.weight"
        V = self.c._meta[name]["shape"][0]
        out = torch.empty(V)
        B = 8192
        for beg in range(0, V, B):
            rows = list(range(beg, min(beg + B, V)))
            out[beg:beg + len(rows)] = deq_rows(self.c, name, rows) @ x
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--ids", help="comma-separated token ids (from `waste tokenize`)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--no-rope", action="store_true",
                    help="skip the rotary, i.e. what the C engine does today")
    ap.add_argument("--no-mscale", action="store_true",
                    help="drop YaRN's mscale^2 from the attention scale, which "
                         "the engine also omits; with --no-rope this reproduces "
                         "the engine")
    ap.add_argument("--dump-hidden", help="per-layer residual stream, engine's format")
    ap.add_argument("--dump", default="",
                    help="last token's logits as f32, the layout test_forward "
                         "writes — this is what tests/run.sh diffs against")
    ap.add_argument("--upto", type=int, help="stop after N layers (bisecting)")
    ap.add_argument("--threads", type=int,
                    help="torch intra-op threads; default is one per physical core")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    ids = [int(x) for x in a.ids.replace(" ", ",").split(",") if x]
    c = Container(a.container)
    m = DeepseekRef(c, rope=not a.no_rope, verbose=a.verbose)
    if a.no_mscale:
        m.att_mul = 1.0
    print(f"container {a.container}", file=sys.stderr)
    print(f"rope {'OFF (engine behaviour)' if a.no_rope else 'ON'}"
          f"  att_mul {m.att_mul:.4f}  layers {m.n_layers}  ntok {len(ids)}",
          file=sys.stderr)
    t0 = time.time()
    lg = m.forward(ids, dump=a.dump_hidden, upto=a.upto)
    if lg is None:
        print(f"stopped after {a.upto} layers, hidden dumped", file=sys.stderr)
        return 0
    if a.dump:
        v = lg.float().tolist()
        with open(a.dump, "wb") as f:
            f.write(struct.pack(f"<{len(v)}f", *v))
        print(f"dumped logits -> {a.dump}", file=sys.stderr)
    pr = lg.softmax(-1)
    top = torch.topk(lg, a.top)
    print(json.dumps({
        "prompt_tokens": len(ids),
        "rope": not a.no_rope,
        "elapsed_s": round(time.time() - t0, 1),
        "top": [{"id": int(i), "logit": round(float(v), 4),
                 "prob": round(float(pr[i]), 6)}
                for v, i in zip(top.values, top.indices)],
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
