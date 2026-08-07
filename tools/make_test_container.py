#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
make_test_container.py — a valid WASTE container of a few megabytes.

The end-to-end checks in tests/run.sh all need a container, and the
smallest real one is 19 GB. That put the engine's most valuable tests —
chunked prefill against token-at-a-time, the expert cache against no
cache, session state round-trip, the RAM budget — out of reach of CI and
of anyone who has not downloaded a model.

None of those checks look at whether the weights mean anything. They
compare the engine against itself: same tokens, two paths, same logits.
So this writes a structurally valid container full of deterministic
noise, at dimensions that make a forward pass take milliseconds.

Deliberately stdlib-only. The converter needs torch and a C extension;
requiring either here would just move the barrier rather than remove it.

  python3 tools/make_test_container.py /tmp/tiny.waste
  ./waste run /tmp/tiny.waste "hello" -n 4 --budget 1G
"""

import argparse
import json
import os
import random
import struct
import sys
import zlib

MAGIC_EXPERT = 0x50584557        # 'WEXP'
MAGIC_CODEBOOK = 0x4B424357      # 'WCBK'
ALIGN = 4096
FMT_F32, FMT_Q8G, FMT_Q4G, FMT_VQ3R = 0, 2, 3, 4
VEC_DIM, CB_ENTRIES, STAGES, IDX_BLOCK = 8, 256, 3, 64
GROUP = 128
KINDS = ("gate", "up", "down")

# A Kimi-Linear at 1/18 scale. The shapes are the ones the engine reads,
# not a guess: every dimension below is the small counterpart of something
# in a real manifest, and the layer mix keeps one dense layer, KDA layers
# and at least one Gated MLA layer so all three paths are exercised.
H_KDA, D_KDA = 4, 32                      # KDA heads, head dim -> C = 128
CFG = {
    "model_type": "kimi_linear",
    # a real converted config carries this, and `info` names the model from
    # it — model_type says kimi_linear on every model in the family
    "architectures": ["KimiLinearForCausalLM"],
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "first_k_dense_replace": 1,           # layer 0 dense, the rest MoE
    "intermediate_size": 256,             # the dense layer's FFN
    "moe_intermediate_size": 64,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "num_shared_experts": 1,
    "moe_renormalize": True,
    "moe_router_activation_func": "sigmoid",
    "routed_scaling_factor": 2.0,
    "num_attention_heads": 4,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "kv_lora_rank": 32,
    "q_lora_rank": None,
    "mla_use_nope": True,
    "rms_norm_eps": 1e-5,
    "vocab_size": 256,
    "tie_word_embeddings": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "linear_attn_config": {
        "head_dim": D_KDA,
        "num_heads": H_KDA,
        "short_conv_kernel_size": 4,
        # 1-based in the manifest, as the engine expects
        "kda_layers": [1, 2, 4],
        "full_attn_layers": [3],
    },
}
C_KDA = H_KDA * D_KDA

# --rope turns the above into a DeepSeek-V3 at the same scale, which is the
# only shape that reaches src/model.c's rotary: the Kimi models set
# mla_use_nope and pass the qk_rope dims through unrotated, so a container
# built from CFG as it stands leaves rope_init and rope_apply dead.
#
# The rope block is Kimi-K2-Instruct's config.json verbatim. DeepSeek-V3 and
# R1 ship the same shape with factor 40 and beta_fast 32; K2's beta_fast ==
# beta_slow == 1.0 is the more awkward of the two because it collapses YaRN's
# correction range to a two-dim ramp, so it is the one worth pinning.
V3_ROPE = {
    "rope_theta": 50000.0,
    "rope_scaling": {"beta_fast": 1.0, "beta_slow": 1.0, "factor": 32.0,
                     "mscale": 1.0, "mscale_all_dim": 1.0,
                     "original_max_position_embeddings": 4096,
                     "type": "yarn"},
}


def f32(vals):
    return struct.pack("<%df" % len(vals), *vals)


def f16(vals):
    return struct.pack("<%de" % len(vals), *vals)


class Trunk:
    """Accumulates trunk.bin and its index, in the layout model.c reads:
    F32 verbatim, Q8G/Q4G as quantized rows followed by one fp16 scale per
    group."""

    def __init__(self, rng, prefix=""):
        self.buf = bytearray()
        self.index = []
        self.rng = rng
        self.prefix = prefix

    def f32(self, name, shape):
        n = 1
        for s in shape:
            n *= s
        off = len(self.buf)
        self.buf += f32([self.rng.uniform(-0.5, 0.5) for _ in range(n)])
        self.index.append({"name": self.prefix + name, "fmt": FMT_F32,
                           "off": off, "shape": list(shape),
                           "bytes": len(self.buf) - off})

    def quant(self, name, shape, bits=None, prefixed=True):
        """Quantized rows plus one fp16 scale per group.

        The width mirrors tools/convert.py: 4 bits for the bulk, 8 at the
        two ends of the network where error is least forgiving. Emitting
        Q8G everywhere — which this did until a Q4G trunk turned out not to
        load under WASTE_Q8=0 — leaves the 4-bit paths with no container CI
        can reach, and a default conversion produces almost nothing else.
        """
        rows, N = 1, shape[-1]
        for s in shape[:-1]:
            rows *= s
        if bits is None:
            bits = 8 if name.endswith(("embed_tokens.weight",
                                       "lm_head.weight")) else 4
        ng = (N + GROUP - 1) // GROUP
        n = ng * GROUP          # payload is padded to whole groups, as the
        off = len(self.buf)     # converter does
        if bits == 8:
            self.buf += bytes((self.rng.randrange(-127, 128) & 0xFF)
                              for _ in range(rows * n))
        else:
            # two signed nibbles per byte, low one first, biased by +8 —
            # the packing src/model.c's 4-bit decode assumes. n is a whole
            # number of groups of 128, so the pairs never straddle a row.
            v = [self.rng.randrange(-8, 8) for _ in range(rows * n)]
            self.buf += bytes(((v[i] + 8) & 0x0F) | (((v[i + 1] + 8) & 0x0F) << 4)
                              for i in range(0, len(v), 2))
        soff = len(self.buf)
        self.buf += f16([self.rng.uniform(0.002, 0.02)
                         for _ in range(rows * ng)])
        self.index.append({"name": (self.prefix if prefixed else "") + name,
                           "fmt": FMT_Q8G if bits == 8 else FMT_Q4G,
                           "off": off, "shape": list(shape), "group": GROUP,
                           "scale_off": soff, "bytes": len(self.buf) - off})


def block_indices(idx, M, N):
    """[stages][M*N/8] -> [M/B][pos][row_in_block][stage], the layout the
    gather loop walks. Rows are padded to a whole block with zeros."""
    nvr = N // VEC_DIM
    pad = (-M) % IDX_BLOCK
    nb = (M + pad) // IDX_BLOCK
    out = bytearray(nb * nvr * IDX_BLOCK * STAGES)
    for b in range(nb):
        for v in range(nvr):
            for r in range(IDX_BLOCK):
                row = b * IDX_BLOCK + r
                dst = ((b * nvr + v) * IDX_BLOCK + r) * STAGES
                if row >= M:
                    continue                      # padding stays zero
                src = row * nvr + v
                for s in range(STAGES):
                    out[dst + s] = idx[s][src]
    return bytes(out)


def write_expert(f, layer, eid, cb_base, shapes, rng):
    hdr_size = 48
    off, offsets, body = hdr_size, [], bytearray()
    for (M, N) in shapes:
        nvec = M * N // VEC_DIM
        idx = [[rng.randrange(CB_ENTRIES) for _ in range(nvec)]
               for _ in range(STAGES)]
        offsets.append(off)
        b = block_indices(idx, M, N)
        body += b
        off += len(b)
    corr_off = off
    for (M, _N) in shapes:
        b = f16([rng.uniform(0.01, 0.05) for _ in range(M)])
        body += b
        off += len(b)

    total = hdr_size + len(body)
    blocks = (total + ALIGN - 1) // ALIGN
    hdr = struct.pack("<IHHBBHHHIIIIIIII",
                      MAGIC_EXPERT, layer, eid, FMT_VQ3R, 0, cb_base, 0, 0,
                      blocks, offsets[0], offsets[1], offsets[2], corr_off,
                      zlib.crc32(bytes(body)) & 0xFFFFFFFF, 0, 0)
    assert len(hdr) == hdr_size
    f.write(hdr)
    f.write(body)
    f.write(b"\0" * (blocks * ALIGN - total))
    return blocks * ALIGN


# ---- optional tokenizer ---------------------------------------------------
#
# Off by default: tests/run.sh states, in several places, that the synthetic
# container carries no tokenizer, and quietly changing that would turn its
# SKIPs into checks nobody asked for. `--tokenizer` is for the callers that
# need one — serve/ has to encode a chat template against real specials, and
# a 983 GB download is not a test dependency.
#
# The vocabulary is the 256 single bytes plus a few merges, which is enough
# to make BPE actually merge rather than emit one token per character, and
# the ranks are the ids, as in a tiktoken mergeable_ranks file.

MERGES = [
    b"he", b"in", b"re", b"on", b"at", b"en", b"nd", b"ti", b"es", b"or",
    b" t", b" a", b" s", b" w", b" o", b" i", b" c", b" b", b" f", b" m",
    b"the", b"ing", b"and", b" th", b" the", b" an", b" to", b" of",
    b"hello", b"world", b"weather", b"Paris", b"city", b"json", b"true",
    b"false", b"null", b"message", b"role", b"user", b"assistant", b"system",
    b"tool", b"call", b"argument", b"response", b"think", b"index", b"type",
]

# K3's four XTML markers, the media block, and the reserved block the
# tokenizer positions BOS/EOS in. Order matters: waste_tok_open puts
# [BOS] at n_tokens, [EOS] at n_tokens+1 and treats n_tokens+2 as the id
# generation_config.json names — <|end_of_msg|> on both Kimi releases.
SPECIALS = [
    "[BOS]", "[EOS]", "<|end_of_msg|>",
    "<|open|>", "<|close|>", "<|sep|>",
    "<|media_begin|>", "<|media_content|>", "<|media_pad|>", "<|media_end|>",
    "<|kimi_image_placeholder|>",
]


def write_tokenizer(outdir):
    """tokenizer.model + specials.json. Returns the total vocabulary size."""
    import base64

    tokens = [bytes([b]) for b in range(256)] + MERGES
    with open(os.path.join(outdir, "tokenizer.model"), "wb") as f:
        for rank, tok in enumerate(tokens):
            f.write(base64.b64encode(tok) + b" " + str(rank).encode() + b"\n")

    base = len(tokens)
    specials = [{"id": base + i, "text": s} for i, s in enumerate(SPECIALS)]
    with open(os.path.join(outdir, "specials.json"), "w") as f:
        json.dump(specials, f, indent=1)

    return base + len(SPECIALS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0,
                    help="same seed, byte-identical container")
    ap.add_argument("--tokenizer", action="store_true",
                    help="also write a small tiktoken-style vocabulary with "
                         "K3's XTML specials, so the container can be driven "
                         "with text instead of raw ids")
    ap.add_argument("--prefix", default="", metavar="PFX",
                    help="put the text tensors under a tensor_prefix, e.g. "
                         "language_model., and add one tensor outside it — "
                         "K3's shape, and the one the loader skips")
    ap.add_argument("--rope", action="store_true",
                    help="a DeepSeek-V3 instead of a Kimi-Linear: every layer "
                         "MLA, no mla_use_nope, and rope_theta with YaRN — the "
                         "only shape that reaches the engine's rotary")
    ap.add_argument("--qk-rope", type=int, metavar="N",
                    help="override qk_rope_head_dim. With --rope, a slice "
                         "wider than the build's WASTE_MAX_ROPE_HALF pair "
                         "table has to be refused at load, not run unrotated")
    ap.add_argument("--nope-false", action="store_true",
                    help="write mla_use_nope: false rather than omitting it. "
                         "Same model either way — a loader that tests the key "
                         "for presence reads it as NoPE and skips the rotation")
    ap.add_argument("--rope-type", metavar="T",
                    help="override rope_scaling.type, e.g. linear — a scaling "
                         "the engine does not implement has to be refused, not "
                         "quietly run as plain RoPE")
    ap.add_argument("--mscale", type=float, metavar="X",
                    help="override rope_scaling.mscale, leaving mscale_all_dim "
                         "at 1.0. Unequal mscales put a ratio on cos/sin that "
                         "the engine does not apply, so it refuses instead")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    cfg = dict(CFG)
    if args.rope:
        # Dropping linear_attn_config is what makes every layer MLA, so the
        # rotation is exercised at depth rather than in the one full-attention
        # layer the Kimi mix leaves. It also makes the container readable by
        # tools/deepseek_ref.py, which — unlike kimi_ref.py — does not index
        # linear_attn_config and does apply the rotary.
        del cfg["mla_use_nope"], cfg["linear_attn_config"]
        cfg["model_type"] = "deepseek_v3"
        cfg["architectures"] = ["DeepseekV3ForCausalLM"]
        cfg.update(V3_ROPE)
        if args.nope_false:
            cfg["mla_use_nope"] = False
        if args.rope_type:
            cfg["rope_scaling"] = dict(cfg["rope_scaling"], type=args.rope_type)
        if args.mscale is not None:
            cfg["rope_scaling"] = dict(cfg["rope_scaling"], mscale=args.mscale)
    if args.qk_rope:
        cfg["qk_rope_head_dim"] = args.qk_rope
    if args.tokenizer:
        # Every special has to be a real row of the embedding table and the
        # head: a container whose vocab_size stops short of its own specials
        # is one where tokenizing a chat template indexes out of the model.
        vocab = write_tokenizer(args.out)
        cfg["vocab_size"] = vocab
        cfg["bos_token_id"] = vocab - len(SPECIALS)
        cfg["eos_token_id"] = vocab - len(SPECIALS) + 2   # <|end_of_msg|>
    hid = cfg["hidden_size"]
    nh, vh = cfg["num_attention_heads"], cfg["v_head_dim"]
    qd = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    kvl, rope = cfg["kv_lora_rank"], cfg["qk_rope_head_dim"]
    moe, dense = cfg["moe_intermediate_size"], cfg["intermediate_size"]
    kda = {l - 1 for l in cfg.get("linear_attn_config", {}).get("kda_layers", [])}

    t = Trunk(rng, args.prefix)
    t.quant("model.embed_tokens.weight", [cfg["vocab_size"], hid])
    for L in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{L}."
        t.f32(p + "input_layernorm.weight", [hid])
        t.f32(p + "post_attention_layernorm.weight", [hid])
        if L in kda:
            a = p + "self_attn."
            for w in ("q_proj", "k_proj", "v_proj"):
                t.quant(a + w + ".weight", [C_KDA, hid])
            t.quant(a + "b_proj.weight", [H_KDA, hid])
            t.quant(a + "f_a_proj.weight", [D_KDA, hid])
            t.quant(a + "f_b_proj.weight", [C_KDA, D_KDA])
            t.quant(a + "g_a_proj.weight", [D_KDA, hid])
            t.quant(a + "g_b_proj.weight", [C_KDA, D_KDA])
            t.quant(a + "o_proj.weight", [hid, C_KDA])
            for w in ("q_conv1d", "k_conv1d", "v_conv1d"):
                t.f32(a + w + ".weight", [C_KDA, 1, 4])
            t.f32(a + "A_log", [1, 1, H_KDA, 1])
            t.f32(a + "dt_bias", [C_KDA])
            t.f32(a + "o_norm.weight", [D_KDA])
        else:
            a = p + "self_attn."
            t.quant(a + "q_proj.weight", [nh * qd, hid])
            t.quant(a + "kv_a_proj_with_mqa.weight", [kvl + rope, hid])
            t.f32(a + "kv_a_layernorm.weight", [kvl])
            t.quant(a + "kv_b_proj.weight", [nh * (cfg["qk_nope_head_dim"] + vh), kvl])
            t.quant(a + "o_proj.weight", [hid, nh * vh])
        if L < cfg["first_k_dense_replace"]:
            t.quant(p + "mlp.gate_proj.weight", [dense, hid])
            t.quant(p + "mlp.up_proj.weight", [dense, hid])
            t.quant(p + "mlp.down_proj.weight", [hid, dense])
        else:
            m = p + "block_sparse_moe."
            t.quant(m + "gate.weight", [cfg["num_experts"], hid])
            t.f32(m + "gate.e_score_correction_bias", [cfg["num_experts"]])
            sh = moe * cfg["num_shared_experts"]
            t.quant(m + "shared_experts.gate_proj.weight", [sh, hid])
            t.quant(m + "shared_experts.up_proj.weight", [sh, hid])
            t.quant(m + "shared_experts.down_proj.weight", [hid, sh])
    t.f32("model.norm.weight", [hid])
    t.quant("lm_head.weight", [cfg["vocab_size"], hid])
    if args.prefix:
        # One tensor outside the prefix, which is what the loader declines
        # to load: it sets on_disk and continues before `group` is
        # assigned, and the row-scratch sizing then divided by that zero.
        # arm64's sdiv answers 0 and x86's idiv raises #DE, so K3 was an
        # instant SIGFPE on every x86 build while this suite stayed green
        # — a prefix-less container has no tensor that takes the skip.
        # Naming it vision_tower is K3's case; any name outside the prefix
        # reproduces it. issue #10.
        t.quant("vision_tower.encoder.layers.0.fc0.weight", [128, hid],
                prefixed=False)
    with open(os.path.join(args.out, "trunk.bin"), "wb") as f:
        f.write(t.buf)

    # one codebook per (kind, stage) per MoE layer
    shapes = [(moe, hid), (moe, hid), (hid, moe)]      # gate, up, down
    moe_layers = [L for L in range(cfg["num_hidden_layers"])
                  if L >= cfg["first_k_dense_replace"]]
    layers, cb_base = {}, 0
    with open(os.path.join(args.out, "codebooks.bin"), "wb") as cf:
        for L in moe_layers:
            for ki in range(len(KINDS)):
                for si in range(STAGES):
                    cid = cb_base + ki * STAGES + si
                    cf.write(struct.pack("<IHBBII", MAGIC_CODEBOOK,
                                         cid & 0xFFFF, FMT_VQ3R, VEC_DIM,
                                         CB_ENTRIES, 0))
                    cf.write(f16([rng.uniform(-0.3, 0.3)
                                  for _ in range(CB_ENTRIES * VEC_DIM)]))
            name = f"experts-L{L}.bin"
            with open(os.path.join(args.out, name), "wb") as bf:
                total = sum(write_expert(bf, L, e, cb_base, shapes, rng)
                            for e in range(cfg["num_experts"]))
            layers[str(L)] = {"file": name, "experts": cfg["num_experts"],
                              "bytes": total, "codebook_base": cb_base}
            cb_base += len(KINDS) * STAGES

    manifest = {
        "format_version": 0,
        "arch": cfg["model_type"],
        "tensor_prefix": args.prefix,
        "config": cfg,
        "expert_quant": {"fmt": "VQ3R", "stages": STAGES, "vec_dim": VEC_DIM,
                         "entries": CB_ENTRIES, "index_block": IDX_BLOCK,
                         "bits_per_weight": STAGES},
        "layers": layers,
        "trunk": t.index,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out))
    print(f"wrote {args.out}: {cfg['num_hidden_layers']} layers, "
          f"{cfg['num_experts']} experts, {total / (1 << 20):.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
