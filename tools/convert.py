#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
convert.py — safetensors (Kimi family) -> WASTE container.

Writes the layout in docs/FORMAT.md with records binary-compatible with
src/waste_format.h: 4 KiB-aligned expert records holding gate/up/down
adjacently so one pread() yields a whole expert.

Experts use VQ3R by default: 3-stage residual VQ over 8-dim vectors,
256 entries per stage (3.0 bits/weight), plus one FP16 scale per output
channel. Codebooks are trained per (layer, matrix kind) on a sample and
stored once. Gate 3 measured this recipe on real Kimi experts.

Reads a Kimi checkpoint as published — the 1.42 TB of moonshotai/Kimi-K3
that tools/fetch_weights.sh leaves on the staging disk, or any other
member of the family (Kimi-Linear) by pointing --src elsewhere.

  uv run --with torch python tools/convert.py \
      --src /path/to/hf-checkpoint \
      --out model.waste \
      --layers 1,2                       # subset for a fast first pass

Resumable: a layer whose bank file already exists is skipped. Never holds
more than one layer of experts in memory.

Peak *disk*, though, was the source plus the container — 1.42 TB of K3
staging alongside a 982 GiB container, which is two disks. --reclaim
removes that: every tensor here has exactly one consumer (the expert layer
it belongs to, or the single trunk pass), so a shard whose last consumer
has finished will never be opened again and is deleted while the run
continues. Peak staging becomes the container plus the shards still owed.

  ... --reclaim dry     # say which shards it would delete, delete nothing
  ... --reclaim on      # delete them

It is off by default and it is not reversible: a reclaimed shard has to be
downloaded again, and tools/verify_container.py can no longer check the
container against its source. Prove a recipe with that first, reclaim on
the runs after.
"""

import argparse
import io
import json
import os
import shutil
import struct
import sys
import time
import zlib

import torch


def atomic_copyfile(src, dst):
    """Copy a published sidecar without exposing a partially written file."""
    tmp = dst + ".tmp"
    with open(src, "rb") as inp, open(tmp, "wb") as out:
        while True:
            chunk = inp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, dst)


def atomic_json(path, value):
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as out:
        json.dump(value, out, indent=1)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def atomic_text(path, value):
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as out:
        out.write(value)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxfp4 import ST, unblock_scale                             # noqa: E402

# --- native VQ encoder (optional; ~15x the torch path) --------------------
_VQ = None


def _load_vq():
    """libwastevq: the assign fused with the argmin, so the [n, 256] distance
    matrix never exists. torch has to materialize it — 1.4 GB per stage for
    one expert matrix — which is what made conversion memory-bound."""
    global _VQ
    if _VQ is not None:
        return _VQ or None
    import ctypes
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("libwastevq.dylib", "libwastevq.so", "libwastevq.dll"):
        path = os.path.join(here, name)
        if os.path.exists(path):
            lib = ctypes.CDLL(path)
            lib.waste_vq_encode.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
            lib.waste_vq_encode.restype = None
            _VQ = (lib, ctypes)
            return _VQ
    _VQ = False
    return None

MAGIC_EXPERT = 0x50584557        # 'WEXP'
MAGIC_CODEBOOK = 0x4B424357      # 'WCBK'
ALIGN = 4096
FMT_F32, FMT_F16, FMT_Q8G, FMT_Q4G, FMT_VQ3R, FMT_VQ2R = 0, 1, 2, 3, 4, 5
FMT_Q3G = 7
FMT_VQ4P = 8
VEC_DIM = 8
CB_ENTRIES = 256
TRAIN_VECTORS = 300000       # vectors k-means sees per (layer, matrix kind)
IDX_BLOCK = 64          # rows per index block; matches VQ_TILE in the engine
# Two MoE tensor namings exist in this family and they are not compatible:
#
#   Mixtral / Kimi-Linear / K3   layers.L.block_sparse_moe.experts.E.{w1,w3,w2}.weight
#   DeepSeek / Kimi K2           layers.L.mlp.experts.E.{gate,up,down}_proj.weight
#
# The engine only knows the first one — it looks up the router, the shared experts and
# the score-correction bias under `block_sparse_moe` (src/model.c:2532 among others).
# So a DeepSeek-named checkpoint is READ under its own names and WRITTEN under the
# engine's. Detected per checkpoint, never flagged: a flag is a thing to get wrong, and
# `st.have()` already knows the answer.
#
# `kind` is what this file calls the matrix; `tag` is the suffix the source uses.
MOE_LAYOUTS = (
    # name        moe segment          gate/up/down source tags
    ("mixtral",   "block_sparse_moe",  ("w1", "w3", "w2")),
    ("deepseek",  "mlp",               ("gate_proj", "up_proj", "down_proj")),
)
KIND_ORDER = ("gate", "up", "down")
KINDS = tuple(zip(KIND_ORDER, MOE_LAYOUTS[0][2]))   # default: Mixtral naming

# The two families disagree on MoE *config* keys as well as on tensor names,
# and this half is the more dangerous one: the engine reads `num_experts` and
# a DeepSeek config only spells it `n_routed_experts`, so it loads 0 experts
# and refuses the container with no diagnostic — after the conversion has
# already run. `num_experts_per_tok` vs `..._per_token` is one letter and
# would leave top_k at 0.
#
# The manifest is WASTE's format, not HF's, so it is normalised here for the
# same reason the tensor names are: one spelling reaches the engine. Written
# only when absent, so a config that already uses the canonical key wins.
CONFIG_ALIASES = (
    ("num_experts",           "n_routed_experts"),
    ("num_experts_per_token", "num_experts_per_tok"),
    ("num_shared_experts",    "n_shared_experts"),
)
# `moe_renormalize` is keyed on the field being PRESENT, not on its value, so
# a plain alias of DeepSeek's `norm_topk_prob` would silently turn
# renormalisation on for a checkpoint that sets it false. Emit only when true.
CONFIG_FLAG_ALIASES = (("moe_renormalize", "norm_topk_prob"),)


def normalise_cfg(cfg):
    """A copy of the HF config with MoE keys under the names the engine reads."""
    out = dict(cfg)
    for canon, hf in CONFIG_ALIASES:
        if canon not in out and hf in out:
            out[canon] = out[hf]
    for canon, hf in CONFIG_FLAG_ALIASES:
        if canon not in out and out.get(hf):
            out[canon] = True
    return out


def moe_layout(st, prefix, layer):
    """Which of MOE_LAYOUTS this checkpoint uses, from what is actually on disk."""
    for name, seg, tags in MOE_LAYOUTS:
        probe = f"{prefix}model.layers.{layer}.{seg}.experts.0.{tags[0]}.weight"
        if st.have(probe) or st.have(probe + "_packed"):
            return name, seg, tuple(zip(KIND_ORDER, tags))
    return None, None, None


# Trunk tensors the engine expects under `block_sparse_moe` but a DeepSeek checkpoint
# ships under `mlp`. The router gate and the shared experts, not the dense-layer FFN:
# `mlp.gate.weight` and `mlp.shared_experts.*` are MoE, while `mlp.gate_proj.weight`
# is layer 0's dense FFN and the engine wants that one left exactly where it is.
def trunk_rename(name, seg):
    if seg != "mlp":
        return name
    for tail in (".mlp.gate.weight", ".mlp.gate.e_score_correction_bias"):
        if name.endswith(tail):
            return name[: -len(tail)] + tail.replace(".mlp.", ".block_sparse_moe.", 1)
    if ".mlp.shared_experts." in name:
        return name.replace(".mlp.shared_experts.", ".block_sparse_moe.shared_experts.", 1)
    return name


# ---------------------------------------------------------------- reading --

class ShardReader:
    """Lazy safetensors reader over a sharded model directory."""

    def __init__(self, model_dir):
        self.dir = model_dir
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
        self.wm = idx["weight_map"]
        self._hdr = {}
        # fp8 block size is a property of the checkpoint and is stated in its
        # config; it must not be inferred from the weight/scale shape ratio,
        # which is ambiguous whenever a dimension is not a multiple of the tile.
        self.fp8_block = (128, 128)
        cfgp = os.path.join(model_dir, "config.json")
        if os.path.exists(cfgp):
            qc = (json.load(open(cfgp)).get("quantization_config") or {})
            if qc.get("weight_block_size"):
                self.fp8_block = tuple(qc["weight_block_size"])

    def _header(self, fn):
        if fn not in self._hdr:
            with open(os.path.join(self.dir, fn), "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                self._hdr[fn] = (json.loads(f.read(hlen)), 8 + hlen)
        return self._hdr[fn]

    def names(self):
        return self.wm.keys()

    def get(self, name):
        fn = self.wm[name]
        hdr, base = self._header(fn)
        meta = hdr[name]
        beg, end = meta["data_offsets"]
        with open(os.path.join(self.dir, fn), "rb") as f:
            f.seek(base + beg)
            raw = f.read(end - beg)
        dt = {"BF16": torch.bfloat16, "F16": torch.float16,
              "F32": torch.float32,
              # The current generation of large MoEs ships fp8 with one f32
              # scale per weight_block_size tile in a companion tensor (K2,
              # DeepSeek V3/R1). Reading the values without applying those
              # scales yields plausible-looking garbage, so a missing companion
              # is an error rather than a fallback.
              "F8_E4M3": torch.float8_e4m3fn,
              "F8_E5M2": torch.float8_e5m2}[meta["dtype"]]
        t = torch.frombuffer(bytearray(raw), dtype=dt).view(*meta["shape"])
        if dt in (torch.float8_e4m3fn, torch.float8_e5m2):
            sname = name + "_scale_inv"
            if sname not in self.wm:
                raise KeyError(f"{name} is {meta['dtype']} but {sname} is missing; "
                               "refusing to read fp8 without its block scales")
            return unblock_scale(t.float(), self.get(sname), self.fp8_block)
        return t.float()


# ------------------------------------------------------------ quantizers --

def train_codebooks(X, n_stages, dev, iters=10, sample=300000, seed=0,
                    entries=None):
    """Residual k-means: one codebook per stage, fitted on the running
    residual. X is [n, VEC_DIM] on `dev`."""
    entries = entries or CB_ENTRIES
    g = torch.Generator(device="cpu").manual_seed(seed)
    books = []
    resid = X[torch.randperm(X.shape[0], generator=g)[:sample]].to(dev)
    for _ in range(n_stages):
        C = resid[torch.randperm(resid.shape[0], generator=g)[:entries]].clone()
        for _ in range(iters):
            idx = assign(resid, C)
            for j in range(entries):
                m = idx == j
                if m.any():
                    C[j] = resid[m].mean(0)
        books.append(C)
        resid = resid - C[assign(resid, C)]
    return books


def assign(X, C, chunk=1 << 20):
    """Nearest centroid via the ||x||^2 - 2x.c + ||c||^2 expansion (a GEMM)."""
    cn = (C * C).sum(1)
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for s in range(0, X.shape[0], chunk):
        x = X[s:s + chunk]
        d = cn.unsqueeze(0) - 2.0 * (x @ C.T)
        out[s:s + chunk] = d.argmin(1)
    return out


def block_indices(idx, M, N, stages):
    """Reorder [stages, nvec] -> [M/B][v][row_in_block][stage].

    The engine walks a tile of rows for one vector position at a time; in
    row-major order those rows sit N/8*stages bytes apart, so each one is a
    separate cache line. Blocked, a tile's indices for a given position are
    contiguous. Measured 1.44x on the gather loop.
    """
    nvr = N // VEC_DIM
    t = idx.view(stages, M, nvr).permute(1, 2, 0)            # [M, nvr, st]
    pad = (-M) % IDX_BLOCK
    if pad:
        t = torch.cat([t, torch.zeros(pad, nvr, stages, dtype=t.dtype)], 0)
    nb = t.shape[0] // IDX_BLOCK
    t = t.view(nb, IDX_BLOCK, nvr, stages).permute(0, 2, 1, 3)
    return t.contiguous()


def block_indices_packed(idx, M, N, stages):
    """block_indices, then four 6-bit fields squeezed into three bytes.

    Same [M/B][v][row_in_block][...] blocking; only the trailing per-row run
    changes, from four whole bytes to three packed ones. That keeps VQ4P at
    3.00 bits/weight — the same record size as VQ3R — while giving each
    stage a 64-byte table that a single NEON vqtbl4q can address.

    Little-endian bit order, LSB of stage 0 at bit 0, because the unpack in
    the engine is a byte shift and mask per stage and any other order costs
    it a shuffle:

        byte0 = s0 | s1<<6      byte1 = s1>>2 | s2<<4     byte2 = s2>>4 | s3<<2
    """
    t = block_indices(idx, M, N, stages)          # [nb, nvr, B, stages] u8
    if stages != 4:
        raise ValueError(f"packed indices need 4 stages, got {stages}")
    if int(t.max()) > 63:
        raise ValueError("packed indices need entries <= 64")
    s = [t[..., i].to(torch.int32) for i in range(4)]
    b0 = (s[0] | (s[1] << 6)) & 0xFF
    b1 = ((s[1] >> 2) | (s[2] << 4)) & 0xFF
    b2 = ((s[2] >> 4) | (s[3] << 2)) & 0xFF
    return torch.stack([b0, b1, b2], dim=-1).to(torch.uint8).contiguous()


def quantize_vq(W, books, dev, entries=None):
    """W [out, in] -> (indices uint8 [stages, nvec], per-channel fp16 scale)."""
    entries = entries or CB_ENTRIES
    M, N = W.shape
    scale = W.abs().amax(-1, keepdim=True).clamp(min=1e-8)

    vq = _load_vq()
    if vq is not None:
        lib, ctypes = vq
        X = (W / scale).reshape(-1, VEC_DIM).contiguous().float()
        B = torch.stack([C.detach().cpu().float() for C in books]).contiguous()
        n, st = X.shape[0], len(books)
        out = torch.empty(n * st, dtype=torch.uint8)
        fp = ctypes.POINTER(ctypes.c_float)
        # nthreads=0 means "every core", capped at 64 — right on a laptop,
        # thrash on a many-core box with --jobs > 1. Size the native pool to
        # the same per-worker share as torch's intra-op pool (set in main()).
        nthreads = int(os.environ.get("OMP_NUM_THREADS") or 0)
        lib.waste_vq_encode(
            ctypes.cast(X.data_ptr(), fp), n,
            ctypes.cast(B.data_ptr(), fp), st, entries, VEC_DIM,
            ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_uint8)), nthreads)
        return out.view(n, st).T.contiguous(), scale.half().flatten().cpu()

    X = (W / scale).to(dev).reshape(-1, VEC_DIM)
    idxs, resid = [], X
    for C in books:
        i = assign(resid, C)
        idxs.append(i.to(torch.uint8).cpu())
        resid = resid - C[i]
    return torch.stack(idxs), scale.half().flatten().cpu()


def quantize_q3g(W, group=128):
    """3 bits per weight, packed per row, fp16 scale per group.

    Values live in [-4, 3] biased by +4 into 0..7, written LSB-first at bit
    offset 3*i within the row, and each row is padded to a fixed stride
    that includes one guard byte so the decoder's two-byte read is always
    in bounds. The engine uses the identical indexing.

    Packing the whole tensor as one stream instead of per row is the
    obvious shortcut and it is wrong: the decoder addresses rows by a fixed
    stride, and with a guard byte in that stride every row after the first
    lands one byte off. It is invisible at 4 bits, where a row is a whole
    number of bytes with no guard needed.
    """
    orig = W.shape
    X = W.reshape(-1, orig[-1])
    rows, N = X.shape
    pad = (-N) % group
    if pad:
        X = torch.nn.functional.pad(X, (0, pad))
    ng = X.shape[1] // group
    n = ng * group
    Xg = X.view(rows, ng, group)
    scale = Xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 3.0
    u = (torch.clamp(torch.round(Xg / scale), -4, 3).to(torch.int32) + 4).view(rows, n)

    rowbytes = (n * 3 + 7) // 8 + 1                   # + guard
    buf = torch.zeros(rows, rowbytes, dtype=torch.int32)
    off = torch.arange(n) * 3
    byte, shift = off // 8, off % 8
    idx = byte.unsqueeze(0).expand(rows, n)
    lo = ((u << shift) & 0xFF).to(torch.int32).contiguous()
    buf.scatter_add_(1, idx.contiguous(), lo)
    hi = torch.where(shift > 5, (u >> (8 - shift)) & 0xFF,
                     torch.zeros_like(u)).to(torch.int32).contiguous()
    buf.scatter_add_(1, (byte + 1).unsqueeze(0).expand(rows, n).contiguous(), hi)
    return buf.to(torch.uint8).flatten(), scale.half().flatten(), list(orig)


def quantize_q4g(W, group=128):
    """int4 packed two per byte (low nibble first), fp16 scale per group.

    The trunk is the RAM floor and K3's is 54 B parameters — at 8 bits that
    is 54 GB, over the budget of the machine this is meant to run on. At
    4 bits it is 27."""
    orig = W.shape
    X = W.reshape(-1, orig[-1])
    N = X.shape[-1]
    pad = (-N) % group
    if pad:
        X = torch.nn.functional.pad(X, (0, pad))
    Xg = X.view(X.shape[0], -1, group)
    scale = Xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7.0
    Q = torch.clamp(torch.round(Xg / scale), -8, 7).to(torch.int32).flatten()
    nib = (Q + 8).to(torch.uint8) & 0x0F                 # 0..15, biased
    packed = (nib[0::2] | (nib[1::2] << 4)).contiguous()
    return packed, scale.half().flatten(), list(orig)


def quantize_q8g(W, group=128):
    """int8 + fp16 scale per group of `group` inputs. Returns (bytes, meta)."""
    orig = W.shape
    X = W.reshape(-1, orig[-1])
    N = X.shape[-1]
    pad = (-N) % group
    if pad:
        X = torch.nn.functional.pad(X, (0, pad))
    Xg = X.view(X.shape[0], -1, group)
    scale = Xg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
    Q = torch.clamp(torch.round(Xg / scale), -127, 127).to(torch.int8)
    return Q.flatten(), scale.half().flatten(), list(orig)


# ---------------------------------------------------------------- writing --

def raw_bytes(t):
    """Contiguous little-endian bytes of a tensor (torch has no .tobytes())."""
    t = t.detach().cpu().contiguous()
    n = t.numel() * t.element_size()
    buf = torch.empty(n, dtype=torch.uint8)
    buf.view(t.dtype)[:t.numel()] = t.flatten()
    return bytes(memoryview(buf.numpy() if False else bytearray(buf.tolist())))

def write_expert_record(f, layer, eid, cb_base, payloads, scales, shapes,
                        packed=False):
    """One 4 KiB-aligned WEXP record: header, then gate|up|down indices,
    then the per-channel scales for all three."""
    hdr_size = 48
    off = hdr_size
    offsets = []
    body = bytearray()
    blk = block_indices_packed if packed else block_indices
    for i, p in enumerate(payloads):          # [stages, nvec] uint8
        offsets.append(off)
        M, N = shapes[i]
        b = raw_bytes(blk(p, M, N, p.shape[0]))
        body += b
        off += len(b)
    corr_off = off
    for s in scales:
        b = raw_bytes(s)
        body += b
        off += len(b)

    total = hdr_size + len(body)
    blocks = (total + ALIGN - 1) // ALIGN
    pad = blocks * ALIGN - total
    crc = zlib.crc32(bytes(body)) & 0xFFFFFFFF

    hdr = struct.pack("<IHHBBHHHIIIIIIII",
                      MAGIC_EXPERT, layer, eid,
                      FMT_VQ4P if packed else FMT_VQ3R, 0, cb_base, 0, 0,
                      blocks, offsets[0], offsets[1], offsets[2], corr_off,
                      crc, 0, 0)
    assert len(hdr) == hdr_size, len(hdr)
    f.write(hdr)
    f.write(body)
    f.write(b"\0" * pad)
    return blocks


def bank_is_sound(path, layer, n_exp):
    """True if `path` is a whole bank: n_exp well-formed records for `layer`,
    tiling the file exactly, each one naming the expert the index will ask
    for at that offset.

    Costs 48 bytes per record — it walks the file by each record's own block
    count — and catches the failure that makes --reclaim dangerous: a bank
    truncated by a kill, a full disk, or a rename that landed torn. It says
    nothing about the *numbers* in the payloads; that is
    tools/verify_container.py's job, and it needs the source weights that
    reclaiming deletes. Which is the order to work in: prove a recipe with
    verify_container.py once, reclaim on the runs after.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    hdr_size, seen, off = 48, 0, 0
    with open(path, "rb") as f:
        while off < size:
            f.seek(off)
            raw = f.read(hdr_size)
            if len(raw) < hdr_size:
                return False
            (magic, L, eid, _fmt, _flags, _cb, lowrank, _r0, blocks,
             g_off, u_off, d_off, corr_off, _crc, _r1, _r2) = struct.unpack(
                 "<IHHBBHHHIIIIIIII", raw)
            total = blocks * ALIGN
            if (magic != MAGIC_EXPERT or L != layer or eid != seen
                    or lowrank != 0 or blocks < 1 or off + total > size
                    or not hdr_size <= g_off < u_off < d_off < corr_off <= total):
                return False
            off += total
            seen += 1
    return seen == n_exp and off == size


class ShardDebt:
    """Which source shards some part of this conversion still has to read.

    Peak staging is otherwise the source *plus* the container — 1.42 TB and
    982 GiB for K3, which is two disks. But a shard holds whole tensors, and
    every tensor here has exactly one consumer: the expert layer it belongs
    to, or the single trunk pass. Once the last consumer of a shard has
    published its output, nothing in this script will open that shard again,
    so it can go while the conversion is still running.

    This class only tracks and reports. Deleting is `release`, and main()
    calls it only under --reclaim.
    """

    TRUNK = ("trunk",)

    @staticmethod
    def name(who):
        return f"layer {who[1]}" if who[0] == "layer" else " ".join(map(str, who))

    @staticmethod
    def ledger(src):
        """Shards an earlier run already reclaimed from this checkpoint.

        Without it a resumed conversion cannot tell a shard it consumed
        itself from one that never downloaded, and has to refuse both. It
        lives beside fetch_weights.sh's .download-state because it describes
        the same thing: what is, and is no longer, in the staging area.
        """
        path = os.path.join(src, ".reclaimed")
        try:
            with io.open(path, encoding="utf-8") as f:
                return set(f.read().split())
        except OSError:
            return set()

    def __init__(self, weight_map, src):
        self.src = src
        self.ledger_path = os.path.join(src, ".reclaimed")
        self.owed = {}            # shard -> consumers that have not finished
        self.held_by = {}         # consumer -> the shards it is holding up
        for name, shard in weight_map.items():
            # A shard an earlier run already reclaimed is not staging this
            # one can give back, and counting it would report a saving twice.
            if not os.path.exists(os.path.join(src, shard)):
                continue
            who = self.consumer(name)
            self.owed.setdefault(shard, set()).add(who)
            self.held_by.setdefault(who, set()).add(shard)
        self.freed = 0
        self.released = []

    @staticmethod
    def consumer(name):
        """The one part of this script that reads `name`.

        The trunk pass takes everything that is not an expert. mxfp4 stores a
        tensor as a _packed/_scale pair and ST rejoins them, so both halves
        belong wherever the tensor they encode does.
        """
        base = name
        for suffix in ("_packed", "_scale"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
        if ".experts." not in base:
            return ShardDebt.TRUNK
        parts = base.split(".")
        try:
            return ("layer", int(parts[parts.index("layers") + 1]))
        except (ValueError, IndexError):
            # An expert whose layer cannot be read is an expert nobody can
            # prove is spent. Hold its shards forever rather than guess.
            return ("unreadable", base)

    def release(self, who, delete):
        """Mark a consumer finished. Returns the (shard, bytes) it freed,
        having deleted them when `delete`, or merely counted them when not."""
        freed = []
        for shard in sorted(self.held_by.get(who, ())):
            pending = self.owed.get(shard)
            if pending is None:
                continue              # already released by another consumer
            pending.discard(who)
            if pending:
                continue
            path = os.path.join(self.src, shard)
            try:
                size = os.path.getsize(path)
            except OSError:
                del self.owed[shard]
                continue
            if delete:
                # Ledger first: a shard recorded but not yet deleted costs a
                # resumed run nothing, while one deleted but not recorded is
                # indistinguishable from a download that never finished.
                try:
                    with io.open(self.ledger_path, "a", encoding="utf-8") as f:
                        f.write(shard + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                    os.remove(path)
                except OSError as e:
                    print(f"  reclaim: cannot remove {shard}: {e}",
                          file=sys.stderr)
                    continue
            del self.owed[shard]
            self.freed += size
            self.released.append(shard)
            freed.append((shard, size))
        return freed

    def still_owed(self):
        """Shards no consumer has released, and the bytes they hold."""
        total = 0
        for shard in self.owed:
            try:
                total += os.path.getsize(os.path.join(self.src, shard))
            except OSError:
                pass
        return len(self.owed), total


def human(n):
    """Sizes here run from a test fixture to 1.4 TB; print both legibly."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def reclaim(debt, mode, who, what):
    """Release `who`'s shards and say what that did, or would do."""
    if debt is None:
        return
    freed = debt.release(who, mode == "on")
    if not freed:
        return
    verb = "freed" if mode == "on" else "would free"
    print(f"  reclaim: {what} {verb} {len(freed)} shard(s), "
          f"{human(sum(s for _, s in freed))} "
          f"({human(debt.freed)} total)", flush=True)


def bank_codebook_base(path):
    """The codebook base a finished bank was written with, or -1.

    Every record carries it absolutely — the engine reads codebook_id from
    the record, not from the manifest — so a bank that already exists has a
    base that is not ours to reassign. Reading it back is what lets a run
    interrupted before it published a manifest still resume.
    """
    try:
        with open(path, "rb") as f:
            hdr = f.read(12)
    except OSError:
        return -1
    if len(hdr) < 12 or struct.unpack_from("<I", hdr, 0)[0] != MAGIC_EXPERT:
        return -1
    return struct.unpack_from("<H", hdr, 10)[0]


# ------------------------------------------------------------- worker ----

def convert_layer(job):
    """One layer, in its own process. Layers share nothing: separate bank
    file and separate codebook file. The parent decides the base — from the
    published manifest, from an existing bank's own records, or new after
    the published record count — and never renumbers a bank that exists."""
    (L, src, out, prefix, n_exp, stages, entries, index_bits, device,
     cb_sample, cb_base, cached_ok) = job
    import time as _t
    bank = os.path.join(out, f"experts-L{L}.bin")
    cbf = os.path.join(out, f"codebooks-L{L}.bin")
    # Whether a finished bank counts as done is the parent's call: it is the
    # only side that can see the manifest, the merged codebooks and this
    # bank's own base together. Deciding it here — from the bank's existence
    # alone — is what used to renumber a resumed layer's codebooks.
    if cached_ok and os.path.exists(bank):
        return (L, os.path.getsize(bank), cb_base, "cached")

    st = ST(src)
    dev = torch.device(device)

    lname, seg, kinds = moe_layout(st, prefix, L)
    if lname is None:
        return (L, 0, cb_base, "missing")

    def ename(e, tag):
        return f"{prefix}model.layers.{L}.{seg}.experts.{e}.{tag}.weight"

    t0 = _t.time()
    shapes = [tuple(st.tensor(ename(0, tag)).shape) for _, tag in kinds]

    books, sample_ids = {}, list(range(0, n_exp, max(1, n_exp // cb_sample)))[:cb_sample]
    per = max(1, TRAIN_VECTORS // len(sample_ids))
    with open(cbf + ".tmp", "wb") as cf:
        for ki, (kind, tag) in enumerate(kinds):
            chunks = []
            for e in sample_ids:
                W = st.tensor(ename(e, tag))
                sc = W.abs().amax(-1, keepdim=True).clamp(min=1e-8)
                V = (W / sc).reshape(-1, VEC_DIM)
                g = torch.Generator().manual_seed(1234 + e)
                chunks.append(V[torch.randperm(V.shape[0], generator=g)[:per]])
                del W, V
            X = torch.cat(chunks); del chunks
            books[kind] = train_codebooks(X, stages, dev, sample=TRAIN_VECTORS,
                                          entries=entries)
            del X
            for si, C in enumerate(books[kind]):
                cid = cb_base + ki * stages + si
                cf.write(struct.pack("<IHBBII", MAGIC_CODEBOOK, cid & 0xFFFF,
                                     FMT_VQ3R, VEC_DIM, entries, 0))
                cf.write(raw_bytes(C.cpu().half()))
    os.replace(cbf + ".tmp", cbf)

    with open(bank + ".tmp", "wb") as f:
        for e in range(n_exp):
            payloads, scales = [], []
            for kind, tag in kinds:
                W = st.tensor(ename(e, tag))
                idx, sc = quantize_vq(W, books[kind], dev, entries=entries)
                payloads.append(idx); scales.append(sc)
                del W
            write_expert_record(f, L, e, cb_base, payloads, scales, shapes,
                                packed=(index_bits == 6))
    os.replace(bank + ".tmp", bank)
    return (L, os.path.getsize(bank), cb_base, f"{_t.time()-t0:.0f}s")



def build_trunk(args, sr, st, existing, manifest_path):
    """Write trunk.bin.tmp and return its index, or None if the run must stop.

    A function rather than a stretch of main() because --reclaim has to run
    it *before* the expert layers instead of after: the trunk pass is the
    consumer of every tensor that is not an expert, so until it has run
    almost no shard is fully spent and there is nothing to delete.
    """
    trunk_path = os.path.join(args.out, "trunk.bin")
    trunk_tmp = trunk_path + ".tmp"
    tindex = []
    # Which MoE naming this checkpoint uses. Read from the names themselves rather than
    # probed: build_trunk takes no prefix to build a probe with, and the index already
    # says. Only affects what the router and shared experts are WRITTEN as — see
    # trunk_rename.
    _trunk_seg = ("mlp" if any(".mlp.experts." in n for n in sr.names())
                  else "block_sparse_moe")
    if args.skip_trunk:
        # "The trunk is unchanged between runs" — so carry its published
        # index forward and still publish. Returning here instead left the
        # banks this run had just converted unreferenced by any manifest,
        # and the codebooks.bin the merge had already replaced described by
        # an old one: a container that reports itself as fine and whose
        # layer bases point into records that are no longer there.
        tindex = (existing or {}).get("trunk")
        if not isinstance(tindex, list) or not tindex:
            print(f"--skip-trunk keeps the trunk {manifest_path} describes, "
                  "and it describes none — convert once without it",
                  file=sys.stderr)
            return None
        if not os.path.exists(trunk_path):
            print(f"--skip-trunk needs {trunk_path}, which is not there",
                  file=sys.stderr)
            return None
        print(f"skipping trunk: keeping the published {len(tindex)} tensors")
    else:
        # Keep the currently published trunk intact for the entire
        # conversion. A K3 trunk takes hours to rebuild; opening the final
        # path with "wb" destroyed a working container at the start of that
        # interval.
        with open(trunk_tmp, "wb") as tf:
            for name in sorted(sr.names()):
                if ".experts." in name or name.endswith(("_packed", "_scale")):
                    continue
                if not st.have(name):
                    continue                  # shard not downloaded yet
                t = st.tensor(name)
                name = trunk_rename(name, _trunk_seg)
                off = tf.tell()
                if t.dim() == 1 or t.numel() < 1 << 16:
                    tf.write(raw_bytes(t.float()))
                    tindex.append({"name": name, "fmt": FMT_F32, "off": off,
                                   "shape": list(t.shape),
                                   "bytes": tf.tell() - off})
                else:
                    # 4 bits for the bulk; the embedding table and the output
                    # head keep 8, they are small and sit at both ends of the
                    # network where error is least forgiving
                    big = not (name.endswith("embed_tokens.weight")
                               or name.endswith("lm_head.weight"))
                    bits = 8 if (args.trunk8 or not big) else args.trunk_bits
                    if bits == 3:
                        q, sc, shape = quantize_q3g(t); fmt = FMT_Q3G
                    elif bits == 4:
                        q, sc, shape = quantize_q4g(t); fmt = FMT_Q4G
                    else:
                        q, sc, shape = quantize_q8g(t); fmt = FMT_Q8G
                    tf.write(raw_bytes(q))
                    sc_off = tf.tell()
                    tf.write(raw_bytes(sc))
                    tindex.append({"name": name, "fmt": fmt,
                                   "off": off, "shape": shape, "group": 128,
                                   "scale_off": sc_off,
                                   "bytes": tf.tell() - off})
            tf.flush()
            os.fsync(tf.fileno())
        print(f"trunk: {os.path.getsize(trunk_tmp)/2**20:.0f} MB, "
              f"{len(tindex)} tensors")
    return tindex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="HF checkpoint directory, as published")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="", help="comma list; default = all MoE layers")
    ap.add_argument("--stages", type=int, choices=(2, 3, 4, 6), default=3,
                    help="3 = VQ3R, 2 = VQ2R; 4 and 6 exist to pair with "
                         "--entries below the byte")
    # bits/weight is stages * log2(entries) / 8 per 8-dim vector, so 3x256,
    # 4x64 and 6x16 all encode 24 bits per vector. They differ in whether the
    # per-stage table fits a SIMD register: 256 entries does not, which is
    # why the gather is scalar. The engine reads `entries` from the manifest
    # and indexes with a whole byte, so a container written below 256 is
    # correct but not yet smaller — the 6-bit/4-bit index packing is a
    # separate change. Written for that experiment; 256 remains the default.
    ap.add_argument("--entries", type=int, default=CB_ENTRIES,
                    choices=(16, 32, 64, 128, 256),
                    help="codebook entries per stage (default 256)")
    ap.add_argument("--index-bits", type=int, default=8, choices=(6, 8),
                    help="6 packs four 6-bit indices into three bytes "
                         "(WQ_VQ4P, 3.00 b/w); needs --stages 4 --entries 64")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--experts", type=int, default=0, help="limit experts (debug)")
    ap.add_argument("--trunk-bits", type=int, default=4, choices=(3, 4, 8),
                    help="bit width for the bulk of the trunk; it is the RAM "
                         "floor, so this is the main lever on cache size")
    ap.add_argument("--trunk8", action="store_true",
                    help="keep the whole trunk at 8 bits (needs the RAM)")
    ap.add_argument("--skip-trunk", action="store_true",
                    help="experts only; keeps the trunk and the trunk index "
                         "the existing manifest describes, and republishes "
                         "the manifest with the layers this run converted")
    ap.add_argument("--jobs", type=int, default=3,
                    help="layers converted in parallel; measured sweet spot "
                         "is 3 — beyond that the native encoder is already "
                         "using every core")
    ap.add_argument("--cb-sample", type=int, default=12,
                    help="experts sampled per layer to fit the codebooks")
    ap.add_argument("--reclaim", choices=("off", "dry", "on"), default="off",
                    help="delete each source shard as soon as no part of "
                         "this conversion will read it again, so peak "
                         "staging is the container plus the shards still "
                         "owed rather than the container plus the whole "
                         "checkpoint; 'dry' only says what it would delete")
    args = ap.parse_args()
    if args.jobs < 1:
        ap.error("--jobs must be at least 1")
    if args.cb_sample < 1:
        ap.error("--cb-sample must be at least 1")
    if args.reclaim != "off" and args.experts:
        # A layer converted with --experts N never read the tensors of the
        # other experts, so its shards are not spent and deleting them would
        # take weights this container does not contain.
        ap.error("--reclaim and --experts are exclusive: a layer converted "
                 "with an expert subset has not consumed its shards")
    # The packing is 4x6 into 3 bytes and nothing else; a 6-bit field cannot
    # hold an index into more than 64 entries, and the engine's unpack is
    # written for exactly four stages.
    if args.index_bits == 6 and (args.stages != 4 or args.entries != 64):
        ap.error("--index-bits 6 requires --stages 4 --entries 64")

    # torch sizes its intra-op pool from os.cpu_count() by default, so N
    # worker processes would otherwise spawn N*cpus threads and thrash each
    # other off the machine. Cap each worker at its fair share of the cores;
    # the env is inherited by the spawn'd workers before they import torch,
    # which is the only point the setting is read. setdefault leaves an
    # explicit OMP_NUM_THREADS the caller's to control.
    per = max(1, (os.cpu_count() or 1) // args.jobs)
    os.environ.setdefault("OMP_NUM_THREADS", str(per))
    os.environ.setdefault("MKL_NUM_THREADS", str(per))

    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(os.path.join(args.src, "config.json")))
    prefix = ""
    if "text_config" in cfg:                     # K3 nests the text model
        cfg = {**cfg["text_config"], "_outer": {k: v for k, v in cfg.items()
                                                if k != "text_config"}}
        prefix = "language_model."
    st = ST(args.src)
    sr = ShardReader(args.src)
    dev = torch.device(args.device)

    # ---- staging reclaim: refuse before anything is deleted, not during --
    debt = None
    if args.reclaim != "off":
        src_real, out_real = os.path.realpath(args.src), os.path.realpath(args.out)
        if (src_real == out_real
                or out_real.startswith(src_real + os.sep)
                or src_real.startswith(out_real + os.sep)):
            print("--reclaim refuses to delete shards that live inside the "
                  "container's own directory, or a container inside the "
                  "checkpoint: keep the two apart (docs/GATES.md, Gate H)",
                  file=sys.stderr)
            return 1
        # A shard that is absent is either one an earlier reclaim consumed —
        # fine, and the ledger says so — or one the download never finished.
        # The second is what must not proceed: build_trunk silently skips a
        # tensor whose shard is not there, so a partial checkpoint yields a
        # short trunk, and reclaiming would then delete the shards that fed
        # the parts that *were* right.
        #
        # Absence is asked of the filesystem, not of ST.have(): that reads
        # fetch_weights.sh's .download-state, which still lists a shard this
        # tool deleted and would report it present. have() is asked the other
        # question — whether a shard that IS there finished downloading.
        spent = ShardDebt.ledger(args.src)
        probe = {}                          # shard -> one tensor inside it
        for name, fn in st.wm.items():
            probe.setdefault(fn, name)
        gone, absent, unverified = set(), [], []
        for fn, one in sorted(probe.items()):
            if not os.path.exists(os.path.join(args.src, fn)):
                gone.add(fn)
                if fn not in spent:
                    absent.append(fn)
            elif not st.have(one):
                unverified.append(fn)
        if absent or unverified:
            why = (f"{len(absent)} missing (first {absent[0]})" if absent
                   else f"{len(unverified)} unverified (first {unverified[0]})")
            print(f"--reclaim needs every shard either on disk and verified "
                  f"or already reclaimed: {why}. Finish the download first.",
                  file=sys.stderr)
            return 1
        # The trunk cannot be rebuilt from a checkpoint whose non-expert
        # shards are already gone. Say which flag makes that a run rather
        # than a truncated container.
        if gone and not args.skip_trunk and any(
                ShardDebt.consumer(name) == ShardDebt.TRUNK
                and fn in gone for name, fn in st.wm.items()):
            print(f"--reclaim already consumed the shards the trunk is built "
                  f"from; rerun with --skip-trunk to keep the trunk "
                  f"{os.path.join(args.out, 'manifest.json')} publishes",
                  file=sys.stderr)
            return 1
        debt = ShardDebt(st.wm, args.src)
        n_shards, held = debt.still_owed()
        print(f"reclaim={args.reclaim}: {n_shards} shards, "
              f"{human(held)} of staging to give back"
              + ("" if args.reclaim == "on" else " (nothing is deleted)"))
        if args.reclaim == "on":
            print("  a reclaimed shard is gone: tools/verify_container.py "
                  "cannot check this container against its source afterwards, "
                  "so prove the recipe on a smaller model first.")
    print(f"device={dev}  stages={args.stages}  entries={args.entries}  "
          f"index_bits={args.index_bits} "
          f"({args.stages * args.index_bits / VEC_DIM:.2f} b/w)")

    n_layers = cfg["num_hidden_layers"]
    n_exp = cfg.get("num_experts") or cfg.get("n_routed_experts")
    print(f"prefix {prefix!r}  layers {n_layers}  experts {n_exp}")
    first_dense = cfg.get("first_k_dense_replace", 0)
    if args.experts:
        n_exp = min(n_exp, args.experts)
    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else list(range(first_dense, n_layers)))

    # ---- expert banks, one layer at a time ------------------------------
    manifest_path = os.path.join(args.out, "manifest.json")
    existing = None
    if os.path.exists(manifest_path):
        try:
            existing = json.load(open(manifest_path))
        except (OSError, ValueError):
            existing = None

    cb_record_bytes = struct.calcsize("<IHBBII") + args.entries * VEC_DIM * 2
    merged = os.path.join(args.out, "codebooks.bin")
    old_books = 0
    old_quant = existing.get("expert_quant", {}) if existing else {}
    compatible = bool(
        existing and os.path.exists(merged) and
        old_quant.get("stages") == args.stages and
        old_quant.get("vec_dim") == VEC_DIM and
        old_quant.get("entries") == args.entries and
        old_quant.get("index_bits", 8) == args.index_bits and
        os.path.getsize(merged) % cb_record_bytes == 0)
    if compatible:
        old_books = os.path.getsize(merged) // cb_record_bytes
        manifest_layers = dict(existing.get("layers", {}))
    else:
        manifest_layers = {}


    # Layout is a property of the checkpoint, so probe once on the first MoE layer
    # rather than per layer: a per-layer probe would mask a checkpoint that mixes them,
    # which would be a corrupt source rather than a shape to support.
    _lname, _seg, _kinds = (None, None, None)
    for _L in range(n_layers):
        _lname, _seg, _kinds = moe_layout(st, prefix, _L)
        if _lname:
            break
    if _lname is None:
        _seg, _kinds = MOE_LAYOUTS[0][1], KINDS

    def ename(L, e, tag):
        return f"{prefix}model.layers.{L}.{_seg}.experts.{e}.{tag}.weight"

    n_cb_per_layer = 3 * args.stages
    next_base = old_books
    jobs = []
    for L in layers:
        meta = manifest_layers.get(str(L), {})
        bank = os.path.join(args.out, f"experts-L{L}.bin")
        part = os.path.join(args.out, f"codebooks-L{L}.bin")
        base = meta.get("codebook_base", -1)
        cached_ok = bool(
            compatible and isinstance(base, int) and base >= 0 and
            base + n_cb_per_layer <= old_books and
            meta.get("experts") == n_exp and os.path.exists(bank) and
            meta.get("bytes") == os.path.getsize(bank))
        if not cached_ok:
            # A run interrupted before it published anything leaves no
            # manifest and no codebooks.bin, so there is nothing to read a
            # base from — and re-deriving one positionally is the bug this
            # scheme exists to avoid. But a finished bank names its own
            # base in every record it holds, and its unmerged codebook part
            # is still on disk beside it. Together those are a complete,
            # self-describing layer, so honour them.
            recovered = bank_codebook_base(bank)
            # Bounded by the whole model, not by this invocation's --layers:
            # the base in the bank reflects the run that wrote it, which may
            # have been converting far more layers than this one is.
            if (old_books <= recovered <=
                    old_books + n_layers * n_cb_per_layer and
                    os.path.exists(part) and
                    os.path.getsize(part) == n_cb_per_layer * cb_record_bytes
                    and os.path.getsize(bank) > 0):
                base = recovered
                cached_ok = True
        _t0 = _kinds[0][1]
        source_ok = (st.have(ename(L, 0, _t0)) or
                     st.have(ename(L, 0, _t0) + "_packed"))
        if not cached_ok:
            base = next_base
            if source_ok:
                next_base += n_cb_per_layer
        elif base + n_cb_per_layer > next_base:
            next_base = base + n_cb_per_layer
        jobs.append((L, args.src, args.out, prefix, n_exp, args.stages,
                     args.entries, args.index_bits, str(dev), args.cb_sample,
                     base, cached_ok))

    tindex = None
    if debt is not None:
        # The trunk pass consumes every tensor that is not an expert, so
        # until it has run almost no shard is fully spent. Run it first. It
        # writes trunk.bin.tmp either way and the published trunk.bin is
        # still only replaced together with the manifest, so nothing about
        # what this run can survive changes — only the order.
        tindex = build_trunk(args, sr, st, existing, manifest_path)
        if tindex is None:
            return 1
        reclaim(debt, args.reclaim, ShardDebt.TRUNK, "trunk")

        # A resumed conversion is holding the shards of every layer an
        # earlier run already finished. Give those back too, on the same
        # evidence the engine would use: the manifest names the bank, and
        # the bank is a whole bank.
        for L in sorted(set(range(first_dense, n_layers)) - set(layers)):
            meta = manifest_layers.get(str(L), {})
            bank = os.path.join(args.out, f"experts-L{L}.bin")
            if (meta.get("experts") == n_exp and os.path.exists(bank)
                    and meta.get("bytes") == os.path.getsize(bank)
                    and bank_is_sound(bank, L, n_exp)):
                reclaim(debt, args.reclaim, ("layer", L), f"layer {L} (earlier run)")

    def reclaim_layer(L, size):
        """Give back layer L's shards, once its bank proves it is finished."""
        if debt is None or not size:
            return
        bank = os.path.join(args.out, f"experts-L{L}.bin")
        if not bank_is_sound(bank, L, n_exp):
            print(f"  reclaim: keeping layer {L}'s shards — experts-L{L}.bin "
                  f"is not a complete bank of {n_exp} records", file=sys.stderr)
            return
        reclaim(debt, args.reclaim, ("layer", L), f"layer {L}")

    if args.jobs > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")     # torch/MPS is not fork-safe
        print(f"converting {len(jobs)} layers with {args.jobs} processes", flush=True)
        with ctx.Pool(args.jobs) as pool:
            results = []
            for res in pool.imap_unordered(convert_layer, jobs):
                results.append(res)
                Lr, sz, base, how = res
                print(f"  layer {Lr}: {sz/2**20:.0f} MB, cb base {base} [{how}] "
                      f"({len(results)}/{len(jobs)})", flush=True)
                reclaim_layer(Lr, sz)
    else:
        results = []
        for j in jobs:
            res = convert_layer(j)
            results.append(res)
            print(f"  layer {res[0]}: {res[1]/2**20:.0f} MB [{res[3]}]", flush=True)
            reclaim_layer(res[0], res[1])

    for L, sz, base, how in sorted(results):
        if sz:
            manifest_layers[str(L)] = {"file": f"experts-L{L}.bin",
                                       "experts": n_exp, "bytes": sz,
                                       "codebook_base": base}
        else:
            manifest_layers.pop(str(L), None)

    # Append new records after the already-published books. A cached base
    # comes from the old manifest or from the bank itself; a new one starts
    # at the old record count. Either way the base is fixed before we get
    # here, so this writes each part *at* its base and never renumbers.
    parts = [(res[2], os.path.join(args.out, f"codebooks-L{res[0]}.bin"))
             for res in results]
    if any(os.path.exists(p) for _, p in parts):
        with open(merged + ".tmp", "wb") as cb_out:
            if compatible:
                with open(merged, "rb") as old:
                    shutil.copyfileobj(old, cb_out)
            for base, part in sorted(parts):
                if os.path.exists(part):
                    if os.path.getsize(part) != n_cb_per_layer * cb_record_bytes:
                        raise RuntimeError(f"malformed codebook part: {part}")
                    expected = cb_out.tell() // cb_record_bytes
                    if base < expected:
                        raise RuntimeError(
                            f"codebook base {base} overlaps the {expected} "
                            f"records already written")
                    if base > expected:
                        # A resume can recover a bank whose base sits past
                        # the end of what is being written, because an
                        # earlier run finished a layer this invocation is
                        # not redoing. No record names the ids in between,
                        # so pad them: the bases inside the banks are the
                        # engine's truth and cannot be moved.
                        cb_out.write(b"\0" * ((base - expected) *
                                              cb_record_bytes))
                    with open(part, "rb") as pf:
                        shutil.copyfileobj(pf, cb_out)
                    os.remove(part)
            cb_out.flush()
            os.fsync(cb_out.fileno())
        os.replace(merged + ".tmp", merged)
    else:
        print("codebooks: all layers cached, keeping existing codebooks.bin")


    # ---- tokenizer: copy it in so the container is self-contained -------
    for name in ("tiktoken.model", "tokenizer.model"):
        src_tok = os.path.join(args.src, name)
        if os.path.exists(src_tok):
            atomic_copyfile(src_tok, os.path.join(args.out, "tokenizer.model"))
            print(f"tokenizer: copied {name}")
            break

    # ---- special tokens --------------------------------------------------
    # tiktoken's rank file holds only ordinary merges; the markup tokens live
    # in tokenizer_config.json. Without them the engine splits <|open|> into
    # six ordinary tokens, which silently destroys any chat template and the
    # media markers an image would be wrapped in.
    p_cfg = os.path.join(args.src, "tokenizer_config.json")
    if os.path.exists(p_cfg):
        dec = json.load(open(p_cfg)).get("added_tokens_decoder", {})
        specials = sorted(((int(i), v["content"]) for i, v in dec.items()),
                          key=lambda x: x[0])
        if specials:
            atomic_json(os.path.join(args.out, "specials.json"),
                        [{"id": i, "text": t} for i, t in specials])
            print(f"special tokens: {len(specials)} written")

    # ---- chat template ---------------------------------------------------
    # HF keeps it either as a field in tokenizer_config.json or, more
    # recently, as its own .jinja file. Copy whichever exists so `waste chat`
    # can address an instruct model in the format it was trained on instead
    # of as raw continuation. Not every release ships one — K3 describes its
    # template in the technical report without distributing it — so a
    # container without a template is normal and the CLI falls back.
    tpl = None
    for name in ("chat_template.jinja", "chat_template.json"):
        p_tpl = os.path.join(args.src, name)
        if os.path.exists(p_tpl):
            tpl = io.open(p_tpl, encoding="utf-8").read()
            break
    if tpl is None:
        p_cfg = os.path.join(args.src, "tokenizer_config.json")
        if os.path.exists(p_cfg):
            tpl = json.load(open(p_cfg)).get("chat_template")
    if tpl:
        atomic_text(os.path.join(args.out, "chat_template.jinja"), tpl)
        print(f"chat template: copied ({len(tpl)} bytes)")

    # ---- chat.json -------------------------------------------------------
    # The declarative format the CLI reads. Neither Kimi release ships a
    # template, so there is nothing to convert from — but for the models we
    # have transcribed from the reference encoder, examples/ holds one, and
    # a container that has to be finished by hand is a container that will
    # be used unfinished. Copied from examples/ rather than embedded here,
    # so there is one copy of each template and not two that drift.
    #
    # Only for an architecture we actually have: for anything else the CLI
    # keeps saying so and falling back, which is better than a guessed
    # format that produces plausible wrong answers. And never over an
    # existing file — a hand-edited chat.json outranks the shipped one.
    _tmpl_for = {"kimi-k3": "chat-k3.json"}
    _hf0 = ((cfg.get("_outer", {}).get("architectures")
             or cfg.get("architectures") or [""]))[0]
    _arch0 = ("kimi-k3" if "KimiK3" in _hf0 else
              "kimi-linear" if "KimiLinear" in _hf0 else "")
    _dst = os.path.join(args.out, "chat.json")
    _name = _tmpl_for.get(_arch0)
    if _name and not os.path.exists(_dst):
        _src_tmpl = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "examples", _name)
        if os.path.exists(_src_tmpl):
            atomic_copyfile(_src_tmpl, _dst)
            print(f"chat.json: {_name} — `waste chat` will use it")
        else:
            print(f"chat.json: examples/{_name} not found; the CLI will "
                  f"fall back to raw continuation")
    elif os.path.exists(_dst):
        print("chat.json: already present, left alone")
    elif _arch0:
        print(f"chat.json: none known for {_arch0}; the CLI will fall back "
              f"to raw continuation (see examples/README.md)")

    # ---- vision config ---------------------------------------------------
    # The tower's shape lives in the release's nested vision_config, and the
    # engine reads it from vision.json — only from there. Without this file
    # waste_image_add returns WASTE_E_UNSUPPORTED and a multimodal container
    # silently has no images, which is exactly what happened before it was
    # written: the file had to be produced by hand after every conversion.
    #
    # Two keys are ours rather than the release's:
    #   vt_rms_eps    the tower's RMSNorms are nn.RMSNorm(dim) with no eps,
    #                 so PyTorch uses finfo(float32).eps. Written explicitly
    #                 so the oracle in tools/vision_ref.py reads the same
    #                 number the engine compiles in.
    #   max_patches   our patch budget, i.e. how much of the context an
    #                 image costs. The release's own limit is far higher
    #                 (in_patch_limit 65536, 512 patches on a side); an
    #                 image is priced as text of the same length, so this
    #                 is a deliberate cap and halving it halves the prompt.
    #
    # image_mean/std are the release's, read from preprocessor_config.json,
    # where `media_proc_cfg` is what kimi_k3_vision_processing.py actually
    # applies. They were hardcoded to the CLIP constants here for exactly
    # one day, on the belief that K3 shipped no preprocessor config — it
    # does, and K3 normalizes to [-1, 1] with mean = std = 0.5, which is
    # not what CLIP does. Guess nothing that the release states.
    vc = cfg.get("_outer", {}).get("vision_config")
    if vc:
        vj = {k: v for k, v in vc.items() if not k.startswith("_")}
        vj["vt_rms_eps"] = 1.1920928955078125e-07
        mp = cfg.get("_outer", {}).get("media_placeholder_token_id")
        if mp is not None:
            vj["media_placeholder_token_id"] = mp
        vj["max_patches"] = 1024

        p_pre = os.path.join(args.src, "preprocessor_config.json")
        if os.path.exists(p_pre):
            mpc = json.load(open(p_pre)).get("media_proc_cfg", {})
            for k in ("image_mean", "image_std"):
                if mpc.get(k) is not None:
                    vj[k] = mpc[k]
            for k in ("patch_size", "in_patch_limit", "patch_limit_on_one_side"):
                if mpc.get(k) is not None:
                    vj[k] = mpc[k]
            print(f"vision: normalization from preprocessor_config.json, "
                  f"mean {vj.get('image_mean')}")
        else:
            # No config to read: say so loudly rather than invent constants.
            # A tower fed the wrong normalization still matches its oracle,
            # because the oracle reads this same file — the error hides.
            vj["image_mean"] = [0.5, 0.5, 0.5]
            vj["image_std"] = [0.5, 0.5, 0.5]
            print("vision: WARNING no preprocessor_config.json in --src; "
                  "assuming mean=std=0.5. Check it against the release.")

        atomic_json(os.path.join(args.out, "vision.json"), vj)
        print(f"vision: {vj.get('vt_num_hidden_layers', '?')}-layer tower, "
              f"patch {vj.get('patch_size', '?')}, "
              f"max_patches {vj['max_patches']}")

    # ---- trunk ----------------------------------------------------------
    # --reclaim has already run this, before the experts, so that the shards
    # holding non-expert tensors become deletable at all.
    if tindex is None:
        tindex = build_trunk(args, sr, st, existing, manifest_path)
        if tindex is None:
            return 1
    trunk_path = os.path.join(args.out, "trunk.bin")
    trunk_tmp = trunk_path + ".tmp"

    # `arch` is descriptive only — the engine derives its own from
    # config._outer.architectures, because the *inner* architectures says
    # KimiLinearForCausalLM on both models. It used to be written from
    # model_type, which for K3 is the text model's and reads "kimi_linear"
    # in a K3 container. Write what `waste info` reports, so a container
    # inspected by hand and a container opened by the engine agree.
    _hf = ((cfg.get("_outer", {}).get("architectures")
            or cfg.get("architectures") or [""]))[0]
    arch = ("kimi-k3" if "KimiK3" in _hf else
            "kimi-linear" if "KimiLinear" in _hf else
            _hf or cfg.get("model_type", "unknown"))

    manifest = {
        "format_version": 0,
        "arch": arch,
        "tensor_prefix": prefix,
        "config": normalise_cfg(cfg),
        # The record's fmt byte is FMT_VQ3R for every stage count but 2 — the
        # engine takes the stage and entry counts from here, not from the
        # byte, and only refuses a fmt that is neither VQ3R nor VQ2R.
        # bits_per_weight is `stages` because the index is still a whole byte
        # per stage; --entries below 256 buys quality headroom, not bytes,
        # until the packed index lands.
        "expert_quant": {"fmt": ("VQ4P" if args.index_bits == 6 else
                                 "VQ2R" if args.stages == 2 else "VQ3R"),
                         "stages": args.stages, "vec_dim": VEC_DIM,
                         "entries": args.entries, "index_block": IDX_BLOCK,
                         "index_bits": args.index_bits,
                         "bits_per_weight":
                             args.stages * args.index_bits / VEC_DIM},
        "layers": manifest_layers,
        "trunk": tindex,
    }
    # A manifest that lists fewer expert layers than the one it replaces
    # publishes a container the engine will refuse to open, and the banks it
    # drops are still on disk taking up room. Never intended; say so.
    dropped = sorted(set((existing or {}).get("layers", {})) -
                     set(manifest_layers), key=int)
    if dropped:
        print(f"WARNING dropping {len(dropped)} expert layer(s) the previous "
              f"manifest listed: {', '.join(dropped)}", file=sys.stderr)

    manifest_tmp = manifest_path + ".tmp"
    with open(manifest_tmp, "w") as f:
        json.dump(manifest, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    # The manifest is the commit record and is always published last.  An
    # exception before these replaces leaves the old container untouched.
    if not args.skip_trunk:
        os.replace(trunk_tmp, trunk_path)
    os.replace(manifest_tmp, manifest_path)
    print(f"wrote {args.out}/manifest.json")
    if debt is not None:
        n_shards, held = debt.still_owed()
        verb = "reclaimed" if args.reclaim == "on" else "reclaimable"
        print(f"reclaim: {human(debt.freed)} {verb} from "
              f"{len(debt.released)} shard(s); {n_shards} shard(s) "
              f"({human(held)}) still held")
        if n_shards:
            # Almost always the layers this invocation did not convert. Say
            # which, because "still held" with no reason reads like a bug.
            waiting = sorted({who for s in debt.owed for who in debt.owed[s]})
            print(f"  waiting on: "
                  f"{', '.join(ShardDebt.name(w) for w in waiting[:8])}"
                  f"{' ...' if len(waiting) > 8 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
