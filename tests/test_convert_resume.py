#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""test_convert_resume.py — tools/convert.py must resume, and must never
renumber a bank that already exists.

Written after a resume fix removed the one thing resume depended on: the
cache check needed a manifest, and the manifest is published last, so a
first conversion interrupted at layer 40 of 92 re-did all 40. Six
scenarios, because the interesting ones are the partial states.

No torch, no source weights, no minutes of quantization: the expensive
half is stubbed and the decision half is convert.py's own.

Only the quantizer is stubbed: convert_layer is replaced by a fake that
writes the bank and codebook part a real one would, using the base the
parent handed it. Everything that decides *which* base that is, and how the
parts are merged, is convert.py's own code.

  python3 tests/test_convert_resume.py
"""
import json
import os
import shutil
import struct
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_LAYERS, N_EXPERTS, FIRST_DENSE, STAGES = 6, 4, 1, 3
MOE_LAYERS = list(range(FIRST_DENSE, N_LAYERS))


TRUNK_TENSOR = "model.norm.weight"      # one small f32 trunk tensor is enough

# A shard layout with the shape that makes --reclaim interesting: one shard
# nobody but the trunk pass reads, one that only layer 3 reads, and two that
# straddle a layer boundary and are therefore owed to two consumers at once.
SHARDS = {
    "shard-trunk.safetensors": [TRUNK_TENSOR],
    "shard-a.safetensors": [(1, 2)],
    "shard-b.safetensors": [(3,)],
    "shard-c.safetensors": [(4, 5)],
}


def expert_names(layer):
    return [f"model.layers.{layer}.block_sparse_moe.experts.{e}.{tag}.weight"
            for e in range(N_EXPERTS) for tag in ("w1", "w2", "w3")]


WEIGHT_MAP = {}
for _shard, _holds in SHARDS.items():
    for _h in _holds:
        for _n in ([_h] if isinstance(_h, str)
                   else [n for L in _h for n in expert_names(L)]):
            WEIGHT_MAP[_n] = _shard


class FakeTensor:
    """Just enough tensor for the trunk loop's f32 branch."""
    shape = (8,)

    def dim(self): return 1
    def numel(self): return 8
    def float(self): return self


def install_stubs():
    torch = types.ModuleType("torch")
    torch.device = lambda s: s
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False))
    sys.modules["torch"] = torch
    mx = types.ModuleType("mxfp4")

    class ST:
        def __init__(self, src):
            self.dir = src
            self.wm = dict(WEIGHT_MAP)

        def have(self, name):
            # mxfp4.ST's own rule, and the trap in it: when
            # fetch_weights.sh's .download-state is there it is believed
            # over the filesystem, so a shard --reclaim deleted still reads
            # as present. convert.py must not decide absence from this.
            shard = self.wm.get(name)
            if not shard:
                return False
            state = os.path.join(self.dir, ".download-state")
            if os.path.exists(state):
                return shard in open(state).read().split()
            return os.path.exists(os.path.join(self.dir, shard))

        def tensor(self, name): return FakeTensor()
        def names(self): return [TRUNK_TENSOR]
    mx.ST = ST
    # convert.py imports this alongside ST for fp8 block-scaled checkpoints.
    # Resume never reaches a dequant, but the stub has to satisfy the import
    # or every check in this file fails at module load.
    mx.unblock_scale = lambda q, scale, block: q
    sys.modules["mxfp4"] = mx
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import convert
    convert.ShardReader = lambda src: types.SimpleNamespace(
        names=lambda: [TRUNK_TENSOR])
    # Serialization is not what this checks, and the real one needs torch.
    convert.raw_bytes = lambda t: b"\0" * (t.numel() * 4)
    return convert


CONV = install_stubs()
REC = struct.calcsize("<IHBBII") + CONV.CB_ENTRIES * CONV.VEC_DIM * 2
PER_LAYER = 3 * STAGES

converted = []           # layers the fake quantizer actually ran on


def fake_convert_layer(job):
    (L, src, out, prefix, n_exp, stages, entries, index_bits, device,
     cb_sample, cb_base, cached_ok) = job
    bank = os.path.join(out, f"experts-L{L}.bin")
    if cached_ok and os.path.exists(bank):
        return (L, os.path.getsize(bank), cb_base, "cached")
    converted.append(L)
    # A bank whose every record names cb_base, as write_expert_record does —
    # and with the whole 48-byte header, because bank_is_sound reads the
    # block count and the three payload offsets before --reclaim will let a
    # layer's shards go.
    with open(bank, "wb") as f:
        for eid in range(n_exp):
            f.write(struct.pack("<IHHBBHHHIIIIIIII", CONV.MAGIC_EXPERT, L, eid,
                                6, 0, cb_base, 0, 0,
                                1, 48, 1048, 2048, 3048, 0, 0, 0))
            f.write(b"\0" * (CONV.ALIGN - 48))
    with open(os.path.join(out, f"codebooks-L{L}.bin"), "wb") as f:
        for i in range(PER_LAYER):
            # Tag each record with its absolute id so the merge can be checked.
            f.write(struct.pack("<I", cb_base + i) + b"\0" * (REC - 4))
    return (L, os.path.getsize(bank), cb_base, "converted")


CONV.convert_layer = fake_convert_layer


def run(out, src, layers=None, jobs=1, skip_trunk=False, want_rc=0,
        reclaim=None):
    del converted[:]
    argv = ["convert.py", "--src", src, "--out", out, "--jobs", str(jobs),
            "--stages", str(STAGES)]
    if layers:
        argv += ["--layers", ",".join(str(x) for x in layers)]
    if skip_trunk:
        argv += ["--skip-trunk"]
    if reclaim:
        argv += ["--reclaim", reclaim]
    old = sys.argv
    sys.argv = argv
    try:
        rc = CONV.main()
    finally:
        sys.argv = old
    assert rc == want_rc, f"convert.main() returned {rc}, wanted {want_rc}"
    return list(converted)


def manifest(out):
    p = os.path.join(out, "manifest.json")
    return json.load(open(p)) if os.path.exists(p) else None


def books(out):
    p = os.path.join(out, "codebooks.bin")
    if not os.path.exists(p):
        return []
    raw = open(p, "rb").read()
    assert len(raw) % REC == 0, "codebooks.bin is not a whole number of records"
    return [struct.unpack_from("<I", raw, i * REC)[0]
            for i in range(len(raw) // REC)]


fails = 0


def ck(cond, what):
    global fails
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        fails += 1


def check_consistency(out, expect_layers):
    """Every bank's own base must index its own records in codebooks.bin."""
    b = books(out)
    for L in expect_layers:
        base = CONV.bank_codebook_base(os.path.join(out, f"experts-L{L}.bin"))
        if base < 0 or base + PER_LAYER > len(b):
            return False, f"L{L} base {base} outside {len(b)} records"
        got = b[base:base + PER_LAYER]
        if got != list(range(base, base + PER_LAYER)):
            return False, f"L{L} base {base} points at {got[:3]}..."
    return True, f"{len(expect_layers)} banks, {len(b)} records"


def make_src(path):
    """A checkpoint directory: the config, and one file per shard. --reclaim
    deletes those files, so every scenario that reclaims needs its own."""
    os.makedirs(path, exist_ok=True)
    json.dump({"num_hidden_layers": N_LAYERS, "num_experts": N_EXPERTS,
               "first_k_dense_replace": FIRST_DENSE,
               "num_experts_per_token": 2, "hidden_size": 8,
               "moe_intermediate_size": 8, "intermediate_size": 8,
               "vocab_size": 32},
              open(os.path.join(path, "config.json"), "w"))
    for i, shard in enumerate(sorted(SHARDS)):
        # Distinct sizes so the reported saving is checkable, not just > 0.
        with open(os.path.join(path, shard), "wb") as f:
            f.write(b"\0" * (1024 * (i + 1)))
    return path


def shards_on_disk(src):
    return sorted(s for s in SHARDS if os.path.exists(os.path.join(src, s)))


def main():
    tmp = tempfile.mkdtemp(prefix="resume-")
    src = make_src(os.path.join(tmp, "src"))
    try:
        print("a fresh conversion")
        out = os.path.join(tmp, "fresh.waste")
        did = run(out, src)
        ck(did == MOE_LAYERS, f"converts every MoE layer {did}")
        ck(books(out) == list(range(len(MOE_LAYERS) * PER_LAYER)),
           "codebooks.bin is dense and in order")
        ok, why = check_consistency(out, MOE_LAYERS)
        ck(ok, f"each bank indexes its own records ({why})")

        print("resume of a run interrupted before it published anything")
        out = os.path.join(tmp, "interrupted.waste")
        os.makedirs(out)
        # Emulate a crash: layers 1,2,3 finished (bank + unmerged part),
        # nothing merged, no manifest — exactly what is on disk mid-run.
        for i, L in enumerate(MOE_LAYERS[:3]):
            fake_convert_layer((L, src, out, "", N_EXPERTS, STAGES, CONV.CB_ENTRIES, 8, "cpu", 1,
                                i * PER_LAYER, False))
        ck(not os.path.exists(os.path.join(out, "manifest.json")),
           "no manifest exists, as after a real crash")
        did = run(out, src)
        ck(did == MOE_LAYERS[3:],
           f"re-converts only the unfinished layers {did}")
        ck(books(out) == list(range(len(MOE_LAYERS) * PER_LAYER)),
           "and the merged codebooks are still dense")
        ok, why = check_consistency(out, MOE_LAYERS)
        ck(ok, f"each bank indexes its own records ({why})")

        print("resume of a parallel run that finished layers out of order")
        out = os.path.join(tmp, "gap.waste")
        os.makedirs(out)
        for L in (1, 2, 3, 5):                 # layer 4 never finished
            fake_convert_layer((L, src, out, "", N_EXPERTS, STAGES, CONV.CB_ENTRIES, 8, "cpu", 1,
                                (L - FIRST_DENSE) * PER_LAYER, False))
        did = run(out, src)
        ck(did == [4], f"re-converts only the hole {did}")
        ok, why = check_consistency(out, MOE_LAYERS)
        ck(ok, f"each bank still indexes its own records ({why})")

        print("resume of a completed run")
        out = os.path.join(tmp, "done.waste")
        run(out, src)
        before = books(out)
        did = run(out, src)
        ck(did == [], f"converts nothing {did}")
        ck(books(out) == before, "and leaves codebooks.bin byte-identical")

        print("resume with a different --layers, the case that used to renumber")
        out = os.path.join(tmp, "subset.waste")
        os.makedirs(out)
        for L in (1, 2, 3):
            fake_convert_layer((L, src, out, "", N_EXPERTS, STAGES, CONV.CB_ENTRIES, 8, "cpu", 1,
                                (L - FIRST_DENSE) * PER_LAYER, False))
        did = run(out, src, layers=[3])
        ck(did == [], f"layer 3 is recognised as already done {did}")
        base3 = CONV.bank_codebook_base(os.path.join(out, "experts-L3.bin"))
        ck(base3 == 2 * PER_LAYER,
           f"and keeps the base its records name ({base3}), not 0")
        b = books(out)
        ck(b[base3:base3 + PER_LAYER] ==
           list(range(base3, base3 + PER_LAYER)),
           "its records sit at that base in codebooks.bin")

        print("--skip-trunk keeps the trunk and still publishes")
        out = os.path.join(tmp, "skiptrunk.waste")
        run(out, src, layers=[1])
        keep = manifest(out)["trunk"]
        trunk_before = open(os.path.join(out, "trunk.bin"), "rb").read()
        did = run(out, src, layers=[2], skip_trunk=True)
        ck(did == [2], f"converts the layer it was asked for {did}")
        m = manifest(out)
        ck("2" in m["layers"],
           "and the manifest it publishes lists that layer")
        ck(m["trunk"] == keep,
           "carrying the trunk index forward instead of emptying it")
        ck(open(os.path.join(out, "trunk.bin"), "rb").read() == trunk_before,
           "and leaving trunk.bin itself untouched")
        ok, why = check_consistency(out, [1, 2])
        ck(ok, f"both banks index their own records ({why})")
        ck(not os.path.exists(os.path.join(out, "trunk.bin.tmp")),
           "no staged trunk left behind")

        print("--skip-trunk with no published trunk to keep")
        out = os.path.join(tmp, "notrunk.waste")
        os.makedirs(out)
        did = run(out, src, layers=[1], skip_trunk=True, want_rc=1)
        ck(manifest(out) is None,
           "refuses rather than publish a manifest with an empty trunk")

        print("a bank whose codebook part was lost cannot be trusted")
        out = os.path.join(tmp, "lostpart.waste")
        os.makedirs(out)
        fake_convert_layer((1, src, out, "", N_EXPERTS, STAGES, CONV.CB_ENTRIES, 8, "cpu", 1,
                            0, False))
        os.remove(os.path.join(out, "codebooks-L1.bin"))
        did = run(out, src, layers=[1])
        ck(did == [1], f"it is re-converted rather than reused {did}")
        ok, why = check_consistency(out, [1])
        ck(ok, f"and comes out consistent ({why})")

        print("--reclaim dry names the spent shards and deletes none")
        rsrc = make_src(os.path.join(tmp, "src-dry"))
        out = os.path.join(tmp, "dry.waste")
        run(out, rsrc, reclaim="dry")
        ck(shards_on_disk(rsrc) == sorted(SHARDS),
           "every shard is still there")
        ck(not os.path.exists(os.path.join(rsrc, ".reclaimed")),
           "and nothing was written to the ledger")

        print("--reclaim on frees a shard as soon as its last reader is done")
        rsrc = make_src(os.path.join(tmp, "src-on"))
        out = os.path.join(tmp, "on.waste")
        # Layers 1..3 only: shard-c holds layers 4 and 5, so it must stay.
        run(out, rsrc, layers=[1, 2, 3], reclaim="on")
        ck(shards_on_disk(rsrc) == ["shard-c.safetensors"],
           f"the shards of the converted layers are gone "
           f"{shards_on_disk(rsrc)}")
        ck(CONV.ShardDebt.ledger(rsrc) ==
           {"shard-trunk.safetensors", "shard-a.safetensors",
            "shard-b.safetensors"},
           "and the ledger names exactly those")
        ok, why = check_consistency(out, [1, 2, 3])
        ck(ok, f"the container is unharmed ({why})")

        print("a shard owed to two layers survives until both have landed")
        # shard-c is layers 4 and 5. Converting only 4 must not free it.
        run(out, rsrc, layers=[4], reclaim="on", skip_trunk=True)
        ck(shards_on_disk(rsrc) == ["shard-c.safetensors"],
           "layer 4 alone does not free the shard layer 5 still needs")
        run(out, rsrc, layers=[5], reclaim="on", skip_trunk=True)
        ck(shards_on_disk(rsrc) == [],
           "layer 5 lands and the last shard goes")
        ok, why = check_consistency(out, MOE_LAYERS)
        ck(ok, f"every bank still indexes its own records ({why})")

        print("a resumed reclaim tells a spent shard from one never downloaded")
        rsrc = make_src(os.path.join(tmp, "src-resume"))
        out = os.path.join(tmp, "resume.waste")
        run(out, rsrc, layers=[1, 2], reclaim="on")
        # The trunk's shard is spent, so the trunk cannot be rebuilt: convert
        # refuses rather than publish a container with a truncated one.
        run(out, rsrc, layers=[3], reclaim="on", want_rc=1)
        ck(shards_on_disk(rsrc) == ["shard-b.safetensors",
                                    "shard-c.safetensors"],
           "the refusal deletes nothing")
        did = run(out, rsrc, layers=[3], reclaim="on", skip_trunk=True)
        ck(did == [3], f"--skip-trunk lets the resume through {did}")
        ck(shards_on_disk(rsrc) == ["shard-c.safetensors"],
           "and it reclaims what it consumed")
        # An unaccounted absence is still a refusal: this is the partial
        # download the ledger exists to distinguish.
        before = manifest(out)
        os.remove(os.path.join(rsrc, "shard-c.safetensors"))
        os.remove(os.path.join(rsrc, ".reclaimed"))
        run(out, rsrc, layers=[4], reclaim="on", skip_trunk=True, want_rc=1)
        ck(manifest(out) == before,
           "a shard absent and unrecorded refuses, publishing nothing")

        print("a download-state listing a reclaimed shard does not hide it")
        # fetch_weights.sh's .download-state is a record of what downloaded,
        # not of what is still there, and ST.have() believes it over the
        # filesystem. Deciding absence from have() made a reclaimed shard
        # read as present, which skipped both refusals above.
        rsrc = make_src(os.path.join(tmp, "src-state"))
        out = os.path.join(tmp, "state.waste")
        with open(os.path.join(rsrc, ".download-state"), "w") as f:
            f.write("\n".join(sorted(SHARDS)) + "\n")
        run(out, rsrc, layers=[1, 2], reclaim="on")
        ck(shards_on_disk(rsrc) == ["shard-b.safetensors",
                                    "shard-c.safetensors"],
           "the run still reclaims what it consumed")
        # The trunk's shard is gone; .download-state still names it. Believing
        # the state file, this run rebuilt the trunk from tensors it could not
        # read; seeing the filesystem, it refuses and touches nothing.
        kept = shards_on_disk(rsrc)
        run(out, rsrc, layers=[3], reclaim="on", want_rc=1)
        ck(shards_on_disk(rsrc) == kept,
           "the next run sees it as gone, refuses, and deletes nothing more")
        # An unverified shard — on disk, absent from .download-state — is
        # the other half of the same question, and still refuses.
        rsrc = make_src(os.path.join(tmp, "src-unver"))
        with open(os.path.join(rsrc, ".download-state"), "w") as f:
            f.write("shard-trunk.safetensors\n")
        run(os.path.join(tmp, "unver.waste"), rsrc, reclaim="on", want_rc=1)
        ck(shards_on_disk(rsrc) == sorted(SHARDS),
           "a half-downloaded checkpoint refuses and keeps every shard")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("RESUME FAILED" if fails else "RESUME OK")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
