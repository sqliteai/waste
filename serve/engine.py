# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
engine.py — the WASTE engine, through ctypes.

waste.h says the engine is a library first and the CLI is one of its
clients. This is the other client. Nothing here reimplements inference:
every function below is a call into libwaste, and the struct definitions
mirror waste.h field for field. When that header changes, this file changes
with it — tests/serve/test_engine.py checks the two agree by round-tripping
real calls rather than by trusting the layouts.

Two things this module owns beyond the binding itself:

**Segment tokenization.** serve/xtml.py produces segments tagged markup or
text; they go to waste_tokenize_markup and waste_tokenize respectively, one
segment at a time. Never concatenated first — see tokenize_segments.

**Serialization.** A waste_ctx is not thread-safe, and the header says so.
One lock, held for a whole generation. On a model that streams experts from
disk at a few tokens a second, a queue is not a limitation anyone will
notice next to the wait for the answer itself.
"""

from __future__ import annotations

import ctypes as C
import os
import platform
import threading
from pathlib import Path
from typing import Callable, Optional

from .xtml import (CLOSE_TOKEN, END_OF_MSG_TOKEN, OPEN_TOKEN, SEP_TOKEN,
                   Segment)

# ---- status codes (waste.h) ----------------------------------------------

WASTE_OK = 0
WASTE_E_IO = -1
WASTE_E_FORMAT = -2
WASTE_E_RAM_BUDGET = -3
WASTE_E_OOM = -4
WASTE_E_ARG = -5
WASTE_E_UNSUPPORTED = -6
WASTE_E_CANCELLED = -7
WASTE_E_BUSY = -8

# waste.h's waste_cache_policy. There is no third: a "pinned" policy was
# listed there and never implemented, so it selected LFRU like everything
# else that was not 1.
CACHE_LFRU, CACHE_LRU = 0, 1


class EngineError(RuntimeError):
    """A non-OK waste_status, with the engine's own name for it."""

    def __init__(self, what: str, status: int, detail: str = ""):
        self.status = status
        self.what = what
        name = _lib().waste_strerror(status).decode() if _LIB else str(status)
        super().__init__(f"{what}: {name}" + (f" ({detail})" if detail else ""))


class Cancelled(Exception):
    """The token callback asked to stop. Not an error: a client hung up."""


# ---- structs (waste.h) ---------------------------------------------------


class MemPlan(C.Structure):
    _fields_ = [("trunk_bytes", C.c_uint64),
                ("state_bytes", C.c_uint64),
                ("scratch_bytes", C.c_uint64),
                ("min_expert_cache", C.c_uint64),
                ("floor_bytes", C.c_uint64),
                ("recommended_bytes", C.c_uint64),
                ("vision_bytes", C.c_uint64)]


class Cfg(C.Structure):
    _fields_ = [("ram_budget_bytes", C.c_uint64),
                ("ctx_tokens", C.c_uint32),
                ("n_threads", C.c_int),
                ("cpu_list", C.c_char_p),
                ("cache_policy", C.c_int),
                ("use_direct_io", C.c_int),
                ("vision", C.c_int),
                ("verify_records", C.c_int),
                ("usage_path", C.c_char_p),
                ("allow_concurrent_open", C.c_int)]


class GenParams(C.Structure):
    _fields_ = [("temperature", C.c_float),
                ("top_p", C.c_float),
                ("top_k", C.c_int),
                ("seed", C.c_uint64),
                ("max_tokens", C.c_uint32),
                ("grammar", C.c_char_p),
                ("stop_tokens", C.POINTER(C.c_int32)),
                ("n_stop", C.c_size_t)]


class TokenInfo(C.Structure):
    _fields_ = [("token_index", C.c_uint32),
                ("token", C.c_int32),
                ("logprob", C.c_float),
                ("experts_hit", C.c_uint32),
                ("experts_missed", C.c_uint32),
                ("bytes_read", C.c_uint64),
                ("ms_total", C.c_double),
                ("ms_io", C.c_double)]


class ModelInfo(C.Structure):
    _fields_ = [("n_layers", C.c_uint32),
                ("n_experts", C.c_uint32),
                ("top_k", C.c_uint32),
                ("hidden", C.c_uint32),
                ("ctx_max", C.c_uint32),
                ("params_total", C.c_uint64),
                ("params_active", C.c_uint64),
                ("arch", C.c_char_p),
                ("quant_summary", C.c_char_p)]


class Stats(C.Structure):
    _fields_ = [("tokens_generated", C.c_uint64),
                ("experts_hit", C.c_uint64),
                ("experts_missed", C.c_uint64),
                ("bytes_read", C.c_uint64),
                ("sec_total", C.c_double),
                ("sec_io", C.c_double),
                ("direct_io", C.c_int)]


TOKEN_CB = C.CFUNCTYPE(C.c_int, C.POINTER(TokenInfo), C.c_char_p, C.c_void_p)

# ---- loading -------------------------------------------------------------

_LIB = None
_LIB_LOCK = threading.Lock()


def _soext() -> str:
    """What `make` called the shared object on this platform."""
    return {"Darwin": "dylib", "Windows": "dll"}.get(platform.system(), "so")


def library_candidates() -> list[Path]:
    """Where libwaste might be, most specific first.

    WASTE_LIB wins outright. Otherwise the repo's own build directory, which
    is where `make` puts it and where a developer running from a checkout
    expects it to be found without installing anything.
    """
    soext = _soext()
    out: list[Path] = []
    env = os.environ.get("WASTE_LIB")
    if env:
        out.append(Path(env))
    here = Path(__file__).resolve().parent
    for base in (here.parent, here):          # repo root, then serve/
        out.append(base / f"libwaste.{soext}")
    out.append(Path(f"libwaste.{soext}"))     # let the loader search
    return out


def _lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        tried = []
        for path in library_candidates():
            try:
                lib = C.CDLL(str(path))
            except OSError as e:
                tried.append(f"{path}: {e}")
                continue
            _bind(lib)
            _LIB = lib
            return _LIB
        raise EngineError(
            "libwaste not found", WASTE_E_IO,
            f"build it with `make libwaste.{_soext()}`, or point WASTE_LIB "
            "at it.\n  " + "\n  ".join(tried))


def _bind(lib) -> None:
    """Declare argument and return types.

    Not optional housekeeping: without restype on the char*-returning
    functions ctypes truncates the pointer to int on 64-bit, and without
    argtypes a Python int passed where a pointer belongs is silently
    marshalled as 32 bits.
    """
    lib.waste_version.restype = C.c_char_p
    lib.waste_version.argtypes = []
    lib.waste_version_number.restype = C.c_int
    lib.waste_version_number.argtypes = []
    lib.waste_build_info.restype = C.c_char_p
    lib.waste_build_info.argtypes = []
    lib.waste_strerror.restype = C.c_char_p
    lib.waste_strerror.argtypes = [C.c_int]

    lib.waste_plan_memory.restype = C.c_int
    lib.waste_plan_memory.argtypes = [C.c_char_p, C.c_uint32,
                                      C.POINTER(MemPlan)]
    lib.waste_cfg_init.restype = None
    lib.waste_cfg_init.argtypes = [C.POINTER(Cfg)]
    lib.waste_open.restype = C.c_int
    lib.waste_open.argtypes = [C.c_char_p, C.POINTER(Cfg),
                               C.POINTER(C.c_void_p)]
    lib.waste_close.restype = None
    lib.waste_close.argtypes = [C.c_void_p]
    lib.waste_memory_used.restype = C.c_int
    lib.waste_memory_used.argtypes = [C.c_void_p, C.POINTER(MemPlan)]

    for name in ("waste_tokenize", "waste_tokenize_markup"):
        fn = getattr(lib, name)
        fn.restype = C.c_int
        fn.argtypes = [C.c_void_p, C.c_char_p, C.c_int,
                       C.POINTER(C.c_int32), C.c_size_t, C.POINTER(C.c_size_t)]
    lib.waste_detokenize.restype = C.c_int
    lib.waste_detokenize.argtypes = [C.c_void_p, C.POINTER(C.c_int32),
                                     C.c_size_t, C.c_char_p, C.c_size_t,
                                     C.POINTER(C.c_size_t)]

    lib.waste_image_add.restype = C.c_int
    lib.waste_image_add.argtypes = [C.c_void_p, C.c_char_p,
                                    C.POINTER(C.c_size_t)]
    lib.waste_image_dimensions.restype = C.c_int
    lib.waste_image_dimensions.argtypes = [C.c_char_p, C.POINTER(C.c_int),
                                           C.POINTER(C.c_int)]
    lib.waste_image_expand.restype = C.c_int
    lib.waste_image_expand.argtypes = [C.c_void_p, C.POINTER(C.c_int32),
                                       C.c_size_t, C.POINTER(C.c_int32),
                                       C.c_size_t, C.POINTER(C.c_size_t)]
    lib.waste_image_clear.restype = None
    lib.waste_image_clear.argtypes = [C.c_void_p]

    lib.waste_gen_params_init.restype = None
    lib.waste_gen_params_init.argtypes = [C.POINTER(GenParams)]
    lib.waste_generate.restype = C.c_int
    lib.waste_generate.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.c_size_t,
                                   C.POINTER(GenParams), TOKEN_CB, C.c_void_p]
    lib.waste_eval.restype = C.c_int
    lib.waste_eval.argtypes = [C.c_void_p, C.POINTER(C.c_int32), C.c_size_t,
                               C.POINTER(C.POINTER(C.c_float)),
                               C.POINTER(C.c_size_t)]
    # Which expert record failed, when one does. A status of "I/O error" is
    # not something an operator can act on for a container holding 89,000
    # of them; this names the record.
    lib.waste_error_detail.restype = C.c_char_p
    lib.waste_error_detail.argtypes = [C.c_void_p]

    lib.waste_state_save.restype = C.c_int
    lib.waste_state_save.argtypes = [C.c_void_p, C.c_char_p]
    lib.waste_state_load.restype = C.c_int
    lib.waste_state_load.argtypes = [C.c_void_p, C.c_char_p]
    lib.waste_state_reset.restype = None
    lib.waste_state_reset.argtypes = [C.c_void_p]

    lib.waste_model_get_info.restype = C.c_int
    lib.waste_model_get_info.argtypes = [C.c_void_p, C.POINTER(ModelInfo)]
    lib.waste_save_usage.restype = C.c_int
    lib.waste_save_usage.argtypes = [C.c_void_p]
    lib.waste_get_stats.restype = C.c_int
    lib.waste_get_stats.argtypes = [C.c_void_p, C.POINTER(Stats)]
    lib.waste_physical_ram.restype = C.c_uint64
    lib.waste_physical_ram.argtypes = []
    lib.waste_usable_ram.restype = C.c_uint64
    lib.waste_usable_ram.argtypes = []


def version() -> str:
    return _lib().waste_version().decode()


def build_info() -> str:
    return _lib().waste_build_info().decode()


def physical_ram() -> int:
    return int(_lib().waste_physical_ram())


def usable_ram() -> int:
    """Physical RAM, or a smaller cgroup-v2 limit. What a 0 budget sizes
    against — in a container it is the only one of the two that is true."""
    return int(_lib().waste_usable_ram())


def _bounded_int(name: str, value: int, lo: int, hi: int) -> int:
    if (isinstance(value, bool) or not isinstance(value, int) or
            not lo <= value <= hi):
        raise ValueError(f"{name} must be an integer between {lo} and {hi}")
    return value


def plan_memory(model_path: str, ctx_tokens: int = 0) -> MemPlan:
    ctx_tokens = _bounded_int("ctx_tokens", ctx_tokens, 0, (1 << 32) - 1)
    plan = MemPlan()
    st = _lib().waste_plan_memory(str(model_path).encode(), ctx_tokens,
                                  C.byref(plan))
    if st != WASTE_OK:
        raise EngineError("plan_memory", st, str(model_path))
    return plan


# ---- text chunking, upstream's way ---------------------------------------

# tokenization_kimi.py splits before encoding, and the split changes the
# token ids at its boundaries, so a faithful prompt has to split the same
# way. Both limits are upstream's, and both exist to keep tiktoken's Rust
# core from panicking rather than for any modelling reason.
_MAX_ENCODE_CHARS = 400_000
_MAX_RUN_CHARS = 25_000


def _split_runs(s: str, limit: int):
    """Yield pieces with no more than `limit` consecutive spaces or non-spaces.

    A transcription of upstream's _split_whitespaces_or_nonwhitespaces. The
    counter resets on every change of class, so ordinary prose is yielded
    whole and only a pathological run gets cut.
    """
    if not s:
        yield s
        return
    current_len = 0
    current_is_space = s[0].isspace()
    start = 0
    for i, ch in enumerate(s):
        is_space = ch.isspace()
        if current_is_space ^ is_space:
            current_len = 1
            current_is_space = is_space
        else:
            current_len += 1
            if current_len > limit:
                yield s[start:i]
                start = i
                current_len = 1
    yield s[start:]


def _encode_pieces(text: str):
    """The substrings upstream would hand the encoder, in order."""
    for i in range(0, len(text), _MAX_ENCODE_CHARS):
        for piece in _split_runs(text[i:i + _MAX_ENCODE_CHARS], _MAX_RUN_CHARS):
            if piece:
                yield piece


# ---- the engine ----------------------------------------------------------

TokenCallback = Callable[[int, str, TokenInfo], bool]
"""(token_id, piece, info) -> keep going? Returning False cancels."""


class Engine:
    """One loaded model, and the lock that keeps it to one caller at a time."""

    def __init__(self, model_path: str, *,
                 ram_budget_bytes: int = 0,
                 ctx_tokens: int = 0,
                 n_threads: int = 0,
                 cpu_list: Optional[str] = None,
                 cache_policy: int = CACHE_LFRU,
                 # waste_cfg_init's default, and the one the engine's own
                 # hit-rate numbers are measured under. Defaulting to
                 # False here quietly served every request with the page
                 # cache in the way.
                 direct_io: bool = True,
                 vision: bool = False,
                 verify_records: bool = False,
                 usage_path: Optional[str] = None,
                 allow_concurrent_open: bool = False):
        ram_budget_bytes = _bounded_int(
            "ram_budget_bytes", ram_budget_bytes, 0, (1 << 64) - 1)
        ctx_tokens = _bounded_int("ctx_tokens", ctx_tokens, 0, (1 << 32) - 1)
        n_threads = _bounded_int("n_threads", n_threads, 0, (1 << 31) - 1)
        self.lib = _lib()
        self.model_path = str(model_path)
        self._lock = threading.RLock()
        self._ctx = C.c_void_p()
        self._closed = False

        cfg = Cfg()
        self.lib.waste_cfg_init(C.byref(cfg))
        cfg.ram_budget_bytes = ram_budget_bytes
        cfg.ctx_tokens = ctx_tokens
        cfg.n_threads = n_threads
        # Kept alive for the same reason usage_path is, below. waste_open
        # refuses a list it cannot parse or cannot honour, so a bad one
        # surfaces here as EngineError rather than as a run that quietly
        # did not pin anything.
        self._cpus = cpu_list.encode() if cpu_list else None
        cfg.cpu_list = self._cpus
        cfg.cache_policy = cache_policy
        cfg.use_direct_io = 1 if direct_io else 0
        cfg.vision = 1 if vision else 0
        cfg.verify_records = 1 if verify_records else 0
        # Keep the encoded bytes alive: c_char_p stores a borrowed pointer,
        # and a temporary would be freed before waste_open reads it.
        self._usage = usage_path.encode() if usage_path else None
        cfg.usage_path = self._usage
        cfg.allow_concurrent_open = 1 if allow_concurrent_open else 0

        st = self.lib.waste_open(self.model_path.encode(), C.byref(cfg),
                                 C.byref(self._ctx))
        if st != WASTE_OK:
            raise EngineError("open", st, self.model_path)

        self.vision = bool(vision)
        self._markers: Optional[dict[int, str]] = None

    # ---- lifecycle ------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.lib.waste_close(self._ctx)
            self._ctx = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _check(self):
        if self._closed:
            raise EngineError("engine is closed", WASTE_E_ARG)

    def _detail(self) -> str:
        """The engine's own account of the last failure, if it has one —
        which record failed its checksum, rather than just "I/O error"."""
        if self._closed or not self._ctx:
            return ""
        d = self.lib.waste_error_detail(self._ctx)
        return d.decode() if d else ""

    @property
    def lock(self) -> threading.RLock:
        """Held for a whole generation; a waste_ctx takes one caller."""
        return self._lock

    # ---- introspection --------------------------------------------------

    def model_info(self) -> dict:
        self._check()
        info = ModelInfo()
        st = self.lib.waste_model_get_info(self._ctx, C.byref(info))
        if st != WASTE_OK:
            raise EngineError("model_get_info", st)
        return {
            "n_layers": info.n_layers, "n_experts": info.n_experts,
            "top_k": info.top_k, "hidden": info.hidden,
            "ctx_max": info.ctx_max,
            "params_total": info.params_total,
            "params_active": info.params_active,
            "arch": info.arch.decode() if info.arch else "",
            "quant_summary": (info.quant_summary.decode()
                              if info.quant_summary else ""),
        }

    def memory_used(self) -> dict:
        self._check()
        plan = MemPlan()
        st = self.lib.waste_memory_used(self._ctx, C.byref(plan))
        if st != WASTE_OK:
            raise EngineError("memory_used", st)
        return {f: getattr(plan, f) for f, _ in MemPlan._fields_}

    def stats(self) -> dict:
        self._check()
        s = Stats()
        st = self.lib.waste_get_stats(self._ctx, C.byref(s))
        if st != WASTE_OK:
            raise EngineError("get_stats", st)
        return {f: getattr(s, f) for f, _ in Stats._fields_}

    def save_usage(self) -> None:
        self._check()
        st = self.lib.waste_save_usage(self._ctx)
        if st != WASTE_OK:
            raise EngineError("save_usage", st)

    # ---- tokenizer ------------------------------------------------------

    def tokenize(self, text: str, *, markup: bool = False,
                 add_bos: bool = False) -> list[int]:
        """Encode one string. `markup=True` recognises `<|...|>` as controls.

        Which of the two entry points a string goes through is a security
        decision, not a performance one: see waste.h and tokenize_segments.
        """
        self._check()
        fn = (self.lib.waste_tokenize_markup if markup
              else self.lib.waste_tokenize)
        out: list[int] = []
        for piece in _encode_pieces(text):
            data = piece.encode("utf-8")
            # A token is at least one byte, so bytes+1 is always enough and
            # the "would not fit" path cannot be reached.
            cap = len(data) + 1
            buf = (C.c_int32 * cap)()
            n = C.c_size_t()
            st = fn(self._ctx, data, 1 if (add_bos and not out) else 0,
                    buf, cap, C.byref(n))
            if st != WASTE_OK:
                raise EngineError("tokenize", st)
            out.extend(buf[:n.value])
        return out

    def tokenize_segments(self, segments: list[Segment],
                          *, add_bos: bool = False) -> list[int]:
        """Encode a rendered prompt, one segment at a time.

        Two rules, both load-bearing:

        - A markup segment goes to waste_tokenize_markup and everything
          else to waste_tokenize. That is what stops a user, a document or
          a tool result from closing a turn or opening a system message.
        - Segments are encoded *separately*, never concatenated first.
          Upstream's _encode_chat_segments calls the encoder per segment,
          and BPE is not associative: ` role` + `="` + `user` encoded apart
          is a different token sequence than ` role="user"` encoded whole.
          Merging would be a cheap win and a wrong prompt.
        """
        self._check()
        out: list[int] = []
        for seg in segments:
            if not seg.text:
                continue
            out.extend(self.tokenize(seg.text, markup=seg.markup,
                                     add_bos=add_bos and not out))
        return out

    def detokenize(self, tokens: list[int]) -> str:
        self._check()
        if not tokens:
            return ""
        arr = (C.c_int32 * len(tokens))(*tokens)
        cap = 4 * len(tokens) + 64
        while True:
            buf = C.create_string_buffer(cap)
            n = C.c_size_t()
            st = self.lib.waste_detokenize(self._ctx, arr, len(tokens),
                                           buf, cap, C.byref(n))
            if st == WASTE_OK:
                return buf.raw[:n.value].decode("utf-8", "replace")
            # The engine reports a short buffer as E_ARG; grow and retry
            # rather than guessing a bound for arbitrary vocabularies.
            if st == WASTE_E_ARG and cap < (1 << 26):
                cap *= 4
                continue
            raise EngineError("detokenize", st)

    def marker_ids(self) -> dict[int, str]:
        """Control-token id -> marker, for RegionParser.feed_token.

        Each marker must encode to exactly one token in markup mode. One
        that does not is not in the container's specials, which means the
        chat format and the tokenizer disagree — a prompt built from it
        would be markup the model reads as prose. Better to say so here
        than to serve wrong answers.
        """
        if self._markers is not None:
            return self._markers
        markers: dict[int, str] = {}
        for text in (OPEN_TOKEN, CLOSE_TOKEN, SEP_TOKEN, END_OF_MSG_TOKEN):
            ids = self.tokenize(text, markup=True)
            if len(ids) != 1:
                raise EngineError(
                    f"{text} is not a single token in this container "
                    f"(got {len(ids)}): its specials.json does not carry "
                    f"K3's XTML markers", WASTE_E_UNSUPPORTED)
            markers[ids[0]] = text
        self._markers = markers
        return markers

    # ---- images ---------------------------------------------------------

    @staticmethod
    def image_dimensions(path: str) -> tuple[int, int]:
        """Source pixels, from the file header. No context needed."""
        w, h = C.c_int(), C.c_int()
        st = _lib().waste_image_dimensions(str(path).encode(),
                                           C.byref(w), C.byref(h))
        if st != WASTE_OK:
            raise EngineError("image_dimensions", st, str(path))
        return w.value, h.value

    def image_add(self, path: str) -> int:
        """Encode an image and queue it; returns how many positions it takes."""
        self._check()
        n = C.c_size_t()
        st = self.lib.waste_image_add(self._ctx, str(path).encode(), C.byref(n))
        if st != WASTE_OK:
            raise EngineError("image_add", st, str(path))
        return n.value

    def image_expand(self, tokens: list[int]) -> list[int]:
        """Rewrite each media placeholder into the run of slots it needs."""
        self._check()
        if not tokens:
            return []
        src = (C.c_int32 * len(tokens))(*tokens)
        cap = len(tokens)
        while True:
            dst = (C.c_int32 * cap)()
            n = C.c_size_t()
            st = self.lib.waste_image_expand(self._ctx, src, len(tokens),
                                             dst, cap, C.byref(n))
            if st == WASTE_OK:
                return list(dst[:n.value])
            if st == WASTE_E_ARG and cap < (1 << 22):
                cap *= 4          # one placeholder becomes many slots
                continue
            raise EngineError("image_expand", st)

    def image_clear(self) -> None:
        self._check()
        self.lib.waste_image_clear(self._ctx)

    # ---- state ----------------------------------------------------------

    def state_reset(self) -> None:
        self._check()
        self.lib.waste_state_reset(self._ctx)

    def state_save(self, path: str) -> None:
        self._check()
        st = self.lib.waste_state_save(self._ctx, str(path).encode())
        if st != WASTE_OK:
            raise EngineError("state_save", st, str(path))

    def state_load(self, path: str) -> None:
        self._check()
        st = self.lib.waste_state_load(self._ctx, str(path).encode())
        if st != WASTE_OK:
            raise EngineError("state_load", st, str(path))

    # ---- generation -----------------------------------------------------

    def generate(self, prompt: list[int], on_token: TokenCallback, *,
                 temperature: float = 0.0,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 seed: Optional[int] = None,
                 max_tokens: int = 512,
                 grammar: Optional[str] = None,
                 stop_tokens: Optional[list[int]] = None) -> bool:
        """Run one generation. Returns False if the callback stopped it.

        The lock is held throughout: one waste_ctx, one caller. The callback
        runs on this thread, inside the C call, so it must not re-enter the
        engine — the server writes SSE from it and nothing else.

        An exception raised in the callback cannot propagate through the C
        frame, so it is stashed, the callback asks the engine to stop, and
        it is re-raised here once waste_generate has returned cleanly.
        """
        self._check()
        if not prompt:
            raise EngineError("generate", WASTE_E_ARG, "empty prompt")
        max_tokens = _bounded_int("max_tokens", max_tokens, 1, (1 << 32) - 1)
        if seed is not None:
            seed = _bounded_int("seed", seed, 0, (1 << 64) - 1)
        if top_k is not None:
            top_k = _bounded_int("top_k", top_k, 0, (1 << 31) - 1)

        params = GenParams()
        self.lib.waste_gen_params_init(C.byref(params))
        params.temperature = temperature
        if top_p is not None:
            params.top_p = top_p
        if top_k is not None:
            params.top_k = top_k
        if seed is not None:
            params.seed = seed
        params.max_tokens = max_tokens
        self._grammar = grammar.encode() if grammar else None   # keep alive
        params.grammar = self._grammar

        stops = None
        if stop_tokens:
            stops = (C.c_int32 * len(stop_tokens))(*stop_tokens)
            params.stop_tokens = stops
            params.n_stop = len(stop_tokens)

        raised: list[BaseException] = []
        stopped = [False]

        def trampoline(info_ptr, piece_ptr, _user):
            try:
                info = info_ptr.contents
                piece = (piece_ptr.decode("utf-8", "replace")
                         if piece_ptr else "")
                if on_token(info.token, piece, info):
                    return 0
                stopped[0] = True
                return 1
            except BaseException as e:                  # noqa: BLE001
                raised.append(e)
                stopped[0] = True
                return 1

        cb = TOKEN_CB(trampoline)
        arr = (C.c_int32 * len(prompt))(*prompt)
        with self._lock:
            self._check()
            st = self.lib.waste_generate(self._ctx, arr, len(prompt),
                                         C.byref(params), cb, None)
        if raised:
            raise raised[0]
        if st == WASTE_E_CANCELLED or stopped[0]:
            return False
        if st != WASTE_OK:
            raise EngineError("generate", st, self._detail())
        return True
