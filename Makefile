# WASTE — embeddable MoE inference engine + CLI
#
#   make            library + cli
#   make test       the validation binaries
#   make WASTE_ENABLE_METAL=1     (accelerators are build-time options)

# Explicit, because per-object rules below would otherwise make the first
# of them the default goal.
.DEFAULT_GOAL := all

CC      ?= cc
# gnu11, not c11: with -std=c11 glibc sets __STRICT_ANSI__ and hides every
# POSIX extension, so pread, fcntl, posix_memalign and pthread_* would all
# be implicitly declared on Linux. Only model.c defines _GNU_SOURCE itself.
CFLAGS  ?= -O2 -std=gnu11 -Wall -Wextra
LDLIBS  := -lm -lpthread

# The target triple rather than `uname -m`, because those differ the moment
# anyone cross-compiles: building for Windows from this ARM laptop, uname
# says arm64 and would put the NEON translation unit into an x86 binary
# while leaving out the two SIMD ones the target actually dispatches to.
TRIPLE  := $(shell $(CC) -dumpmachine 2>/dev/null)
# Stock MinGW ships gcc.exe and no cc.exe, so the default CC does not exist
# there and -dumpmachine answered nothing. The fallback below was `uname -m`,
# which on MSYS says x86_64 and never contains "mingw" — so WINDOWS, EXE and
# SOEXT stayed unset and make quietly built for the wrong platform, first
# noticed when it tried to link an extensionless test_kda with a compiler
# that is not installed. A missing compiler must not look like a Linux host.
CC0     := $(CC)
# A CC given on the command line wins over this assignment, by make's own
# rules — so an explicitly named compiler that does not work is an error
# rather than something quietly replaced.
ifeq (,$(TRIPLE))
CC      := gcc
TRIPLE  := $(shell $(CC) -dumpmachine 2>/dev/null)
endif
ifeq (,$(TRIPLE))
$(error no working C compiler: `$(CC0)` does not answer -dumpmachine$(if \
$(filter-out $(CC0),$(CC)), and neither does `$(CC)`). Name one that does, \
e.g. `make CC=gcc` on MinGW)
endif
ARCH    := $(TRIPLE)

# Shared-library suffix: convert.py and serve/engine.py look for all three
# through ctypes.
ifneq (,$(findstring mingw,$(ARCH)))
WINDOWS := 1
endif
ifdef WINDOWS
SOEXT   := dll
EXE     := .exe
# The archiver has to match the target as well as the compiler. macOS `ar`
# writes an archive of PE objects without complaining; what comes out is
# 96 bytes and a page of undefined references at link time, with nothing
# anywhere saying the archiver was the wrong one.
#
# The -posix/-win32 strip is for Debian and Ubuntu, which ship the two
# threading models as separate compilers. Only -posix has pthread.h, so
# that is the one to build with — and its name does not end in `gcc`.
CCBASE  := $(patsubst %-posix,%,$(patsubst %-win32,%,$(CC)))
AR      := $(if $(filter %gcc,$(CCBASE)),$(patsubst %gcc,%ar,$(CCBASE)),ar)
# msvcrt's printf has no %zu and no C99 anything. mingw ships a conforming
# one; this is what selects it, and without it every size_t we print is
# garbage rather than a compile error.
CFLAGS  += -D__USE_MINGW_ANSI_STDIO=1
# One self-contained .exe/.dll: winpthreads and libgcc go inside rather
# than turning `waste.exe` into a DLL hunt on a machine with no toolchain.
# libwastevq needs it for the same reason from the other direction — it is
# loaded by convert.py through ctypes, where a missing dependency surfaces
# as "could not be found" naming the library that was found.
STATIC  := -static
LDLIBS  += $(STATIC)
# ld exports everything from a DLL only until one symbol is marked
# dllexport. Nothing here is, but saying it keeps serve/'s ctypes binding
# working the day one is.
SHLDFLAGS := -Wl,--export-all-symbols
else ifeq ($(shell uname -s),Darwin)
SOEXT   := dylib
# Same rule as the Windows branch above, from the other direction: the
# archiver has to match the target, and here PATH is what gets it wrong.
# Apple's ld requires every 64-bit Mach-O member of an archive to begin on
# an 8-byte boundary, which Apple's ar arranges by padding the member's
# name field (`#1/20` for a 10-character name). GNU ar writes the short
# name form instead, putting the first member at offset 68, and the build
# then gets all the way to the link before failing with "64-bit mach-o
# member 'kda_neon.o' not 8-byte aligned in 'libwaste.a'" — issue #16, on a
# machine with Homebrew's binutils ahead of /usr/bin. Name the system
# archiver rather than trusting PATH. A command-line AR= still overrides.
AR      := $(if $(wildcard /usr/bin/ar),/usr/bin/ar,ar)
else
SOEXT   := so
endif
# Windows code is position-independent by definition and gcc warns, once
# per file, when told again.
PICFLAG := $(if $(WINDOWS),,-fPIC)
VQ_SUPER ?= 2
CFLAGS  += -DVQ_SUPER=$(VQ_SUPER)
# Track header dependencies. Without this a changed struct in a header
# leaves stale objects compiled against the old layout — which links fine
# and then corrupts memory at run time.
CFLAGS  += -MMD -MP

SRC := src/model.c src/kda.c src/backend.c src/ecache.c src/version.c \
       src/tokenizer.c src/waste.c src/vq.c src/vision.c src/image.c \
       src/crc32.c src/memory.c
# Match what backend.c tests for. Linux/aarch64 reports "aarch64", which
# does not contain "arm" — the old findstring left kda_neon.c out of the
# build while backend.c still emitted the call to it, so the link failed
# with an undefined waste_kda_register_neon.
ifneq (,$(filter arm% aarch64%,$(ARCH)))
SRC += src/kda_neon.c
endif

# One translation unit per x86 ISA, each built with its own flags so the
# baseline binary stays runnable on a CPU that has neither. waste_backend_init
# picks between them from CPUID, so a single binary adapts at run time —
# which is the whole reason these are separate files rather than #ifdefs
# inside model.c.
ifneq (,$(filter x86_64% amd64%,$(ARCH)))
X86SRC  := src/simd_avx2.c src/simd_avx512.c
SRC     += $(X86SRC)
endif

# WASTE_NATIVE=1 builds for this exact CPU, which on ARMv8.6 turns on the
# SMMLA batched matmul (still opt-in at runtime with WASTE_I8MM=1 — it
# quantizes activations to int8, so it does not produce the f32 numbers).
# The default build stays portable across ARM.
ifdef WASTE_NATIVE
CFLAGS += -mcpu=native
endif
# Accelerator backends are build-time options, and each needs a source file
# that registers it. Metal has one — src/metal.m, so WASTE_ENABLE_METAL=1
# builds and `waste version` then reports `backend Metal`. CUDA and BLAS do
# not, and before these checks existed their flags were reachable and
# produced only "Undefined symbols: _waste_register_cuda" at link time.
# Fail early and say why instead.
#
# Metal keeps its check even though it passes: the guard is about the file
# being there, not about the backend being unfinished, and it is what turns
# a deleted or unstaged src/metal.m into a sentence rather than a link
# error. Do not read the $(error) text below as the status of a backend —
# it is the message for a missing file, and Make never expands it while the
# file is present.
ifdef WASTE_ENABLE_METAL
ifeq (,$(wildcard src/metal.m))
$(error WASTE_ENABLE_METAL=1, but src/metal.m does not exist — the Metal \
backend is not implemented. Build without the flag: CPU+NEON is the only \
backend this engine has)
endif
CFLAGS += -DWASTE_ENABLE_METAL=1
OBJCSRC := src/metal.m
LDLIBS += -framework Metal -framework Foundation
endif
ifdef WASTE_ENABLE_CUDA
ifeq (,$(wildcard src/cuda.cu))
$(error WASTE_ENABLE_CUDA=1, but src/cuda.cu does not exist — the CUDA \
backend is not implemented. Build without the flag: CPU+NEON is the \
default, and Metal is the only accelerator this engine has)
endif
CFLAGS += -DWASTE_ENABLE_CUDA=1
SRC    += src/cuda.cu
LDLIBS += -lcudart
endif
ifdef WASTE_ENABLE_BLAS
ifeq (,$(wildcard src/blas.c))
$(error WASTE_ENABLE_BLAS=1, but src/blas.c does not exist — the BLAS \
backend is not implemented. Build without the flag: CPU+NEON is the \
default, and Metal is the only accelerator this engine has)
endif
CFLAGS += -DWASTE_ENABLE_BLAS=1
SRC    += src/blas.c
LDLIBS += -lblas
endif

OBJ := $(SRC:.c=.o) $(OBJCSRC:.m=.o)

src/metal.o: src/metal.m
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<

# `override`, and it is load-bearing. A plain target-specific `CFLAGS +=`
# is discarded whenever CFLAGS arrives from the command line — which is
# exactly what `make asan` and `make fuzz-asan` do when they re-enter make
# with the sanitizer flags. Without it those builds compile simd_avx2.c
# with no -mavx2, and gcc then refuses to inline the always_inline AVX
# intrinsics ("target specific option mismatch") instead of doing anything
# so helpful as warning that the flag went missing. Invisible on ARM,
# where these translation units are not in SRC at all.
src/simd_avx2.o:   override CFLAGS += -mavx2 -mfma
src/simd_avx512.o: override CFLAGS += -mavx512f -mavx512bw

all: waste$(EXE) libwaste.a libwaste.$(SOEXT) libwastevq.$(SOEXT)

# `make` builds the shipped artifacts; `make test` also builds the checkers.
# They are separate targets, so remember which one you need — testing a
# stale test binary costs more time than rebuilding it.

# shared object so tools/convert.py can call the encoder through ctypes
libwastevq.$(SOEXT): src/vq.c
	$(CC) $(CFLAGS) -shared $(PICFLAG) -o $@ $< -lm -lpthread $(STATIC)

libwaste.a: $(OBJ)
	$(AR) rcs $@ $^

# The whole engine as a shared object. waste.h says the engine is a
# library first and the CLI is just one of its clients; serve/ is the
# other one, and it reaches the same functions through ctypes rather
# than through a second copy of the engine in Python. Built from its own
# objects because -fPIC is not in CFLAGS for the static path: mixing a
# non-PIC libwaste.a into a shared object fails to link on Linux.
SHOBJ := $(SRC:.c=.pic.o) $(OBJCSRC:.m=.pic.o)

%.pic.o: %.c
	$(CC) $(CFLAGS) $(PICFLAG) -c -o $@ $<

src/metal.pic.o: src/metal.m
	$(CC) $(CFLAGS) $(PICFLAG) -fobjc-arc -c -o $@ $<

src/simd_avx2.pic.o:   override CFLAGS += -mavx2 -mfma
src/simd_avx512.pic.o: override CFLAGS += -mavx512f -mavx512bw

libwaste.$(SOEXT): $(SHOBJ)
	$(CC) $(CFLAGS) -shared -o $@ $^ $(SHLDFLAGS) $(LDLIBS)

waste$(EXE): cli/main.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# One list, used by both `test` and `clean`. A stale test binary is one of
# the two failures tests/run.sh was written to catch, so a binary that
# `test` builds and `clean` forgets defeats the check meant to notice it.
TESTNAMES := test_kda test_container test_forward test_tokenizer test_k3parts \
             test_state test_vision test_image test_memory test_cpus test_lock sweep
TESTBINS  := $(addsuffix $(EXE),$(TESTNAMES))

test: $(TESTBINS)

test_kda$(EXE): tests/test_kda.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
# Every link rule passes LDLIBS, including the one target that does not
# currently need it. `test_image` omitted it and linked fine on macOS for
# a week: clang folded the one sqrt() in image.c, glibc did not, and the
# Linux build failed on an undefined reference the day CI first saw it.
# The one test that links no engine: it reads the container with nothing
# but the format header. crc32.c comes along because the checksum in that
# header is part of the format, and checking it against zlib's values is
# what says the two agree.
test_container$(EXE): tests/test_container.o src/crc32.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_forward$(EXE): tests/test_forward.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Not a check — a measurement harness. It is here because it links the same
# library the checks do and must never drift from it.
sweep$(EXE): tests/sweep.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_tokenizer$(EXE): tests/test_tokenizer.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_k3parts$(EXE): tests/test_k3parts.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_image$(EXE): tests/test_image.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_vision$(EXE): tests/test_vision.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
test_state$(EXE): tests/test_state.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
# Links the cgroup reader alone, not the engine: the point of the paths
# being parameters is that the policy is checkable in milliseconds against
# synthetic files, with no container and no /proc on the host to depend on.
test_memory$(EXE): tests/test_memory.o src/memory.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
# The pool is a header, so this links no engine at all — which is the point:
# it can check placement without a container, and the parse half of it runs
# on the platforms that cannot bind a thread.
test_cpus$(EXE): tests/test_cpus.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

test_lock$(EXE): tests/test_lock.o libwaste.a
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

-include $(OBJ:.o=.d) $(SHOBJ:.o=.d) cli/main.d \
        $(patsubst %.c,%.d,$(wildcard tests/*.c))

clean:
	rm -f $(OBJ) $(SHOBJ) cli/*.o tests/*.o $(OBJ:.o=.d) $(SHOBJ:.o=.d) \
	      cli/*.d tests/*.d libwaste.a waste waste.exe \
	      $(TESTBINS) $(TESTNAMES) $(addsuffix .exe,$(TESTNAMES)) \
	      libwaste.dylib libwaste.so libwaste.dll \
	      libwastevq.dylib libwastevq.so libwastevq.dll
	rm -rf libwastevq.dylib.dSYM libwaste.dylib.dSYM
	rm -rf serve/__pycache__ tests/serve/__pycache__

check: test
	@tests/run.sh

# The Python server's own suite. Separate from `check` because it needs
# libwaste as a shared object rather than the archive the CLI links, and
# because it is the one part of this repo that is not C.
serve-check: libwaste.$(SOEXT)
	@python3 -m unittest discover -s tests/serve -t . -p "test_*.py"

# Sanitizers. Separate targets rather than a flag on `make`, because they
# need a full rebuild: mixing instrumented and uninstrumented objects
# produces false reports and missed ones. ASan's own allocator refuses
# very large mappings, so these run against a synthetic container, which
# is what tests/run.sh falls back to when given a path that does not exist.
SAN_FLAGS := -fsanitize=address,undefined -fno-omit-frame-pointer \
             -fno-sanitize-recover=all -g -O1

asan:
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory all test \
	    CFLAGS="-std=gnu11 -Wall -Wextra -DVQ_SUPER=$(VQ_SUPER) -MMD -MP $(SAN_FLAGS)" \
	    LDLIBS="-lm -lpthread $(SAN_FLAGS)"
	@rc=0; ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 \
	    WASTE_SANITIZED=1 \
	    tests/run.sh /nonexistent-container-use-synthetic || rc=$$?; \
	 $(MAKE) --no-print-directory clean; exit $$rc

fuzz: test
	@python3 tools/fuzz_container.py --runs $(FUZZ_RUNS)

FUZZ_RUNS ?= 200

# What CI runs: the fuzzer against an instrumented build, so a read past
# the end of a buffer aborts instead of returning plausible garbage.
fuzz-asan:
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory all test \
	    CFLAGS="-std=gnu11 -Wall -Wextra -DVQ_SUPER=$(VQ_SUPER) -MMD -MP $(SAN_FLAGS)" \
	    LDLIBS="-lm -lpthread $(SAN_FLAGS)"
	@rc=0; ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=print_stacktrace=1 \
	    python3 tools/fuzz_container.py --runs $(FUZZ_RUNS) || rc=$$?; \
	 $(MAKE) --no-print-directory clean; exit $$rc

.PHONY: all test check serve-check clean asan fuzz fuzz-asan
