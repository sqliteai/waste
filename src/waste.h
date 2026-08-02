/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * waste.h — public API of the WASTE inference engine.
 *
 * The engine is a library first: everything the CLI does is done through
 * the functions below, so the same capabilities are available to any host
 * program (editor plugin, server, app). The CLI in cli/ is a client of
 * this header and gets no private access.
 *
 * Design rules
 *   - C11, no dependencies, no global state: every instance is a
 *     waste_ctx, so a host can hold several models at once.
 *   - The caller owns the RAM budget. waste_cfg.ram_budget_bytes is a hard
 *     ceiling on everything the engine allocates (weights, caches, KV,
 *     scratch). The engine sizes its expert cache to fit and never
 *     exceeds the budget; if the budget is below the floor
 *     (waste_plan_memory's floor_bytes), loading fails with
 *     WASTE_E_RAM_BUDGET instead of swapping the machine to death.
 *   - No hidden I/O: expert streaming happens on the caller's thread or
 *     on the engine's own pool, never via page-fault surprises.
 *   - Errors are returned, never printed. Nothing calls exit().
 *
 * Threading: a waste_ctx is not thread-safe; use one per thread, or
 * serialize calls. Distinct contexts are independent.
 */

#ifndef WASTE_H
#define WASTE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WASTE_API_VERSION 1

/* ---- engine version ----------------------------------------------------
 * Semantic versioning. The numeric form is MAJOR*10000 + MINOR*100 + PATCH,
 * so it compares with a plain integer test (SQLite's convention):
 *     #if WASTE_VERSION_NUMBER >= 10200   // needs >= 1.2.0
 * The compile-time macros describe the headers the caller built against;
 * the functions report the library actually linked, which is what matters
 * for an embeddable engine that may be updated independently.
 */
#define WASTE_VERSION_MAJOR  0
#define WASTE_VERSION_MINOR  6
#define WASTE_VERSION_PATCH  6
#define WASTE_VERSION_STRING "0.6.6"
#define WASTE_VERSION_NUMBER (WASTE_VERSION_MAJOR * 10000 + \
                              WASTE_VERSION_MINOR * 100 + \
                              WASTE_VERSION_PATCH)

const char *waste_version(void);         /* e.g. "0.6.2"                    */
int         waste_version_number(void);  /* e.g. 600                        */
/* Build details: backend, SIMD, container format version. Never NULL. */
const char *waste_build_info(void);

/* ---- errors ------------------------------------------------------------ */

typedef enum {
    WASTE_OK = 0,
    WASTE_E_IO = -1,           /* file missing, short read, bad checksum   */
    WASTE_E_FORMAT = -2,       /* container malformed or wrong version     */
    WASTE_E_RAM_BUDGET = -3,   /* budget below the model's floor           */
    WASTE_E_OOM = -4,
    WASTE_E_ARG = -5,
    WASTE_E_UNSUPPORTED = -6,  /* arch/quant combination not built in      */
    WASTE_E_CANCELLED = -7,    /* callback asked to stop                   */
    WASTE_E_BUSY = -8,         /* another process owns this container      */
} waste_status;

/* Human-readable, static storage; never NULL. A coarse answer by design:
 * an expert record that fails its checksum is a WASTE_E_IO like any
 * other, and waste_error_detail below is what says which record it was. */
const char *waste_strerror(waste_status s);

/* ---- memory planning (usable before loading anything) ------------------ */

typedef struct {
    uint64_t trunk_bytes;        /* resident weights: attn, routers,
                                    shared experts, embeddings, head      */
    uint64_t state_bytes;        /* KDA recurrent state + MLA latent KV    */
    uint64_t scratch_bytes;      /* activations, dequant staging, threads  */
    uint64_t min_expert_cache;   /* smallest cache that can run a layer    */
    uint64_t floor_bytes;        /* sum of the above: the hard minimum     */
    uint64_t recommended_bytes;  /* floor + a cache worth having           */
    /* What vision would add if cfg.vision were set; 0 without a tower.
     * Includes its weights, bounded source decode, tower activations and
     * the queued image embeddings. waste_open folds it into the figures
     * above when vision is on, so the resolved budget accounts for it. */
    uint64_t vision_bytes;
} waste_memplan;

/* Compute the memory floor without loading weights. ctx_tokens sizes the
 * KV/state part. Cheap: reads the manifest and optional vision config. */
waste_status waste_plan_memory(const char *model_path, uint32_t ctx_tokens,
                               waste_memplan *out);

/* ---- configuration ----------------------------------------------------- */

/* Both are implemented; there is no third. A "pinned" policy that honours
 * the hotlist and never evicts it used to be listed here and was never
 * written — waste_ecache_init took a policy argument the engine always
 * passed 0 for, so every one of these selected LFRU. */
typedef enum {
    WASTE_CACHE_LFRU = 0,  /* frequency-first, recency tiebreak (default)  */
    WASTE_CACHE_LRU = 1,
} waste_cache_policy;

typedef struct {
    /* Hard ceiling on all engine allocations, used exactly as given.
     *
     * 0 = the engine picks, and it picks conservatively: expert cache is
     * only useful in whole multiples of one token's working set, so it
     * starts from waste_memplan.recommended_bytes (floor + 3x that set)
     * and steps down a whole multiple at a time until the total fits
     * under 7/8 of waste_usable_ram(). It does not spend the remainder up to
     * that ceiling: the last fraction buys a little hit rate and risks
     * the OS paging the cache out, which costs far more than it gains.
     * When not even floor + 1x fits, it runs at floor_bytes.
     *
     * Loading fails with WASTE_E_RAM_BUDGET if the value given is below
     * floor_bytes. waste_memory_used reports what was actually resolved. */
    uint64_t ram_budget_bytes;

    /* Longest sequence a context holds — prompt plus generation — and a
     * hard bound rather than a hint, because MLA stores one latent per
     * position and the cache is allocated to exactly this length.
     * 0 = 4096. See waste_generate. */
    uint32_t ctx_tokens;

    /* Compute threads. 0 = hardware concurrency; WASTE_THREADS overrides
     * a 0 but not a value set here.
     *
     * The pool is process-wide rather than per context: the first
     * waste_open sizes it and later ones reuse it, so a host holding two
     * models runs both on the same threads. The kernels split by row, so
     * results do not depend on the count either way. */
    int      n_threads;

    /* Which CPUs the compute pool may run on: a Linux-style cpu list,
     * "0-5" or "0-2,6-8" or "3". NULL = wherever the OS puts them, which
     * is the default and stays the default. WASTE_CPUS fills in a NULL but
     * does not override a value set here.
     *
     * Worth setting when the machine's cores are not interchangeable. On a
     * two-CCD Ryzen 9 (two 6-core dies, separate 32 MB L3), six threads on
     * one die measured 16-25% faster than the same six split across both
     * at byte-identical work, because the far die's L3 is an Infinity
     * Fabric hop away and every barrier waits for whoever is slowest —
     * third-party measurement, issue #23. docs/LEARNED.md §47 is the same
     * effect between P-cores and E-cores. There is no good default here:
     * §47 measured capping the pool as a 25% gain on one model and a 34%
     * loss on another, so the engine picks nothing and this is how a host
     * that knows its machine says so.
     *
     * Two consequences worth knowing before setting it:
     *
     *   - n_threads 0 then means one thread per CPU in the list, not one
     *     per CPU in the machine.
     *   - the thread that calls into the engine is a worker, so it is
     *     bound too, the first time it runs a kernel — and it stays bound
     *     afterwards. A host that also uses that thread for something else
     *     is restricting that too.
     *
     * The expert cache's reader threads are deliberately left out: they
     * spend their lives blocked in pread, and putting them on the same
     * CPUs as the kernels they feed is contention for nothing.
     *
     * A list that is malformed, or that names no CPU, fails waste_open
     * with WASTE_E_ARG. A well-formed list on a platform that cannot bind
     * threads — macOS, which has no such call — fails with
     * WASTE_E_UNSUPPORTED rather than being ignored: silently not pinning
     * is indistinguishable from pinning that did not help, and that is the
     * one answer this option must never give. */
    const char *cpu_list;

    waste_cache_policy cache_policy;

    /* Read expert banks with the page cache out of the way: F_NOCACHE on
     * macOS, O_DIRECT on Linux, FILE_FLAG_NO_BUFFERING on Windows.
     * **On by default**, and it is what makes the hit rates the engine
     * reports its own rather than the kernel's — with a 17 GB container
     * on a 64 GB machine, measuring without it measures the kernel.
     *
     * A filesystem may refuse the bypass and a container whose records
     * are not page multiples cannot use it; waste_stats.direct_io then
     * comes back 0. WASTE_DIRECT=0 forces it off for a process, which is
     * how to measure what the page cache is worth. */
    int      use_direct_io;

    int      vision;            /* enable images; reserves vision_bytes,
                                   1.12 GB on K3 — see waste_memplan       */

    /* Check each expert record's crc32 as it comes off the disk. **Off by
     * default**, and that is a throughput decision rather than a claim
     * that containers do not rot: it is a pass over every record on every
     * cache miss, measured at ~5% on Kimi-Linear and ~1% on K3, where the
     * read dominates. Turn it on for a container that has been copied,
     * downloaded or left on a disk you do not trust, and for anything
     * whose wrong answers would be believed.
     *
     * What is checked regardless: a short read, and a record header that
     * does not describe the expert the bank index asked for. Those are
     * O(1) and they are what keeps a damaged offset out of the
     * arithmetic — this flag only adds the pass over the payload.
     *
     * WASTE_VERIFY=1 in the environment turns it on too. Either is
     * enough; neither turns it off. */
    int      verify_records;

    /* Learned hotlist: which experts a previous run used, read at open to
     * warm the cache and written by waste_save_usage. NULL = the
     * container's own, <model_path>/usage.waste. Point it elsewhere to
     * keep per-workload hotlists, or at a read-only path to warm from one
     * without writing it back.
     *
     * The file is trusted only as a hint: entries naming a layer or an
     * expert this container does not have are skipped, because it is one
     * of the few files the engine reads that nobody asked it to. */
    const char *usage_path;

    /* On POSIX hosts, one process owns a container at a time by default.
     * Multiple contexts in that process share the ownership; a competing
     * process receives WASTE_E_BUSY before model-sized allocations begin.
     * Set this only when the host deliberately accepts competing loads.
     * It is ignored on platforms without the ownership lock. */
    int allow_concurrent_open;
} waste_cfg;

/* Removed in 0.6.0, having never done anything: `io_threads` (there is no
 * expert-fetch pool — reads happen on the calling thread), `expert_deferral`
 * (no overlap of a fetch with the next layer), `allow_substitutes` (WQ_SUB1
 * substitute records are specified in the format and not written by the
 * converter in v0, so there is nothing to substitute), and `state_path`
 * (session state is persisted by calling waste_state_save/load, not by
 * configuring a directory). Each was set by the CLI and the server and
 * read by nothing. */

/* Fills cfg with defaults. Always call this before setting fields, so
 * new fields added in later versions stay sane. */
void waste_cfg_init(waste_cfg *cfg);

/* ---- lifecycle --------------------------------------------------------- */

typedef struct waste_ctx waste_ctx;

waste_status waste_open(const char *model_path, const waste_cfg *cfg,
                        waste_ctx **out);
void waste_close(waste_ctx *ctx);

/* What the engine actually allocated, after open. */
waste_status waste_memory_used(const waste_ctx *ctx, waste_memplan *out);

/* What went wrong, specifically, when the status alone is too coarse to
 * act on: "expert 412 of layer 37: checksum mismatch" rather than "I/O
 * error". NULL when there is nothing to add. Owned by the context and
 * valid until the next eval or generate on it.
 *
 * Also set after a *successful* waste_generate that stopped because the
 * context filled rather than because the model finished or max_tokens
 * ran out. That is not an error — the tokens produced are good — but it
 * is the one ending a host cannot tell from the others by the status
 * alone, and a UI that says "the answer is complete" when it is not is
 * worse than one that says nothing.
 *
 * Every expert record carries a crc32, and the engine checks it as the
 * record comes off the disk — so a container that rots after conversion
 * ends the generation with WASTE_E_IO instead of quietly answering with
 * whatever the damaged bytes happen to decode to. The conversation state
 * is undefined afterwards, because the pass stopped in the middle of a
 * layer: call waste_state_reset before reusing the context. */
const char *waste_error_detail(const waste_ctx *ctx);

/* ---- tokenizer --------------------------------------------------------- */

/* Ordinary text. A `<|...|>` marker inside it is encoded as the ordinary
 * tokens it looks like — never as a control token — so text from a user,
 * a document or a tool result cannot close the current turn or forge a
 * system message. Use this for anything you did not write yourself. */
waste_status waste_tokenize(waste_ctx *ctx, const char *text, int add_bos,
                            int32_t *out_tokens, size_t cap, size_t *n_out);

/* The same, with the container's special tokens recognised: this is what
 * a chat template is made of, and what `waste tokenize` reports so a
 * template can be checked marker by marker. Build a prompt by calling
 * this for the markup and waste_tokenize for the content, the way
 * K3's own tokenizer splits allowed_special from disallowed_special —
 * concatenating them into one string and encoding it once hands whoever
 * wrote the content the ability to write the structure too. */
waste_status waste_tokenize_markup(waste_ctx *ctx, const char *text, int add_bos,
                                   int32_t *out_tokens, size_t cap, size_t *n_out);
/* On a short output buffer returns WASTE_E_ARG and stores the required byte
 * count (not including the trailing NUL) in n_out. */
waste_status waste_detokenize(waste_ctx *ctx, const int32_t *tokens, size_t n,
                              char *out, size_t cap, size_t *n_out);

/* ---- images ------------------------------------------------------------ */

/* Requires cfg.vision at open; without the tower these return
 * WASTE_E_UNSUPPORTED.
 *
 * An image is not a token. It becomes a run of them: the tower turns a
 * patch grid into one embedding per merged 2x2 patch, and each of those
 * occupies a position in the sequence. So the flow is three steps, in
 * this order:
 *
 *   waste_image_add(ctx, "photo.png", &n);   -- encode, queue, learn n
 *   waste_image_expand(...)                  -- one placeholder -> n slots
 *   waste_generate(...)                      -- consumes the queue
 *
 * The placeholder is whatever the container names as its media token
 * (<|media_pad|> on K3), and the prompt needs exactly one per queued
 * image; expand rewrites each into as many copies as that image needs,
 * in queue order. Skipping expand is not a shortcut — the model would
 * see one image position and the rest of the embeddings would go
 * nowhere.
 *
 * `out` must not alias `tokens`: expand grows the array, so writing over
 * its own input would overwrite tokens it has not read yet.
 *
 * Encoding happens in add, so its cost is paid before generation starts
 * and a bad file is reported as a load error rather than mid-prompt.
 * The queue is consumed by the next eval or generate: a second turn
 * about the same picture needs no re-add, because the first turn's
 * positions are already in the attention state. */
waste_status waste_image_add(waste_ctx *ctx, const char *path,
                             size_t *n_tokens_out);

/* Source pixel dimensions, read from the file header without decoding it
 * and without needing a context. K3 wraps an image as
 *   <|media_begin|>image WxH<|media_content|><|media_pad|><|media_end|>
 * and those are the *original* dimensions, not the resized patch grid —
 * the model was trained to be told the resolution it is looking at. A
 * host building the media block itself needs them before add(). */
waste_status waste_image_dimensions(const char *path, int *w, int *h);
waste_status waste_image_expand(waste_ctx *ctx, const int32_t *tokens, size_t n,
                                int32_t *out, size_t cap, size_t *n_out);
void waste_image_clear(waste_ctx *ctx);

/* ---- generation -------------------------------------------------------- */

typedef struct {
    float temperature;      /* 0 = greedy                                  */
    float top_p;
    int   top_k;
    uint64_t seed;
    uint32_t max_tokens;
    const char *grammar;    /* GBNF text for constrained output, or NULL   */
    const int32_t *stop_tokens;
    size_t n_stop;
} waste_gen_params;

void waste_gen_params_init(waste_gen_params *p);

/* Per-token statistics handed to the callback: enough for a host to draw
 * a progress UI and for us to see whether the cache is doing its job. */
typedef struct {
    uint32_t token_index;
    int32_t  token;
    float    logprob;
    uint32_t experts_hit;      /* served from RAM cache                    */
    uint32_t experts_missed;   /* fetched from disk this token             */
    uint64_t bytes_read;
    double   ms_total;
    double   ms_io;
} waste_token_info;

/* Return 0 to continue, non-zero to stop generation (WASTE_E_CANCELLED). */
typedef int (*waste_token_cb)(const waste_token_info *info, const char *piece,
                              void *user);

/* Both of these are bounded by cfg.ctx_tokens: MLA keeps one latent per
 * position and the cache is exactly that long, so the sequence a context
 * can hold is fixed at open. A prompt that does not fit in what is left
 * is WASTE_E_ARG, refused before anything is evaluated, so the
 * conversation is left as it was rather than half-prefilled;
 * waste_error_detail gives the position and the ceiling. Generation that
 * *reaches* the ceiling mid-answer stops there and returns WASTE_OK — see
 * waste_error_detail above. Call waste_state_reset to start over, or open
 * with a larger ctx_tokens. */
waste_status waste_generate(waste_ctx *ctx, const int32_t *prompt, size_t n,
                            const waste_gen_params *params,
                            waste_token_cb cb, void *user);

/* Lower-level: one decode step, for hosts driving their own sampling. */
waste_status waste_eval(waste_ctx *ctx, const int32_t *tokens, size_t n,
                        const float **logits_out, size_t *vocab_out);

/* ---- conversation state ------------------------------------------------ */

/* KDA state is O(1) in context length and MLA KV is compressed, so a whole
 * session checkpoint is small — worth persisting when a cold re-prefill
 * costs minutes at streaming speeds. */
waste_status waste_state_save(waste_ctx *ctx, const char *path);
waste_status waste_state_load(waste_ctx *ctx, const char *path);
void         waste_state_reset(waste_ctx *ctx);

/* ---- introspection ----------------------------------------------------- */

typedef struct {
    uint32_t n_layers, n_experts, top_k, hidden, ctx_max;
    /* The language model: every routed expert plus the trunk, which is
     * attention, the shared experts, the latent projections, the norms and
     * the embeddings. A multimodal container's vision tower is not in
     * either figure. `params_active` is what one token touches — the whole
     * trunk except the embedding table, of which it reads a single row,
     * plus `top_k` experts per MoE layer. */
    uint64_t params_total, params_active;
    const char *arch;           /* e.g. "kimi-k3"                          */
    const char *quant_summary;  /* e.g. "experts VQ2R/VQ3R, trunk Q4G/Q8G" */
} waste_model_info;

waste_status waste_model_get_info(const waste_ctx *ctx, waste_model_info *out);

/* Persist which experts this workload used, so the next open can warm the
 * cache instead of starting cold. Written next to the container. */
waste_status waste_save_usage(waste_ctx *ctx);

/* Aggregate counters since open; a CLI `--stats` prints these. */
typedef struct {
    uint64_t tokens_generated, experts_hit, experts_missed, bytes_read;
    double   sec_total, sec_io;
    /* 0 when a bank could not bypass the page cache (no O_DIRECT support on
     * the filesystem, or a container whose records are not page multiples).
     * The hit rate is then partly the kernel's, not the engine's, and the
     * numbers do not transfer to a machine that cannot cache the model. */
    int      direct_io;
} waste_stats;

waste_status waste_get_stats(const waste_ctx *ctx, waste_stats *out);

/* Physical RAM of this machine, or 0 if it cannot be determined. A budget
 * near this number is counterproductive: the OS pages out the engine's own
 * expert cache, and a hit then costs a page fault. */
uint64_t waste_physical_ram(void);

/* How much of it this process may actually use: physical RAM, or a smaller
 * cgroup-v2 limit when one applies. This is what a budget of 0 sizes
 * against, and it is the number a host should size its own ceiling from.
 *
 * The two differ only on Linux, and there they can differ by everything:
 * sysconf(_SC_PHYS_PAGES) reports the host's RAM from inside a container,
 * so a confined process that trusts it asks for memory the kernel will not
 * give. That failure is a kill, not a slowdown — unlike the paging cliff
 * waste_physical_ram warns about, no cache policy softens it. 0 when
 * neither figure can be determined. */
uint64_t waste_usable_ram(void);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_H */
