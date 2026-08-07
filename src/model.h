/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * model.h — WASTE model loading and forward pass (Kimi-family).
 *
 * Trunk tensors are dequantized to f32 at load; routed experts are read
 * one 4 KiB-aligned record at a time and dequantized on demand, which is
 * the streaming path the engine is built around.
 */

#ifndef WASTE_MODEL_H
#define WASTE_MODEL_H

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

#include "ecache.h"

/* Public image requests are decoded before resize.  Keep the source-image
 * allocation finite so the memory planner can include its true worst case. */
#define WASTE_MAX_SOURCE_PIXELS (64u * 1024u * 1024u)
#define WASTE_MAX_IMAGES 32

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[128];
    float *data;          /* F32 tensors, or NULL when kept quantized       */
    int8_t *q;            /* quantized payload, row-major                   */
    uint16_t *qs;         /* scales: one fp16 per group of `group`          */
    int group;
    int bits;             /* 8, 4 (two per byte) or 3 (packed bitstream)    */
    size_t rowbytes;      /* stride between rows of `q`                     */
    int shape[4], ndim;
    size_t n;
    /* Set when the tensor is deliberately left on disk instead of being
     * made resident (embed_tokens: 1.11 GB of which one row per token is
     * ever read). q and qs are NULL in that case. */
    int64_t file_off, file_scale_off;   /* not long: 32 bits on Windows */
    int    on_disk;
} waste_tensor;

typedef struct {
    int n_layers, hidden, n_experts, top_k, moe_inter, dense_inter;
    int n_shared, first_dense, vocab, n_heads;
    int kv_lora, q_lora, qk_nope, qk_rope, v_head;
    int kda_heads, kda_dim, conv_k;
#define WASTE_MAX_LAYERS 128

/* Vector positions sharing one fp32 scale in the int8 LUT (WQ_VQ4P).
 * Bounds the int16 accumulator: 4 stages x 32 positions x 127 = 16256, so
 * a block cannot overflow before it is folded into fp32. It is also what
 * keeps the quantization honest — |LUT| tracks ||x_v||, which varies a lot
 * across a hidden state, and one global scale would flatten the small
 * positions to zero. */
#define WASTE_VQ_LUT_BLK 32

/* Rotary pairs held per layer: qk_rope_head_dim / 2. 64 covers a 128-wide
 * rope slice; every model in the family uses 64. A container needing
 * rotation on a wider slice is refused at load rather than run unrotated. */
#define WASTE_MAX_ROPE_HALF 64
    int kda_layer[WASTE_MAX_LAYERS]; /* 1 if layer is KDA                   */
    float eps, routed_scale;
    int renorm;

    /* --- K3 additions (all absent/0 for Kimi-Linear) ------------------- */
    int   latent_dim;                /* routed_expert_hidden_size; 0 = none */
    int   latent_norm;               /* latent_moe_use_norm                 */
    int   attn_res_block;            /* attn_res_block_size; 0 = no AttnRes */
    int   full_rank_gate;            /* KDA g_proj instead of g_a/g_b       */
    float gate_lower_bound;          /* 0 = softplus form, else bounded     */
    int   mla_output_gate;           /* MLA sigmoid output gate             */
    int   act_situ;                  /* 1 = SiTU instead of SiLU            */
    float situ_beta, situ_linear_beta;
    char  prefix[64];                /* "" or "language_model." (K3)        */
    /* generation_config.json's eos_token_id, mirrored into the container
     * config. The tokenizer used to derive this positionally as
     * base_vocab + 2, which is right on both Kimi models by luck of the
     * reserved-block layout and is a guess everywhere else. Read it. */
    int   eos_token_id;              /* 0 = not stated, keep the default    */
    /* The HF architecture the container was built from. Both models call
     * themselves model_type "kimi_linear", so this is the only field that
     * tells them apart by name rather than by feature. */
    char  arch[64];

    /* --- rotary -------------------------------------------------------- */
    /* The Kimi models set mla_use_nope and are the reason this was absent:
     * with NoPE the qk_rope dims pass through unrotated. Every DeepSeek-V3
     * model (V3, R1, K2) sets no such flag and needs the rotation, and in
     * MLA those dims are the only positional signal — the nope dims are
     * position-free by construction, so skipping it leaves attention unable
     * to order the sequence. */
    int   mla_nope;                  /* mla_use_nope: 1 = no rotation       */
    float rope_inv_freq[WASTE_MAX_ROPE_HALF];   /* qk_rope/2 used, YaRN-adjusted */
    float att_mul;                   /* YaRN mscale^2 on the attn scale, 1 = none */
    char  rope_err[128];             /* non-empty: a shape rope_init does not
                                      * implement, and why. The load refuses on
                                      * it rather than running unrotated.    */
} waste_config;

typedef struct {
    int fd;                          /* positional reads, no page cache     */
    int64_t rec_bytes;
    int n_experts, cb_base;
} waste_bank;

/* Read from vision.json when the container has a tower. */
typedef struct {
    int hidden, heads, qkv_hidden, inter, layers;
    int pos_h, pos_w, text_hidden, patch;
    float eps, proj_eps;
    float mean[3], std[3];       /* pixel normalization — see image.c      */
    int   media_token;           /* the id an image expands from           */
    int   max_patches;
} waste_vision_cfg;

typedef struct {
    waste_config cfg;
    waste_vision_cfg vcfg;
    int  want_vision;                /* load the tower's 434 MB of weights */
    /* Image embeddings for the prefill about to run: one row per merged
     * patch, consumed in order at each media placeholder. */
    const float *media;
    int      media_n, media_used, cfg_media_token;
    waste_tensor *t;
    int n_tensors;
    float *codebooks;                /* [n_books][256][8]                   */
    float *codebooksT;               /* [n_books][8][256], for the LUT build*/
    int n_books, vec_dim, cb_entries, stages;
    /* 8 = one whole byte of index per stage (VQ3R/VQ2R). 6 = WQ_VQ4P, four
     * indices packed into three bytes, which is what lets a stage table be
     * 64 bytes and so live in a NEON register. Sets the per-row stride of
     * the index stream: `stages` bytes when 8, always 3 when 6. */
    int index_bits;
    /* int8 view of `lut`, built alongside it when index_bits == 6: the
     * table-lookup kernels index bytes, not floats. One fp32 scale per
     * VQ_LUT_BLK vector positions — see vq_quant_lut. */
    int8_t *lut8;
    float  *lut8_scale;
    int8_t *cq8;                     /* the same shadow for m->cq          */
    float  *cq8_scale;

    /* Per-expert scratch for the expert-parallel MoE path: one slice each
     * for the top_k experts, so k threads can run a whole expert — gate,
     * up, activation, down — without sharing a buffer. The row-parallel
     * path dispatches three times per expert and the arithmetic between
     * dispatches is microseconds; this trades that for one dispatch a
     * layer. NULL when top_k is 1 or the allocation was declined. */
    float *xga, *xub, *xacc, *xlut, *xqs;
    int8_t *xlut8;
    size_t xlut_sz, xnsc;            /* floats per down LUT, scales per it */
    /* 1 << fmt for every trunk format the language model actually uses,
     * recorded at load because the tensors left on disk never reach the
     * branch that fills in t->bits. */
    uint32_t trunk_fmts;
    waste_bank bank[WASTE_MAX_LAYERS];
    /* gate, up, down shapes. In a latent MoE the experts are as wide as
     * the latent, not the hidden — these are what a record's per-channel
     * scales are counted from, so they have to be the real ones. */
    int expert_m[3], expert_n[3];

    /* per-layer state */
    float *S[WASTE_MAX_LAYERS];                   /* KDA recurrent state                 */
    float *conv[WASTE_MAX_LAYERS];                /* KDA short-conv rings (3 x C x K-1)  */
    /* MLA caches the latent, not the expanded per-head K and V: kv_b_proj
     * is absorbed into the query and the output instead (see mla_layer). */
    float *latcache[WASTE_MAX_LAYERS];            /* [kv_cap][kv_lora + qk_rope]         */
    int n_kv[WASTE_MAX_LAYERS], kv_cap;

    /* scratch */
    float *x, *h, *tmp, *att, *logits;
    /* 1 when at least one layer is MLA, i.e. when the sequence is bounded
     * by kv_cap. KDA state is O(1) in context and imposes no such limit,
     * so a container that is all KDA is not capped by ctx_tokens. */
    int   has_mla;
    /* Set when a step was asked for a position outside kv_cap. Sticky
     * like read_error and cleared by the same call, because the caller
     * has to hear it once rather than once per layer. */
    int   ctx_full;
    /* chunked prefill scratch (allocated on first use) */
    float *cx, *cnorm, *cresid, *cq, *ckv, *clat, *cff, *cexp;
    float *cblockres, *cprefix;
    int   *croute;
    int   *cused;                   /* chunk's distinct experts, ascending  */
    float *crw;
    int    chunk_cap;

    float *blockres;                 /* AttnRes history: [nblocks][hidden]  */
    int    n_blockres;
    float *prefix_sum, *ares;
    int8_t  *mmxq;                  /* int8 activations for the SMMLA path */
    float   *mmxs;
    size_t   mmx_cap, mms_cap;
    int      trunk_fd;              /* stays open for the on-disk tensors  */
    int8_t  *embrow;                /* one embedding row, read per token   */
    uint16_t *embsc;
    float *qabs, *cacc, *mrow;      /* MLA absorption scratch, per head    */
    float *e_gate, *e_up, *e_down, *ff, *lut, *xs;
    int8_t *xq;
    uint64_t expert_reads;
    waste_ecache cache;
    uint8_t *miss_buf;               /* used when the cache is disabled     */
    int      want_direct;            /* the caller asked for the bypass     */
    int      direct_io;              /* 0 = a bank fell back to page cache  */
    /* A record that failed on the way in. Sticky until cleared, because
     * the forward pass that hit it is already wrong and the caller has to
     * hear about it once rather than once per expert. A short read and a
     * header that does not describe the record asked for are always
     * caught; `verify` adds the crc32 over the payload, which is the part
     * that costs, and it is off unless a caller asks — see waste.h. */
    int      read_error, bad_layer, bad_expert, verify;
    /* bank_fetch runs on the cache's reader threads as well as the compute
     * thread once read-ahead is on, and the two fields above plus the read
     * counter are the only things it writes. Everything else it touches —
     * the bank table, the expert shapes, `verify` — is fixed at load. */
    pthread_mutex_t fetch_mu;
} waste_model;

/* Everything the load needs that is not in the container. These are
 * parameters rather than fields set beforehand because the first thing
 * load does is zero the struct.
 *
 *   cache_bytes  hard ceiling for the expert cache; 0 = no cache
 *   want_vision  load the vision tower (434 MB of weights)
 *   n_threads    compute pool size; 0 = WASTE_THREADS, else the CPUs the
 *                pool may use
 *   cpus         cpu list the pool binds to ("0-5"); NULL = WASTE_CPUS,
 *                else the OS decides. Validated by waste_open — a loader
 *                called directly ignores one it cannot parse or honour.
 *   policy       waste_cache_policy: 0 = LFRU, 1 = LRU
 *   direct_io    ask for the page-cache bypass on the expert banks
 */
typedef struct {
    size_t cache_bytes;
    int    want_vision;
    int    n_threads;
    const char *cpus;
    int    policy;
    int    direct_io;
} waste_load_opts;

int  waste_model_load(waste_model *m, const char *dir, int kv_cap,
                      const waste_load_opts *opt);
void waste_model_free(waste_model *m);
/* Runs one token; returns logits (vocab floats, owned by the model), or
 * NULL when an expert record failed to read or failed verification —
 * waste_model_read_error then says which one and why.
 * `pos` is the position in the sequence (0-based). */
const float *waste_model_step(waste_model *m, int token, int pos, int *routed);

/* Why the last read failed, and where. NULL when nothing has. The string
 * is static; `layer` and `expert` name the record. Sticky, so a caller
 * checks it once per call rather than per expert. */
void        waste_model_reset(waste_model *m);
int         waste_model_resize_cache(waste_model *m, size_t cache_bytes);
void        waste_model_set_lookahead(int n);
int         waste_model_get_lookahead(void);
const char *waste_model_read_error(const waste_model *m, int *layer, int *expert);
/* Clears both sticky per-call flags: the record error and the context
 * one. Called to arm a fresh eval or generate. */
void        waste_model_clear_read_error(waste_model *m);

/* Highest position + 1 this model can hold, or 0 when it is unbounded
 * (no MLA layer, so nothing is stored per position). A step or prefill
 * outside it returns NULL and sets the flag below rather than writing
 * past the latent KV cache — which is what it used to do, quietly, on
 * any chat long enough or any prompt longer than ctx_tokens. */
int waste_model_ctx_max(const waste_model *m);
int waste_model_ctx_full(const waste_model *m);

/* Prefill a chunk of `n` tokens starting at `pos0`, returning the logits of
 * the last one. Equivalent to n successive waste_model_step calls, but the
 * projections become GEMMs and — the part that matters for a streaming
 * engine — each distinct expert the chunk routes to is read from disk once
 * instead of once per token. */
const float *waste_model_prefill(waste_model *m, const int *tokens, int n,
                                 int pos0);
/* Prefill chunk size. Declared here because waste_plan_memory has to size
 * the chunk scratch into the RAM budget, and that must be the same number
 * the engine actually allocates. */
#define WASTE_CHUNK_MAX 64
int waste_model_chunk_max(const waste_model *m);

/* Where the MoE router parks its scores inside m->att.
 *
 * That buffer has four users — MLA's per-head score rows, KDA's per-head
 * delta scratch, the router's two score arrays at this offset, and
 * AttnRes's handful — and they take turns rather than overlapping in
 * time. The offset is past anything the others use on a real container,
 * but "on a real container" is not a bound: `--ctx 64` makes the
 * attention part 1088 floats and the router then wrote at 4096. Declared
 * here so the allocation is sized from the same number the code indexes
 * with, and so waste_plan_memory counts the buffer that is really
 * allocated. */
#define WASTE_ATT_ROUTER_OFF 4096

/* Session state: KDA recurrent state + short-conv rings + MLA KV + the
 * AttnRes history. Saving it turns a cold re-prefill into a file read,
 * which at streaming speeds is minutes versus milliseconds. */
/* Learned hotlist: which experts this workload uses, so the next run does
 * not start with an empty cache. `path` is the file itself, so a caller
 * can keep one per workload rather than one per container. */
int waste_model_warm_cache(waste_model *m, const char *path);
int waste_model_save_usage(const waste_model *m, const char *path);

int waste_model_state_save(const waste_model *m, const char *path, int pos);
int waste_model_state_load(waste_model *m, const char *path, int *pos);
const waste_tensor *waste_find(const waste_model *m, const char *name);

/* Internal primitives the vision tower reuses: same numerics, same
 * threading, no second implementation to keep in step. */
void waste_rmsnorm(float *o, const float *x, const float *w, int n, float eps);
void waste_matmul_t(waste_model *m, float *Y, const waste_tensor *t,
                    const float *X, int out, int in, int T);
void waste_deq_row(const waste_tensor *t, long r, int cols, float *dst);

/* One token's embedding row into dst (hidden floats), dequantizing if the
 * table was left quantized. */
int waste_embed_row(waste_model *m, int token, float *dst);

/* Vision: encodes one image's patches into text-embedding space.
 * `pixels` is [h*w][3*14*14] already patchified and normalized, the result
 * [h/2 * w/2][hidden]. Returns 0, or -1 when the container has no vision
 * tower or it was not loaded. */
int waste_vision_encode(waste_model *m, const float *pixels, int h, int w,
                        float *out);
int waste_vision_available(const waste_model *m);

/* Decodes an image and lays it out as [gh*gw][3*14*14], normalized. The
 * grid is chosen to keep the aspect ratio under `max_patches` and to be
 * even on both axes, because the tower merges 2x2. Caller frees. */
/* Source dimensions without decoding the pixels. K3 writes them into the
 * media block as text, so the wrapper needs them before the encode. */
int waste_image_size(const char *path, int *w, int *h);

float *waste_image_load(const char *path, int max_patches,
                        const float *mean, const float *std,
                        int *out_h, int *out_w);

/* Exposed for unit tests (tests/test_k3parts.c) — these are the pieces of
 * K3 whose maths is new, so they are checked against the reference
 * implementation directly rather than only end to end. */
float waste_situ_pair(float gate, float up, float beta, float linear_beta);
void  waste_kda_decay_gate(float *g, const float *A_log, const float *dt_bias,
                           int H, int D, float lower_bound);
void  waste_kda_decay_gate_ex(float *g, const float *A_log, const float *dt_bias,
                              int H, int D, float lower_bound, int per_channel);
void  waste_apply_attn_res(waste_model *m, const float *blockres, int nb,
                           const float *prefix_sum, const float *norm_w,
                           const float *proj_w, float *out);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_MODEL_H */
