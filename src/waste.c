/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * waste.c — the public API, implemented over model.c.
 *
 * Everything the CLI can do goes through here; the CLI links this and
 * nothing private. Errors are returned, never printed, and nothing calls
 * exit().
 */

#include "waste.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <time.h>
#if !defined(_WIN32)
#include <fcntl.h>
#include <pthread.h>
#include <sys/file.h>
#include <sys/stat.h>
#endif

#include "json.h"
#include "memory.h"
#include "model.h"
#include "platform.h"
#include "tokenizer.h"
#include "waste_backend.h"

typedef struct waste_model_lock waste_model_lock;

struct waste_ctx {
    waste_model m;
    waste_tok *tok;
    waste_cfg cfg;
    waste_memplan plan;
    char path[512];
    char usage[512];         /* resolved hotlist path; cfg's is borrowed */
    int pos;                 /* next position in the sequence */
    int warmed;              /* experts preloaded from the hotlist */
    char quant[64];          /* composed at open, reported by get_info */
    char detail[128];        /* which record failed, for waste_error_detail */
    waste_stats stats;
    waste_model_lock *model_lock;

    /* Queued image embeddings, concatenated: img_each[] is how many rows
     * each queued image contributed, which is what expand needs to know
     * to size each placeholder's run. */
    float   *img;
    size_t   img_rows;
    size_t   img_each[WASTE_MAX_IMAGES];
    int      img_n;
};

#if !defined(_WIN32)
struct waste_model_lock {
    dev_t dev;
    ino_t ino;
    int fd;
    unsigned refs;
    pid_t owner;
    waste_model_lock *next;
};

static pthread_mutex_t model_lock_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t model_lock_once = PTHREAD_ONCE_INIT;
static waste_model_lock *model_locks;

/* A forked child is a competing process, not another context in its parent.
 * Close its inherited copies before it can consult the copied registry. The
 * entries themselves remain allocated in the child: free is not async-signal
 * safe, and inherited contexts ignore entries owned by a different pid. */
static void model_lock_atfork_prepare(void)
{
    pthread_mutex_lock(&model_lock_mu);
}

static void model_lock_atfork_parent(void)
{
    pthread_mutex_unlock(&model_lock_mu);
}

static void model_lock_atfork_child(void)
{
    for (waste_model_lock *p = model_locks; p; p = p->next) close(p->fd);
    model_locks = NULL;
    pthread_mutex_unlock(&model_lock_mu);
}

static void model_lock_init(void)
{
    (void)pthread_atfork(model_lock_atfork_prepare, model_lock_atfork_parent,
                         model_lock_atfork_child);
}

/* One OS lock per container and process. Device/inode identity means aliases
 * of the same directory share an entry. The registry supplies the reference
 * semantics flock does not: closing one context must not release ownership
 * while another context in this process still uses the container. */
static waste_model_lock *model_lock_acquire(const char *path, int allow,
                                            waste_status *status)
{
    *status = WASTE_OK;
    if (allow) return NULL;

    pthread_once(&model_lock_once, model_lock_init);
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    const int fd = open(path, flags);
    if (fd < 0) { *status = WASTE_E_IO; return NULL; }
#ifndef O_CLOEXEC
    const int fdflags = fcntl(fd, F_GETFD);
    if (fdflags < 0 || fcntl(fd, F_SETFD, fdflags | FD_CLOEXEC)) {
        close(fd);
        *status = WASTE_E_IO;
        return NULL;
    }
#endif
    struct stat st;
    if (fstat(fd, &st)) { close(fd); *status = WASTE_E_IO; return NULL; }

    pthread_mutex_lock(&model_lock_mu);
    for (waste_model_lock *p = model_locks; p; p = p->next) {
        if (p->dev == st.st_dev && p->ino == st.st_ino) {
            p->refs++;
            pthread_mutex_unlock(&model_lock_mu);
            close(fd);
            return p;
        }
    }

    int rc;
    do rc = flock(fd, LOCK_EX | LOCK_NB); while (rc && errno == EINTR);
    if (rc) {
        const int busy = errno == EWOULDBLOCK || errno == EAGAIN;
        pthread_mutex_unlock(&model_lock_mu);
        close(fd);
        *status = busy ? WASTE_E_BUSY : WASTE_E_IO;
        return NULL;
    }

    waste_model_lock *p = (waste_model_lock *)calloc(1, sizeof *p);
    if (!p) {
        (void)flock(fd, LOCK_UN);
        pthread_mutex_unlock(&model_lock_mu);
        close(fd);
        *status = WASTE_E_OOM;
        return NULL;
    }
    p->dev = st.st_dev;
    p->ino = st.st_ino;
    p->fd = fd;
    p->refs = 1;
    p->owner = getpid();
    p->next = model_locks;
    model_locks = p;
    pthread_mutex_unlock(&model_lock_mu);
    return p;
}

static void model_lock_release(waste_model_lock *entry)
{
    if (!entry || entry->owner != getpid()) return;
    pthread_mutex_lock(&model_lock_mu);
    if (--entry->refs == 0) {
        waste_model_lock **pp = &model_locks;
        while (*pp && *pp != entry) pp = &(*pp)->next;
        if (*pp) *pp = entry->next;
        (void)flock(entry->fd, LOCK_UN);
        close(entry->fd);
        free(entry);
    }
    pthread_mutex_unlock(&model_lock_mu);
}
#else
/* Keep non-POSIX lifecycle behavior unchanged. The public opt-out is ignored
 * on hosts where this advisory ownership lock is not implemented. */
struct waste_model_lock { int unused; };
static waste_model_lock *model_lock_acquire(const char *path, int allow,
                                            waste_status *status)
{
    (void)path;
    (void)allow;
    *status = WASTE_OK;
    return NULL;
}
static void model_lock_release(waste_model_lock *entry) { (void)entry; }
#endif



/* What the container is actually stored as, composed once at open. It used
 * to be one string constant, which was true of the first model converted
 * and wrong about the second: K3's trunk is mostly Q4G where Kimi-Linear's
 * is Q8G. Narrowest quantization first, f32 last. */
static void quant_summary(waste_ctx *c)
{
    static const struct { int fmt; const char *name; } tbl[] = {
        { 7, "Q3G" }, { 3, "Q4G" }, { 2, "Q8G" }, { 1, "F16" }, { 0, "F32" },
    };
    int n = snprintf(c->quant, sizeof c->quant, "experts VQ%dR, trunk",
                     c->m.stages);
    size_t off = (n > 0 && (size_t)n < sizeof c->quant) ? (size_t)n : 0;
    const char *sep = " ";
    for (size_t i = 0; i < sizeof tbl / sizeof *tbl; i++) {
        if (!(c->m.trunk_fmts & (1u << tbl[i].fmt))) continue;
        n = snprintf(c->quant + off, sizeof c->quant - off, "%s%s",
                     sep, tbl[i].name);
        if (n < 0 || (size_t)n >= sizeof c->quant - off) break;
        off += (size_t)n;
        sep = "/";
    }
}

const char *waste_strerror(waste_status s)
{
    switch (s) {
    case WASTE_OK:             return "ok";
    case WASTE_E_IO:           return "I/O error";
    case WASTE_E_FORMAT:       return "malformed container";
    case WASTE_E_RAM_BUDGET:   return "RAM budget below the model's floor";
    case WASTE_E_OOM:          return "out of memory";
    case WASTE_E_ARG:          return "invalid argument";
    case WASTE_E_UNSUPPORTED:  return "unsupported";
    case WASTE_E_CANCELLED:    return "cancelled by callback";
    case WASTE_E_BUSY:         return "container is already open in another process";
    }
    return "unknown error";
}

void waste_cfg_init(waste_cfg *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof *cfg);
    cfg->cache_policy = WASTE_CACHE_LFRU;
    cfg->use_direct_io = 1;
    cfg->ctx_tokens = 4096;
}

void waste_gen_params_init(waste_gen_params *p)
{
    if (!p) return;
    memset(p, 0, sizeof *p);
    p->temperature = 0.0f;      /* greedy */
    p->top_p = 1.0f;
    p->max_tokens = 256;
}

static double nowf(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

/* ---- memory planning ---------------------------------------------------- */

static char *slurp_all(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    /* -1 on a directory or a pipe; see slurp() in model.c */
    if (n < 0) { fclose(f); return NULL; }
    char *b = (char *)malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    b[n] = 0;
    fclose(f);
    return b;
}

/* Physical RAM, or 0 when it cannot be determined. */
uint64_t waste_physical_ram(void)
{
#if defined(__APPLE__)
    uint64_t v = 0;
    size_t n = sizeof v;
    if (sysctlbyname("hw.memsize", &v, &n, NULL, 0) == 0) return v;
    return 0;
#elif defined(_WIN32)
    return waste_physical_ram_bytes();
#elif defined(__linux__)
    const long p = sysconf(_SC_PHYS_PAGES), z = sysconf(_SC_PAGESIZE);
    return (p > 0 && z > 0) ? (uint64_t)p * (uint64_t)z : 0;
#else
    return 0;
#endif
}

/* What the budget is allowed to size against. Physical RAM is how big the
 * machine is; this is how much of it this process may have, and on Linux
 * the two differ by the whole ratio between a host and a container limit —
 * sysconf reads host MemTotal from inside a cgroup. See src/memory.c. */
uint64_t waste_usable_ram(void)
{
    const uint64_t phys = waste_physical_ram();
    const uint64_t cg = waste_cgroup_limit("/proc/self/cgroup", "/sys/fs/cgroup");
    if (!cg) return phys;
    return (!phys || cg < phys) ? cg : phys;
}

waste_status waste_plan_memory(const char *model_path, uint32_t ctx_tokens,
                               waste_memplan *out)
{
    if (!model_path || !out) return WASTE_E_ARG;
    char p[512];
    snprintf(p, sizeof p, "%s/manifest.json", model_path);
    char *src = slurp_all(p);
    if (!src) return WASTE_E_IO;
    js_doc d;
    if (js_parse(&d, src) < 0) { free(src); return WASTE_E_FORMAT; }

    memset(out, 0, sizeof *out);

    /* trunk: sum the tensor payloads as they are stored, minus the ones the
     * engine deliberately leaves on disk (embed_tokens — one row per token) */
    const int trunk = js_get(&d, 0, "trunk");
    char prefix[64];
    js_str(&d, js_get(&d, 0, "tensor_prefix"), prefix, sizeof prefix);
    for (int i = 0; i < js_size(&d, trunk); i++) {
        const int e = js_at(&d, trunk, i);
        char nm[160];
        js_str(&d, js_get(&d, e, "name"), nm, sizeof nm);
        const int fmt = (int)js_int(&d, js_get(&d, e, "fmt"), 0);
        if (fmt != 0 && strstr(nm, "embed_tokens.weight")) continue;
        const uint64_t nb = (uint64_t)js_int(&d, js_get(&d, e, "bytes"), 0);
        /* The vision tower and projector sit outside tensor_prefix. They
         * are loaded only when a caller asks for images, so they are
         * counted apart and folded in by waste_open — counting them here
         * would overstate the floor for every text-only run, and leaving
         * them out entirely understated it for every run with one. */
        if (prefix[0] && strncmp(nm, prefix, strlen(prefix)) != 0) {
            out->vision_bytes += nb;
            continue;
        }
        out->trunk_bytes += nb;
    }

    /* `vision_bytes` includes both optional weights and the peak memory of
     * decoding/encoding images.  The latter used to be invisible to the
     * advertised hard ceiling: stb decoded arbitrary source dimensions,
     * the tower allocated its full activation set, and up to 32 projected
     * images remained queued. */
    if (out->vision_bytes) {
        int vh = 1024, heads = 12, qkv = 1536, vi = 4096;
        int ph = 64, pw = 64, th = 0, mp = 1024;
        snprintf(p, sizeof p, "%s/vision.json", model_path);
        char *vs = slurp_all(p);
        if (vs) {
            js_doc vd;
            if (js_parse(&vd, vs) >= 0) {
                vh = (int)js_int(&vd, js_get(&vd, 0, "vt_hidden_size"), vh);
                heads = (int)js_int(&vd, js_get(&vd, 0, "vt_num_attention_heads"), heads);
                qkv = (int)js_int(&vd, js_get(&vd, 0, "qkv_hidden_size"), qkv);
                vi = (int)js_int(&vd, js_get(&vd, 0, "vt_intermediate_size"), vi);
                ph = (int)js_int(&vd, js_get(&vd, 0, "init_pos_emb_height"), ph);
                pw = (int)js_int(&vd, js_get(&vd, 0, "init_pos_emb_width"), pw);
                th = (int)js_int(&vd, js_get(&vd, 0, "text_hidden_size"), 0);
                mp = (int)js_int(&vd, js_get(&vd, 0, "max_patches"), mp);
                js_free(&vd);
            }
            free(vs);
        }
        const int cfg_i = js_get(&d, 0, "config");
        if (!th) th = (int)js_int(&d, js_get(&d, cfg_i, "hidden_size"), 0);
        if (vh > 0 && vh <= (1 << 20) && heads > 0 && heads <= (1 << 16) &&
            qkv > 0 && qkv <= (1 << 20) && vi > 0 && vi <= (1 << 22) &&
            ph > 0 && ph <= (1 << 16) && pw > 0 && pw <= (1 << 16) &&
            th > 0 && th <= (1 << 20) && mp >= 4 && mp <= (1 << 20)) {
            const uint64_t L = (uint64_t)mp;
            const uint64_t queue = (uint64_t)WASTE_MAX_IMAGES *
                                   ((L + 3) / 4) * (uint64_t)th * 4;
            const uint64_t pixels = L * 3u * 14u * 14u * 4u;
            const uint64_t tower =
                (3u * L * (uint64_t)vh +
                 3u * L * (uint64_t)qkv + L * (uint64_t)qkv +
                 L * (uint64_t)vi + L +
                 (uint64_t)ph * (uint64_t)pw * (uint64_t)vh +
                 2u * L * (uint64_t)vh) * 4u;
            const uint64_t decode = (uint64_t)WASTE_MAX_SOURCE_PIXELS * 3u;
            /* realloc may temporarily retain the previous queue while the
             * enlarged one is allocated, so reserve two full queues. */
            out->vision_bytes += 2u * queue + pixels + tower + decode;
        }
    }

    const int cfg = js_get(&d, 0, "config");
    const int layers = (int)js_int(&d, js_get(&d, cfg, "num_hidden_layers"), 0);
    const int hidden = (int)js_int(&d, js_get(&d, cfg, "hidden_size"), 0);
    const int nheads = (int)js_int(&d, js_get(&d, cfg, "num_attention_heads"), 0);
    const int kv_lora = (int)js_int(&d, js_get(&d, cfg, "kv_lora_rank"), 0);
    const int qk_rope = (int)js_int(&d, js_get(&d, cfg, "qk_rope_head_dim"), 0);
    const int qk_nope = (int)js_int(&d, js_get(&d, cfg, "qk_nope_head_dim"), 0);
    const int v_head = (int)js_int(&d, js_get(&d, cfg, "v_head_dim"), 0);
    const int lac = js_get(&d, cfg, "linear_attn_config");
    const int kh = (int)js_int(&d, js_get(&d, lac, "num_heads"), 0);
    const int kd = (int)js_int(&d, js_get(&d, lac, "head_dim"), 0);
    const int ck = (int)js_int(&d, js_get(&d, lac, "short_conv_kernel_size"), 4);
    const int kl = js_get(&d, lac, "kda_layers");
    const int n_kda = js_size(&d, kl);
    const int n_mla = layers - n_kda;

    /* MLA caches the latent plus the rope dims, not the expanded per-head
     * K and V — kv_b_proj is absorbed into the query and the output. That
     * is 576 floats per token per layer here rather than 30720. */
    out->state_bytes = (uint64_t)n_kda * kh * kd * kd * 4                /* S */
                     + (uint64_t)n_kda * 3 * (ck - 1) * kh * kd * 4      /* conv */
                     + (uint64_t)n_mla * ctx_tokens *
                       ((uint64_t)kv_lora + qk_rope) * 4;                /* KV */
    (void)qk_nope; (void)v_head;

    /* Scratch, counted rather than guessed. The old flat 64 MB was out by
     * 4x on the decode buffers alone (e_gate/e_up/e_down are 252 MB on K3)
     * and ignored the chunked-prefill buffers entirely, which is why peak
     * RSS ran over the budget near the floor. */
    const int moe_inter = (int)js_int(&d, js_get(&d, cfg, "moe_intermediate_size"), 0);
    const int dense_inter = (int)js_int(&d, js_get(&d, cfg, "intermediate_size"), moe_inter);
    const int vocab = (int)js_int(&d, js_get(&d, cfg, "vocab_size"), 0);
    const int lat = (int)js_int(&d, js_get(&d, cfg, "routed_expert_hidden_size"), hidden);
    const int n_shared = (int)js_int(&d, js_get(&d, cfg, "num_shared_experts"), 1);
    const int n_shared_eff = n_shared ? n_shared : 1;
    const int ares = (int)js_int(&d, js_get(&d, cfg, "attn_res_block_size"), 0);
    const int eq = js_get(&d, 0, "expert_quant");
    const int stages = (int)js_int(&d, js_get(&d, eq, "stages"), 3);
    const int vec_dim = (int)js_int(&d, js_get(&d, eq, "vec_dim"), 8);
    const int entries = (int)js_int(&d, js_get(&d, eq, "entries"), 256);
    if (stages < 1 || stages > 8 || vec_dim < 1 || vec_dim > 64 ||
        entries < 1 || entries > 256) {
        js_free(&d); free(src); return WASTE_E_FORMAT;
    }
    const int nb = ares ? layers / ares + 2 : 1;
    const int big = hidden > kh * kd ? hidden : kh * kd;
    const int wide = hidden > lat ? hidden : lat;
    int lut_wide = wide > moe_inter ? wide : moe_inter;
    const int T = WASTE_CHUNK_MAX;

    uint64_t sc = 0;
    sc += (uint64_t)3 * moe_inter * hidden * 4;             /* expert staging  */
    sc += (uint64_t)2 * (dense_inter > moe_inter ? dense_inter : moe_inter) * 4;
    {   /* m->att, and it holds more than attention rows — the same max the
         * allocation in waste_model_load takes, so the plan reports the
         * buffer that is really allocated rather than the smallest of its
         * four users. */
        const int n_exp = (int)js_int(&d, js_get(&d, cfg, "num_experts"), 0);
        uint64_t att = (uint64_t)ctx_tokens * (uint64_t)nheads;
        const uint64_t kda = (uint64_t)kh * (uint64_t)kd;
        const uint64_t route = WASTE_ATT_ROUTER_OFF + 2ull * (uint64_t)n_exp;
        if (kda > att) att = kda;
        if (route > att) att = route;
        sc += (att + 1024) * 4;
    }
    sc += (uint64_t)vocab * 4;                              /* logits          */
    sc += ((uint64_t)8 * big + 8 * moe_inter + 8 * dense_inter + 512) * 4;
    sc += (uint64_t)3 * nheads * (kv_lora ? kv_lora : 1) * 4;   /* MLA absorb  */
    sc += (uint64_t)(nb + 4) * hidden * 4;                  /* AttnRes buffers */
    /* chunked prefill, allocated on first use and never freed */
    sc += (uint64_t)T * hidden * 4 * 3;                     /* cx/cnorm/cresid */
    {   /* Decode keeps three LUTs and chunked prefill keeps 2*T+1.
         * Both allocations use the container's VQ geometry. */
        const uint64_t lut = (uint64_t)(lut_wide / vec_dim + 1) *
                             (uint64_t)stages * (uint64_t)entries;
        sc += ((uint64_t)(2 * T + 1) * lut + 64) * 4;       /* m->cq  */
        sc += 3 * lut * 4;                                  /* m->lut */
        /* The int8 shadows of both, m->lut8 and m->cq8, allocated only for
         * VQ4P and counted for every container anyway: a quarter of tables
         * already counted above, against a cache measured in tens of
         * gigabytes. A plan that under-counts what a load then allocates is
         * the failure this arithmetic exists to prevent. */
        const uint64_t nsc = lut / ((uint64_t)stages * entries)
                             / WASTE_VQ_LUT_BLK + 2;
        sc += 3 * lut + 3 * nsc * 4;                        /* m->lut8 */
        sc += (uint64_t)(2 * T + 1) * (lut + nsc * 4) + 64; /* m->cq8  */
        /* Per-expert scratch for the expert-parallel MoE path: one gate,
         * up, accumulator and down LUT per routed expert, because k threads
         * each run a whole expert. Read here rather than at the bottom of
         * this function, where top_k is fetched for the cache floor. */
        const uint64_t k = (uint64_t)js_int(&d, js_get(&d, cfg,
                                            "num_experts_per_token"), 8);
        sc += k * ((uint64_t)2 * moe_inter + lat) * 4;      /* xga/xub/xacc */
        sc += k * lut * 4;                                  /* m->xlut  */
        sc += k * (lut + nsc * 4);                          /* xlut8/xqs */
    }
    sc += ((uint64_t)T * (2 * moe_inter * n_shared_eff + hidden) + 64) * 4;
    sc += (uint64_t)T * (2 * lat + 2 * hidden) * 4;
    sc += (uint64_t)2 * T * dense_inter * 4;
    sc += (uint64_t)3 * moe_inter * lat * 4;                /* one expert      */
    sc += (uint64_t)T * nb * hidden * 4 + (uint64_t)T * hidden * 4;
    sc += (uint64_t)T * 64 * 12;    /* croute + crw + cused */
    out->scratch_bytes = sc;

    /* one layer's top-k experts, double buffered */
    const int top_k = (int)js_int(&d, js_get(&d, cfg, "num_experts_per_token"), 8);
    const int lyr = js_get(&d, 0, "layers");
    uint64_t rec = 0;
    if (js_size(&d, lyr) > 0) {
        const int first = lyr + 2;   /* first member's value */
        const uint64_t bytes = (uint64_t)js_int(&d, js_get(&d, first, "bytes"), 0);
        const uint64_t n = (uint64_t)js_int(&d, js_get(&d, first, "experts"), 1);
        rec = n ? bytes / n : 0;
    }
    out->min_expert_cache = rec * (uint64_t)top_k * 2;
    out->floor_bytes = out->trunk_bytes + out->state_bytes +
                       out->scratch_bytes + out->min_expert_cache;
    /* A cache below one token's working set keeps nothing alive to the
     * next token (Gate 5), so "recommended" starts at 3x that. */
    out->recommended_bytes = out->floor_bytes +
                             rec * (uint64_t)top_k * (uint64_t)layers * 3;

    js_free(&d);
    free(src);
    return WASTE_OK;
}

/* ---- lifecycle ---------------------------------------------------------- */

waste_status waste_open(const char *model_path, const waste_cfg *cfg_in,
                        waste_ctx **out)
{
    if (!model_path || !out) return WASTE_E_ARG;
    *out = NULL;

    waste_cfg cfg;
    if (cfg_in) cfg = *cfg_in;
    else waste_cfg_init(&cfg);
    if (!cfg.ctx_tokens) cfg.ctx_tokens = 4096;

    /* Before anything is allocated, and refused rather than ignored: a
     * cpuset the engine cannot honour has to be an error, because a run
     * that silently did not pin looks exactly like pinning that did not
     * help. The loader resolves the same string again, through the same
     * function, and by then it can only succeed. */
    {
        waste_cpumask cpus;
        switch (waste_cpus_resolve(cfg.cpu_list, &cpus)) {
        case WASTE_CPUS_BAD:         return WASTE_E_ARG;
        case WASTE_CPUS_UNSUPPORTED: return WASTE_E_UNSUPPORTED;
        default: break;
        }
    }

    waste_ctx *c = (waste_ctx *)calloc(1, sizeof *c);
    if (!c) return WASTE_E_OOM;
    c->cfg = cfg;
    snprintf(c->path, sizeof c->path, "%s", model_path);

    waste_status lock_status = WASTE_OK;
    c->model_lock = model_lock_acquire(model_path, cfg.allow_concurrent_open,
                                       &lock_status);
    if (lock_status != WASTE_OK) { free(c); return lock_status; }

    waste_status st = waste_plan_memory(model_path, cfg.ctx_tokens, &c->plan);
    if (st != WASTE_OK) {
        model_lock_release(c->model_lock);
        free(c);
        return st;
    }

    /* Optional vision weights, decode buffers, tower activations and queued
     * embeddings are real memory, so all of them enter the floor. */
    if (cfg.vision && c->plan.vision_bytes) {
        c->plan.scratch_bytes += c->plan.vision_bytes;
        c->plan.floor_bytes += c->plan.vision_bytes;
        c->plan.recommended_bytes += c->plan.vision_bytes;
    }

    /* Resolve the budget. A budget the caller set is a contract and is used
     * as given. A zero means "you decide", and that decision has to know the
     * machine: recommended_bytes comes from the model alone — 80.63 GB on K3
     * — so defaulting to it on a 64 GB laptop would ask for 51 GB of expert
     * cache and swap, which is the one thing the budget exists to prevent.
     *
     * What it must NOT do is then take every byte up to the cap. Expert
     * cache is only worth anything in whole multiples of one token's
     * working set: below one multiple it keeps nothing alive between
     * tokens (Gate 5), and the fractional remainder above a multiple buys
     * a few points of hit rate while pushing the machine towards paging —
     * where a "hit" costs a page fault and throughput collapses by 8x
     * (docs/LEARNED.md §16). Filling the cap is how the default landed a
     * 27.32 GB cache on this laptop, between two budgets measured at
     * 0.11 and 0.04 tok/s, when 17.32 GB runs at 0.32.
     *
     * So step down a whole working set at a time and take the largest
     * that fits. On 64 GB K3 gets floor + 1x = 46 GB, the measured
     * optimum; on 128 GB it still gets the full floor + 3x; a model whose
     * recommendation already fits, like Kimi-Linear, is unaffected. When
     * not even one multiple fits, run at the floor and say so below.
     *
     * What "the machine" means is waste_usable_ram, not physical RAM: in a
     * cgroup those differ by the ratio between the host and the limit, and
     * sizing against the host there is not a slow run but a killed one. */
    const uint64_t phys = waste_usable_ram();
    const uint64_t cap = phys ? phys - phys / 8 : 0;   /* 12% left to the OS */
    uint64_t budget = cfg.ram_budget_bytes;

    if (!budget) {
        /* exact: recommended_bytes is floor + 3 * working_set by construction */
        const uint64_t ws = (c->plan.recommended_bytes - c->plan.floor_bytes) / 3;
        budget = c->plan.floor_bytes;
        for (int k = 3; k >= 1; k--) {
            const uint64_t b = c->plan.floor_bytes + ws * (uint64_t)k;
            if (!cap || b <= cap) { budget = b; break; }
        }
    }
    if (budget < c->plan.floor_bytes) {
        model_lock_release(c->model_lock);
        free(c);
        return WASTE_E_RAM_BUDGET;
    }

    /* A budget close to physical RAM backfires: the OS starts paging out
     * the engine's own expert cache, and a "hit" then costs a page fault
     * instead of the disk read the engine was managing. Measured on K3:
     * 29.1 GB of cache on a 64 GB machine ran at 0.04 tok/s against 0.32
     * with 28.0 GB — a better hit rate and eight times slower. This tests
     * the resolved budget, not the caller's field: the default used to skip
     * the check entirely by being zero here, which is exactly the case that
     * needed it. What survives the clamp above is a floor that does not fit
     * this machine — worth saying out loud before it crawls. */
    if (cap && budget > cap)
        fprintf(stderr,
                "waste: budget %.1f GB leaves under 12%% of the %.1f GB this "
                "process may use\n       the OS will page out the expert cache "
                "and throughput collapses\n",
                budget / 1073741824.0, phys / 1073741824.0);

    c->cfg.ram_budget_bytes = budget;   /* what we actually run under */

    const uint64_t cache_bytes = budget - c->plan.floor_bytes +
                                 c->plan.min_expert_cache;
    {
        waste_load_opts opt;
        memset(&opt, 0, sizeof opt);
        opt.cache_bytes = (size_t)cache_bytes;
        opt.want_vision = cfg.vision;
        opt.n_threads = cfg.n_threads;
        opt.cpus = cfg.cpu_list;
        opt.policy = (int)cfg.cache_policy;
        opt.direct_io = cfg.use_direct_io;
        const int rc = waste_model_load(&c->m, model_path, (int)cfg.ctx_tokens,
                                        &opt);
        if (rc) {
            /* The load allocates tensors, banks and the cache before it can
             * fail, and freeing only the context left all of it behind —
             * on K3 that is tens of gigabytes lost to one bad manifest. */
            waste_model_free(&c->m);
            model_lock_release(c->model_lock);
            free(c);
            return rc == -2 ? WASTE_E_FORMAT : WASTE_E_IO;
        }
    }
    /* Before the warm, which reads records too. The load already applied
     * WASTE_VERIFY; this is the other way in, and it can only add. */
    if (cfg.verify_records) c->m.verify = 1;
    quant_summary(c);
    /* Resolved once: cfg.usage_path is borrowed from the caller and only
     * has to outlive waste_open, so the context keeps its own copy. */
    if (cfg.usage_path && cfg.usage_path[0])
        snprintf(c->usage, sizeof c->usage, "%s", cfg.usage_path);
    else
        snprintf(c->usage, sizeof c->usage, "%s/usage.waste", model_path);
    c->cfg.usage_path = c->usage;
    c->tok = waste_tok_open(model_path);      /* optional */
    if (c->tok) waste_tok_set_eos(c->tok, c->m.cfg.eos_token_id);
    /* warm the cache from what previous runs learned, if anything */
    c->warmed = waste_model_warm_cache(&c->m, c->usage);
    *out = c;
    return WASTE_OK;
}

void waste_close(waste_ctx *c)
{
    if (!c) return;
    waste_model_free(&c->m);
    waste_tok_free(c->tok);
    free(c->img);
    model_lock_release(c->model_lock);
    free(c);
}

waste_status waste_memory_used(const waste_ctx *c, waste_memplan *out)
{
    if (!c || !out) return WASTE_E_ARG;
    *out = c->plan;
    out->min_expert_cache = (uint64_t)c->m.cache.n_slots * c->m.cache.rec_bytes;
    return WASTE_OK;
}

/* ---- tokenizer ---------------------------------------------------------- */

static waste_status tokenize_1(waste_ctx *c, const char *text, int add_bos,
                               int32_t *out, size_t cap, size_t *n_out,
                               int allow_special)
{
    if (!c || !text || !out || !n_out) return WASTE_E_ARG;
    if (!c->tok) return WASTE_E_UNSUPPORTED;
    size_t n = 0;
    if (add_bos && cap > 0) out[n++] = (int32_t)waste_tok_bos(c->tok);
    const int got = waste_tok_encode(c->tok, text, out + n, (int)(cap - n),
                                     allow_special);
    if (got < 0) return WASTE_E_ARG;
    *n_out = n + (size_t)got;
    return WASTE_OK;
}

waste_status waste_tokenize(waste_ctx *c, const char *text, int add_bos,
                            int32_t *out, size_t cap, size_t *n_out)
{
    return tokenize_1(c, text, add_bos, out, cap, n_out, 0);
}

waste_status waste_tokenize_markup(waste_ctx *c, const char *text, int add_bos,
                                   int32_t *out, size_t cap, size_t *n_out)
{
    return tokenize_1(c, text, add_bos, out, cap, n_out, 1);
}

waste_status waste_detokenize(waste_ctx *c, const int32_t *ids, size_t n,
                              char *out, size_t cap, size_t *n_out)
{
    if (!c || !ids || !out || !n_out) return WASTE_E_ARG;
    /* cap 0 made the capacity below -1 and then wrote the terminator at
     * out[0] — a one-byte overflow of a zero-byte buffer. */
    if (!cap) return WASTE_E_ARG;
    if (!c->tok) return WASTE_E_UNSUPPORTED;
    if (n > INT_MAX || cap > INT_MAX) return WASTE_E_ARG;
    size_t need = 0;
    for (size_t i = 0; i < n; i++) {
        const int one = waste_tok_decode_len1(c->tok, ids[i]);
        if ((size_t)one > SIZE_MAX - need) return WASTE_E_ARG;
        need += (size_t)one;
    }
    *n_out = need;
    if (need >= cap) { out[0] = 0; return WASTE_E_ARG; }
    const int got = waste_tok_decode(c->tok, ids, (int)n, out, (int)cap - 1);
    out[got] = 0;
    *n_out = (size_t)got;
    return WASTE_OK;
}

/* ---- generation --------------------------------------------------------- */

/* Token ids come from the caller and index the embedding table directly,
 * so an out-of-range one is an out-of-bounds read, not a wrong answer.
 * Nothing checked them until a synthetic container with a 256-entry
 * vocabulary was fed ids from a 163840-entry one and quietly returned
 * whatever was past the end of the table. */
static waste_status check_ids(const waste_ctx *c, const int32_t *ids, size_t n)
{
    const int32_t vocab = (int32_t)c->m.cfg.vocab;
    for (size_t i = 0; i < n; i++)
        if (ids[i] < 0 || ids[i] >= vocab) return WASTE_E_ARG;
    return WASTE_OK;
}


/* ---- images ------------------------------------------------------------- */

waste_status waste_image_dimensions(const char *path, int *w, int *h)
{
    if (!path || !w || !h) return WASTE_E_ARG;
    return waste_image_size(path, w, h) == 0 ? WASTE_OK : WASTE_E_IO;
}

void waste_image_clear(waste_ctx *c)
{
    if (!c) return;
    free(c->img);
    c->img = NULL;
    c->img_rows = 0;
    c->img_n = 0;
}

waste_status waste_image_add(waste_ctx *c, const char *path, size_t *n_out)
{
    if (!c || !path) return WASTE_E_ARG;
    if (!c->m.want_vision || !c->m.vcfg.layers) return WASTE_E_UNSUPPORTED;
    if (c->img_n >= WASTE_MAX_IMAGES) return WASTE_E_ARG;

    int gh = 0, gw = 0;
    float *px = waste_image_load(path, c->m.vcfg.max_patches,
                                 c->m.vcfg.mean, c->m.vcfg.std, &gh, &gw);
    if (!px) return WASTE_E_IO;

    const size_t rows = (size_t)(gh / 2) * (size_t)(gw / 2);
    const size_t th = (size_t)c->m.vcfg.text_hidden;
    float *nb = (float *)realloc(c->img, (c->img_rows + rows) * th * sizeof(float));
    if (!nb) { free(px); return WASTE_E_OOM; }
    c->img = nb;

    const int rc = waste_vision_encode(&c->m, px, gh, gw,
                                       c->img + c->img_rows * th);
    free(px);
    if (rc) return WASTE_E_IO;

    c->img_each[c->img_n++] = rows;
    c->img_rows += rows;
    if (n_out) *n_out = rows;
    return WASTE_OK;
}

waste_status waste_image_expand(waste_ctx *c, const int32_t *tokens, size_t n,
                                int32_t *out, size_t cap, size_t *n_out)
{
    if (!c || !tokens || !out) return WASTE_E_ARG;
    if (!c->m.want_vision || !c->m.vcfg.layers) return WASTE_E_UNSUPPORTED;
    const int32_t pad = (int32_t)c->m.vcfg.media_token;

    size_t seen = 0;
    for (size_t i = 0; i < n; i++) if (tokens[i] == pad) seen++;
    /* A mismatch is a prompt-assembly bug, and a silent one would show up
     * much later as an image the model never saw. Refuse it here. */
    if (seen != (size_t)c->img_n) return WASTE_E_ARG;

    size_t k = 0, img = 0;
    for (size_t i = 0; i < n; i++) {
        const size_t rep = (tokens[i] == pad) ? c->img_each[img++] : 1;
        if (k + rep > cap) { if (n_out) *n_out = 0; return WASTE_E_ARG; }
        for (size_t r = 0; r < rep; r++) out[k++] = tokens[i];
    }
    if (n_out) *n_out = k;
    return WASTE_OK;
}

/* The model reads embeddings from m.media as it walks the prompt, so the
 * queue has to be armed before any prefill and released after, or a
 * follow-up turn would splice the same picture in again. */
static void media_arm(waste_ctx *c)
{
    c->m.media = c->img;
    c->m.media_n = (int)c->img_rows;
    c->m.media_used = 0;
    c->m.cfg_media_token = c->m.vcfg.media_token;
}

static void media_release(waste_ctx *c)
{
    if (c->m.media_used > 0) waste_image_clear(c);
    c->m.media = NULL;
    c->m.media_n = c->m.media_used = 0;
}

/* An expert record that fails its checksum stops the pass and comes back
 * as WASTE_E_IO. The status alone is not actionable on a container with
 * 89,000 records in it, so the record that failed is composed into a
 * detail string the host can print — the engine still prints nothing
 * itself. Armed per call so a stale failure cannot be reported twice. */
static void read_error_arm(waste_ctx *c)
{
    waste_model_clear_read_error(&c->m);
    c->detail[0] = 0;
}

/* The sequence has reached the context this container was opened for.
 * Not an I/O failure and not a malformed anything: MLA stores one latent
 * per position and the cache is exactly ctx_tokens long, so position
 * ctx_tokens has nowhere to go. Said with the number, because "invalid
 * argument" on the fortieth turn of a chat is not actionable. */
static waste_status ctx_full_report(waste_ctx *c)
{
    snprintf(c->detail, sizeof c->detail,
             "context of %u tokens is full at position %d — "
             "reset the conversation or open with a larger ctx_tokens",
             c->cfg.ctx_tokens, c->pos);
    return WASTE_E_ARG;
}

/* Positions left before that happens. -1 means the model is unbounded —
 * it stores nothing per position — and never compares equal to 0. */
static long ctx_room(const waste_ctx *c)
{
    const int cm = waste_model_ctx_max(&c->m);
    if (!cm) return -1;
    return (long)cm - (long)c->pos;
}

static waste_status read_error_report(waste_ctx *c)
{
    /* Checked first: it is the one failure that is the caller's doing
     * rather than the container's, and reporting it as an I/O error
     * would send someone to re-download 900 GB over a full context. */
    if (waste_model_ctx_full(&c->m)) return ctx_full_report(c);
    int layer = 0, expert = 0;
    const char *why = waste_model_read_error(&c->m, &layer, &expert);
    /* Past the context check, the forward pass returns no logits for
     * exactly two reasons: a record it could not read or trust, or the
     * chunk scratch it could not allocate. Nothing named a record, so it
     * was the second. */
    if (!why) return WASTE_E_OOM;
    snprintf(c->detail, sizeof c->detail,
             "expert %d of layer %d: %s", expert, layer, why);
    return WASTE_E_IO;
}

const char *waste_error_detail(const waste_ctx *c)
{
    return (c && c->detail[0]) ? c->detail : NULL;
}

waste_status waste_eval(waste_ctx *c, const int32_t *tokens, size_t n,
                        const float **logits_out, size_t *vocab_out)
{
    if (!c || !tokens || !n) return WASTE_E_ARG;
    { const waste_status st = check_ids(c, tokens, n); if (st) return st; }
    /* Refused before anything is written, so a prompt that does not fit
     * leaves the conversation as it was rather than half-prefilled. */
    { const long room = ctx_room(c);
      if (room >= 0 && (size_t)room < n) { read_error_arm(c); return ctx_full_report(c); } }
    const float *lg = NULL;
    const int cmax = waste_model_chunk_max(&c->m);
    media_arm(c);
    read_error_arm(c);
    for (size_t i = 0; i < n; ) {
        int k = (int)(n - i);
        if (k > cmax) k = cmax;
        lg = (k > 1) ? waste_model_prefill(&c->m, tokens + i, k, c->pos)
                     : waste_model_step(&c->m, tokens[i], c->pos, NULL);
        c->pos += k;
        i += (size_t)k;
        if (!lg) break;
    }
    media_release(c);
    if (!lg) return read_error_report(c);
    if (logits_out) *logits_out = lg;
    if (vocab_out) *vocab_out = (size_t)c->m.cfg.vocab;
    return WASTE_OK;
}

static int argmax(const float *logits, int vocab)
{
    int best = 0;
    for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
    return best;
}

/* A candidate for the truncated paths. The id travels with the value so
 * ties break by id on every libc's qsort rather than by whatever order
 * that qsort happened to leave them in. */
typedef struct { float v; int i; } cand;

static int cand_cmp(const void *a, const void *b)
{
    const cand *x = (const cand *)a, *y = (const cand *)b;
    if (x->v > y->v) return -1;
    if (x->v < y->v) return 1;
    return (x->i > y->i) - (x->i < y->i);
}

/* One token from the distribution.
 *
 * What the caller asked to truncate decides the work. With neither top-k
 * nor top-p there is nothing to truncate: the full softmax *is* the
 * distribution and ranking it buys nothing, so that case is one pass.
 * With either, the candidates have to be ranked, and that is a sort.
 *
 * This used to be a selection sort over the top `k`, with `k` falling
 * back to the whole vocabulary whenever top-k was unset — O(vocab^2).
 * Measured on K3's 163840-entry vocabulary: 27.9 seconds per token,
 * single threaded, for `--temp 0.8` with no `--top-k`. Every OpenAI
 * client reached it, because top_k is not a field OpenAI has. */
static int sample(const float *logits, int vocab, const waste_gen_params *p,
                  uint64_t *rng)
{
    if (p->temperature <= 0.0f) return argmax(logits, vocab);

    /* Drawn once per call on every path, so a seed reproduces a run
     * whatever the truncation settings are. */
    *rng = *rng * 6364136223846793005ULL + 1442695040888963407ULL;
    const float u = (float)((*rng >> 11) * 0x1.0p-53);

    float mx = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > mx) mx = logits[i];

    const int k = (p->top_k > 0 && p->top_k < vocab) ? p->top_k : vocab;

    if (k >= vocab && !(p->top_p < 1.0f)) {          /* nothing truncated */
        float *pr = (float *)malloc((size_t)vocab * sizeof(float));
        if (!pr) return argmax(logits, vocab);
        float sum = 0;
        for (int i = 0; i < vocab; i++) {
            pr[i] = expf((logits[i] - mx) / p->temperature);
            sum += pr[i];
        }
        const float r = u * sum;
        float acc = 0;
        int pick = vocab - 1;                 /* only reachable via rounding */
        for (int i = 0; i < vocab; i++) {
            acc += pr[i];
            if (r <= acc) { pick = i; break; }
        }
        free(pr);
        return pick;
    }

    cand *cd = (cand *)malloc((size_t)vocab * sizeof *cd);
    if (!cd) return argmax(logits, vocab);
    for (int i = 0; i < vocab; i++) { cd[i].v = logits[i]; cd[i].i = i; }
    qsort(cd, (size_t)vocab, sizeof *cd, cand_cmp);

    float *pr = (float *)malloc((size_t)k * sizeof(float));
    if (!pr) { free(cd); return argmax(logits, vocab); }
    float sum = 0;
    for (int i = 0; i < k; i++) {
        pr[i] = expf((cd[i].v - mx) / p->temperature);
        sum += pr[i];
    }
    float cum = 0;
    int last = k;
    for (int i = 0; i < k; i++) {
        cum += pr[i] / sum;
        if (cum >= p->top_p) { last = i + 1; break; }
    }
    const float r = u * cum;
    float acc = 0;
    int pick = cd[0].i;
    for (int i = 0; i < last; i++) {
        acc += pr[i] / sum;
        if (r <= acc) { pick = cd[i].i; break; }
    }
    free(pr); free(cd);
    return pick;
}

waste_status waste_generate(waste_ctx *c, const int32_t *prompt, size_t n,
                            const waste_gen_params *params,
                            waste_token_cb cb, void *user)
{
    if (!c || !prompt || !n) return WASTE_E_ARG;
    { const waste_status st = check_ids(c, prompt, n); if (st) return st; }
    /* The prompt has to fit before a single expert is read for it. One
     * position is kept back so there is somewhere to put the first
     * sampled token; a prompt that fills the context exactly can be
     * evaluated but not continued. */
    { const long room = ctx_room(c);
      if (room >= 0 && (size_t)room <= n) { read_error_arm(c); return ctx_full_report(c); } }
    waste_gen_params p;
    if (params) p = *params; else waste_gen_params_init(&p);
    uint64_t rng = p.seed ? p.seed : 0x853c49e6748fea9bULL;

    const uint64_t h0 = c->m.cache.hits, m0 = c->m.cache.misses;
    const float *lg = NULL;
    double t0 = nowf();
    /* Prefill in chunks: tokens in a chunk route to overlapping expert
     * sets, so each distinct expert is read once instead of once per
     * token. Measured 2.35x fewer reads on a 16-token prompt. */
    {
        const int cmax = waste_model_chunk_max(&c->m);
        size_t i = 0;
        media_arm(c);
        read_error_arm(c);
        while (i < n) {
            int k = (int)(n - i);
            if (k > cmax) k = cmax;
            lg = (k > 1) ? waste_model_prefill(&c->m, prompt + i, k, c->pos)
                         : waste_model_step(&c->m, prompt[i], c->pos, NULL);
            c->pos += k;
            i += (size_t)k;
            if (!lg) break;
        }
        media_release(c);
    }
    c->stats.sec_total += nowf() - t0;
    if (!lg) return read_error_report(c);

    waste_status st = WASTE_OK;
    int cur = sample(lg, c->m.cfg.vocab, &p, &rng);
    for (uint32_t t = 0; t < p.max_tokens; t++) {
        /* Out of context is a reason to stop, not a reason to fail: the
         * tokens already produced are good, and this is the same kind of
         * end as max_tokens. The detail string says which one it was for
         * a host that wants to tell them apart. */
        if (ctx_room(c) == 0) { ctx_full_report(c); break; }
        const uint64_t hb = c->m.cache.hits, mb = c->m.cache.misses;
        const uint64_t bb = c->m.cache.bytes_read;
        t0 = nowf();
        lg = waste_model_step(&c->m, cur, c->pos++, NULL);
        const double dt = nowf() - t0;
        c->stats.sec_total += dt;
        c->stats.tokens_generated++;

        int stop = 0;
        for (size_t s = 0; s < p.n_stop; s++) if (p.stop_tokens[s] == cur) stop = 1;
        if (c->tok && cur == waste_tok_eos(c->tok)) stop = 1;

        if (cb) {
            waste_token_info info;
            memset(&info, 0, sizeof info);
            info.token_index = t;
            info.token = cur;
            info.experts_hit = (uint32_t)(c->m.cache.hits - hb);
            info.experts_missed = (uint32_t)(c->m.cache.misses - mb);
            info.bytes_read = c->m.cache.bytes_read - bb;
            info.ms_total = dt * 1000.0;
            char local[256], *piece = local;
            int pn = 0, need = 0;
            if (c->tok) need = waste_tok_decode_len1(c->tok, cur);
            if ((size_t)need + 1 > sizeof local) {
                piece = (char *)malloc((size_t)need + 1);
                if (!piece) return WASTE_E_OOM;
            }
            if (c->tok) pn = waste_tok_decode1(c->tok, cur, piece, need);
            piece[pn] = 0;
            const int cancelled = cb(&info, piece, user) != 0;
            if (piece != local) free(piece);
            if (cancelled) return WASTE_E_CANCELLED;
        }
        /* `cur` was sampled from the previous step and has already gone to
         * the callback; it is the token after it that has no logits. The
         * failure is reported even when this token was a stop token,
         * because a container that failed a check is worth hearing about
         * whether or not the answer happened to be finished. */
        if (!lg) { st = read_error_report(c); break; }
        if (stop) break;
        cur = sample(lg, c->m.cfg.vocab, &p, &rng);
    }
    c->stats.experts_hit += c->m.cache.hits - h0;
    c->stats.experts_missed += c->m.cache.misses - m0;
    c->stats.bytes_read = c->m.cache.bytes_read;
    return st;
}

/* ---- state & introspection ---------------------------------------------- */

void waste_state_reset(waste_ctx *c)
{
    if (!c) return;
    c->pos = 0;
    waste_model_reset(&c->m);
}

waste_status waste_state_save(waste_ctx *c, const char *path)
{
    if (!c || !path) return WASTE_E_ARG;
    return waste_model_state_save(&c->m, path, c->pos) == 0 ? WASTE_OK : WASTE_E_IO;
}

waste_status waste_state_load(waste_ctx *c, const char *path)
{
    if (!c || !path) return WASTE_E_ARG;
    int pos = 0;
    const int rc = waste_model_state_load(&c->m, path, &pos);
    if (rc == -2) return WASTE_E_FORMAT;      /* built for a different model */
    if (rc) {
        if (rc == -3) waste_state_reset(c);
        return WASTE_E_IO;
    }
    c->pos = pos;
    return WASTE_OK;
}

waste_status waste_model_get_info(const waste_ctx *c, waste_model_info *out)
{
    if (!c || !out) return WASTE_E_ARG;
    const waste_config *cf = &c->m.cfg;
    memset(out, 0, sizeof *out);
    out->n_layers = (uint32_t)cf->n_layers;
    out->n_experts = (uint32_t)cf->n_experts;
    out->top_k = (uint32_t)cf->top_k;
    out->hidden = (uint32_t)cf->hidden;
    out->ctx_max = c->cfg.ctx_tokens;
    /* An expert's matrices are as wide as its input, and in a latent MoE
     * that is not the hidden: K3 projects 7168 down to a 3584-wide latent
     * and runs the experts there, so counting them at hidden width reports
     * exactly twice the parameters the container actually stores. */
    const uint64_t ew = cf->latent_dim ? (uint64_t)cf->latent_dim
                                       : (uint64_t)cf->hidden;
    const uint64_t pe = 3ULL * ew * cf->moe_inter;
    const uint64_t moe_layers = (uint64_t)(cf->n_layers - cf->first_dense);

    /* The routed experts are the model's bulk but not the model: the trunk
     * carries attention, the shared experts, the latent projections, the
     * norms and the embeddings, and every token runs all of it. So it
     * counts in both figures — the exception being the embedding table, of
     * which a token reads one row. Tensors outside `prefix` are the vision
     * tower, which this reports on as little as it runs it. */
    uint64_t trunk = 0, trunk_active = 0;
    const size_t plen = strlen(cf->prefix);
    for (int i = 0; i < c->m.n_tensors; i++) {
        const waste_tensor *t = &c->m.t[i];
        if (plen && strncmp(t->name, cf->prefix, plen) != 0) continue;
        trunk += (uint64_t)t->n;
        if (!strstr(t->name, "embed_tokens.weight")) trunk_active += (uint64_t)t->n;
    }

    out->params_total  = pe * (uint64_t)cf->n_experts * moe_layers + trunk;
    out->params_active = pe * (uint64_t)cf->top_k * moe_layers + trunk_active;
    /* Both containers declare model_type "kimi_linear", so the name comes
     * from the architecture they were converted from. One the engine does
     * not recognize is reported verbatim rather than guessed at. */
    out->arch = strstr(cf->arch, "KimiK3")     ? "kimi-k3"
              : strstr(cf->arch, "KimiLinear") ? "kimi-linear"
              : cf->arch[0]                    ? cf->arch
                                               : "unknown";
    out->quant_summary = c->quant;
    return WASTE_OK;
}

waste_status waste_save_usage(waste_ctx *c)
{
    if (!c) return WASTE_E_ARG;
    return waste_model_save_usage(&c->m, c->usage) < 0 ? WASTE_E_IO : WASTE_OK;
}

waste_status waste_get_stats(const waste_ctx *c, waste_stats *out)
{
    if (!c || !out) return WASTE_E_ARG;
    *out = c->stats;
    out->experts_hit = c->m.cache.hits;
    out->experts_missed = c->m.cache.misses;
    out->bytes_read = c->m.cache.bytes_read;
    out->direct_io = c->m.direct_io;
    return WASTE_OK;
}
