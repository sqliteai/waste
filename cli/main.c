/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * cli/main.c — the WASTE command line.
 *
 * A client of src/waste.h with no private access: if the CLI can do it, an
 * embedding host can do it too. That rule is the point of the split.
 *
 *   waste run    MODEL "prompt"   generate once
 *   waste chat   MODEL            REPL, state kept across turns
 *   waste bench  MODEL            throughput and cache behaviour
 *   waste plan   MODEL [--budget] memory floor and what a budget buys
 *   waste info   MODEL            container and build details
 *   waste version
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#include "../src/waste.h"

#define MAXTOK 8192

static int parse_size(const char *s, uint64_t *out)
{
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (errno || end == s || !isfinite(v) || v < 0.0) return -1;
    while (isspace((unsigned char)*end)) end++;
    uint64_t mul = 1;
    if (*end == 'g' || *end == 'G') { mul = 1ULL << 30; end++; }
    else if (*end == 'm' || *end == 'M') { mul = 1ULL << 20; end++; }
    else if (*end == 'k' || *end == 'K') { mul = 1ULL << 10; end++; }
    if (*end == 'b' || *end == 'B') end++;
    while (isspace((unsigned char)*end)) end++;
    if (*end || v >= (double)UINT64_MAX / (double)mul) return -1;
    *out = (uint64_t)(v * (double)mul);
    return 0;
}

static int parse_u64(const char *s, uint64_t *out)
{
    if (!s || *s == '-') return -1;
    char *end = NULL;
    errno = 0;
    const unsigned long long v = strtoull(s, &end, 10);
    while (end && isspace((unsigned char)*end)) end++;
    if (errno || end == s || !end || *end) return -1;
    *out = (uint64_t)v;
    return 0;
}

static int parse_u32(const char *s, uint32_t *out)
{
    uint64_t v = 0;
    if (parse_u64(s, &v) || v > UINT32_MAX) return -1;
    *out = (uint32_t)v;
    return 0;
}

static int parse_nonnegative_int(const char *s, int *out)
{
    uint64_t v = 0;
    if (parse_u64(s, &v) || v > INT_MAX) return -1;
    *out = (int)v;
    return 0;
}

static int parse_float(const char *s, float *out)
{
    char *end = NULL;
    errno = 0;
    const float v = strtof(s, &end);
    while (end && isspace((unsigned char)*end)) end++;
    if (errno || end == s || !end || *end || !isfinite(v)) return -1;
    *out = v;
    return 0;
}

static void human(uint64_t b, char *out, size_t cap)
{
    const double g = (double)b / (1ULL << 30);
    if (g >= 1.0) snprintf(out, cap, "%.2f GB", g);
    else snprintf(out, cap, "%.0f MB", (double)b / (1ULL << 20));
}

/* Parameter counts here span 47 B to 2.78 T, so the unit follows the
 * number rather than printing K3 as 2777.4 B. */
static void params_human(uint64_t n, char *out, size_t cap)
{
    if (n >= 1000000000000ULL) snprintf(out, cap, "%.2f T", (double)n / 1e12);
    else snprintf(out, cap, "%.2f B", (double)n / 1e9);
}

/* ---- shared option parsing --------------------------------------------- */

#define MAX_POS 2

/* Matches the engine's own queue limit; more images than this in one
 * prompt is a batch job, not a command line. */
#define WASTE_MAX_IMAGES_CLI 32

typedef struct {
    uint64_t budget;
    uint32_t ctx, max_tokens;
    float temperature, top_p;
    int top_k, threads, quiet, learn, json, no_echo, allow_concurrent;
    int media_inlined;              /* the media block is already in the
                                       prompt string, inside the user turn */
    uint64_t seed;
    const char *cpus;               /* --cpus: cpu list for the pool */
    const char *file, *stop, *system;
    const char *image[WASTE_MAX_IMAGES_CLI];
    int n_image;
    int raw;
    int verify;                     /* --verify: check every record's crc32 */
    const char *pos[MAX_POS];      /* MODEL, then the command's argument */
    int n_pos;
} opts;

static void opts_init(opts *o)
{
    memset(o, 0, sizeof *o);
    o->ctx = 4096;
    o->max_tokens = 128;
    o->top_p = 1.0f;
}

/* Options and positionals in any order.
 *
 * The prompt used to be argv[3] unconditionally, with options only looked
 * for after it. `waste run M --temp 0 "hi"` therefore generated from the
 * string "--temp", `waste run M -n 3` generated from "-n", and a second
 * positional was dropped without a word. Silently doing the wrong thing is
 * worse than refusing, so positionals are collected here and anything
 * unexpected is an error. */
static int parse_opts(int argc, char **argv, int from, opts *o)
{
    for (int i = from; i < argc; i++) {
        const char *a = argv[i];
        int need = 0;                              /* option takes a value */
        if (!strcmp(a, "--budget") || !strcmp(a, "--ctx") || !strcmp(a, "-n") ||
            !strcmp(a, "--temp") || !strcmp(a, "--top-p") || !strcmp(a, "--top-k") ||
            !strcmp(a, "--seed") || !strcmp(a, "--threads") ||
            !strcmp(a, "--cpus") ||
            !strcmp(a, "--file") || !strcmp(a, "--stop") ||
            !strcmp(a, "--system") || !strcmp(a, "--image")) {
            need = 1;
            /* `--temp --budget 8G` used to make "--budget" the temperature
             * and then complain about 8G. A value that looks like an option
             * is a missing value. */
            if (i + 1 >= argc || (argv[i + 1][0] == '-' && argv[i + 1][1] &&
                                  !isdigit((unsigned char)argv[i + 1][1]) &&
                                  argv[i + 1][1] != '.')) {
                fprintf(stderr, "%s needs a value\n", a);
                return -1;
            }
        }
        const char *v = need ? argv[i + 1] : NULL;
        int bad = 0;
        if (!strcmp(a, "--budget")) bad = parse_size(v, &o->budget);
        else if (!strcmp(a, "--ctx")) bad = parse_u32(v, &o->ctx);
        else if (!strcmp(a, "-n")) bad = parse_u32(v, &o->max_tokens);
        else if (!strcmp(a, "--temp")) bad = parse_float(v, &o->temperature);
        else if (!strcmp(a, "--top-p")) bad = parse_float(v, &o->top_p);
        else if (!strcmp(a, "--top-k")) bad = parse_nonnegative_int(v, &o->top_k);
        else if (!strcmp(a, "--seed")) bad = parse_u64(v, &o->seed);
        else if (!strcmp(a, "--threads")) bad = parse_nonnegative_int(v, &o->threads);
        else if (!strcmp(a, "--cpus")) o->cpus = v;
        else if (!strcmp(a, "--file")) o->file = v;
        else if (!strcmp(a, "--stop")) o->stop = v;
        else if (!strcmp(a, "--system")) o->system = v;
        else if (!strcmp(a, "--image")) {
            if (o->n_image >= WASTE_MAX_IMAGES_CLI) {
                fprintf(stderr, "at most %d images\n", WASTE_MAX_IMAGES_CLI);
                return -1;
            }
            o->image[o->n_image++] = v;
        }
        else if (!strcmp(a, "-q") || !strcmp(a, "--quiet")) o->quiet = 1;
        else if (!strcmp(a, "--learn")) o->learn = 1;
        else if (!strcmp(a, "--json")) o->json = 1;
        else if (!strcmp(a, "--raw")) o->raw = 1;
        else if (!strcmp(a, "--verify")) o->verify = 1;
        else if (!strcmp(a, "--allow-concurrent-open")) o->allow_concurrent = 1;
        else if (!strcmp(a, "-")) {                /* explicit stdin */
            if (o->n_pos >= MAX_POS) { fprintf(stderr, "too many arguments\n"); return -1; }
            o->pos[o->n_pos++] = "-";
        } else if (a[0] == '-' && a[1]) {
            fprintf(stderr, "unknown option %s\n", a);
            return -1;
        } else {
            if (o->n_pos >= MAX_POS) {
                fprintf(stderr, "unexpected argument \"%s\"\n", a);
                return -1;
            }
            o->pos[o->n_pos++] = a;
        }
        if (bad) {
            fprintf(stderr, "invalid value for %s: %s\n", a, v ? v : "");
            return -1;
        }
        i += need;
    }
    /* A sampler set to nonsense produces empty output and no complaint. */
    if (o->temperature < 0.0f) { fprintf(stderr, "--temp must be >= 0\n"); return -1; }
    if (!(o->top_p > 0.0f) || o->top_p > 1.0f) { fprintf(stderr, "--top-p must be in (0, 1]\n"); return -1; }
    if (o->top_k < 0) { fprintf(stderr, "--top-k must be >= 0\n"); return -1; }
    if (o->ctx == 0) { fprintf(stderr, "--ctx must be > 0\n"); return -1; }
    if (o->max_tokens == 0) { fprintf(stderr, "-n must be > 0\n"); return -1; }
    if (o->threads < 0) { fprintf(stderr, "--threads must be >= 0\n"); return -1; }
    return 0;
}

/* The prompt: an argument, a file, or stdin — the last both when asked for
 * with "-" and when nothing else was given and stdin is not a terminal, so
 * `echo hi | waste run M` does what it obviously means. Caller frees. */
static char *read_prompt(const opts *o)
{
    FILE *f = NULL;
    const char *arg = o->n_pos > 1 ? o->pos[1] : NULL;
    if (o->file) {
        f = strcmp(o->file, "-") ? fopen(o->file, "rb") : stdin;
        if (!f) { fprintf(stderr, "cannot read %s\n", o->file); return NULL; }
    } else if (arg && strcmp(arg, "-")) {
        char *d = (char *)malloc(strlen(arg) + 1);
        if (d) memcpy(d, arg, strlen(arg) + 1);
        return d;
    } else if (arg || !isatty(fileno(stdin))) {
        f = stdin;
    } else {
        return NULL;                               /* nothing to read */
    }

    size_t cap = 4096, n = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) { if (f != stdin) fclose(f); return NULL; }
    for (;;) {
        if (n + 1 >= cap) {
            char *g = (char *)realloc(buf, cap *= 2);
            if (!g) { free(buf); if (f != stdin) fclose(f); return NULL; }
            buf = g;
        }
        const size_t got = fread(buf + n, 1, cap - n - 1, f);
        n += got;
        if (got == 0) break;
    }
    buf[n] = 0;
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
    if (f != stdin) fclose(f);
    return buf;
}

static waste_status open_model(const char *path, const opts *o, waste_ctx **ctx)
{
    waste_cfg cfg;
    waste_cfg_init(&cfg);
    cfg.ram_budget_bytes = o->budget;
    cfg.ctx_tokens = o->ctx;
    cfg.n_threads = o->threads;
    cfg.cpu_list = o->cpus;
    /* The tower is 434 MB that a text prompt would never touch, so it is
     * loaded only when there is an image to put through it. */
    cfg.vision = o->n_image > 0;
    /* Off unless asked: a pass over every record on every miss, ~5% on
     * Kimi-Linear. Worth it for a container that was copied or downloaded
     * and has not been read since. */
    cfg.verify_records = o->verify;
    cfg.allow_concurrent_open = o->allow_concurrent;
    const waste_status st = waste_open(path, &cfg, ctx);
    /* Two statuses that say nothing useful on their own when --cpus is
     * what produced them, and it usually is: nothing else here can be
     * malformed, and nothing else is refused for being unsupported on a
     * platform that otherwise runs. */
    if (o->cpus && st == WASTE_E_ARG)
        fprintf(stderr, "--cpus: not a cpu list: %s\n", o->cpus);
    else if (o->cpus && st == WASTE_E_UNSUPPORTED)
        fprintf(stderr, "--cpus: this platform does not bind threads to "
                        "CPUs (Linux and Windows only)\n");
    return st;
}

static int fail(const char *what, waste_status s)
{
    fprintf(stderr, "%s: %s\n", what, waste_strerror(s));
    if (s == WASTE_E_RAM_BUDGET)
        fprintf(stderr, "  (run `waste plan MODEL` to see the floor)\n");
    return 1;
}

/* The same, for a failure the engine can say more about than its status:
 * an expert record that did not survive the read is reported as an I/O
 * error, and which record it was is the part worth printing. */
static int fail_ctx(const char *what, waste_status s, const waste_ctx *c)
{
    const char *detail = c ? waste_error_detail(c) : NULL;
    fail(what, s);
    if (detail) {
        fprintf(stderr, "  %s\n", detail);
        if (s == WASTE_E_IO)
            fprintf(stderr, "  (the container is damaged: re-download or "
                            "re-convert it, then `tools/verify_container.py` "
                            "to find anything else)\n");
    }
    return 1;
}

/* ---- token callback ----------------------------------------------------- */

typedef struct {
    int quiet;
    uint32_t n;
    uint64_t hit, miss;
    double ms;
    const char *stop;              /* --stop, or NULL                       */
    char *pend;                    /* held back until the stop can't match  */
    size_t pend_n, pend_cap;
    int oom;
} sink;

/* A stop sequence can straddle pieces, so search a dynamic pending suffix
 * rather than the piece just produced. Returning nonzero cancels generation;
 * waste_generate reports that as WASTE_E_CANCELLED, a normal finish here. */
static int on_token(const waste_token_info *i, const char *piece, void *user)
{
    sink *s = (sink *)user;
    s->n++;
    s->hit += i->experts_hit;
    s->miss += i->experts_missed;
    s->ms += i->ms_total;

    if (!s->stop || !*s->stop) {
        if (!s->quiet) { fputs(piece, stdout); fflush(stdout); }
    } else {
        const size_t pl = strlen(piece);
        if (pl > SIZE_MAX - s->pend_n - 1) { s->oom = 1; return 1; }
        const size_t need = s->pend_n + pl + 1;
        if (need > s->pend_cap) {
            size_t cap = s->pend_cap ? s->pend_cap : 256;
            while (cap < need && cap <= SIZE_MAX / 2) cap *= 2;
            if (cap < need) cap = need;
            char *p = (char *)realloc(s->pend, cap);
            if (!p) { s->oom = 1; return 1; }
            s->pend = p;
            s->pend_cap = cap;
        }
        memcpy(s->pend + s->pend_n, piece, pl + 1);
        s->pend_n += pl;

        char *cut = strstr(s->pend, s->stop);
        if (cut) {
            if (!s->quiet) {
                fwrite(s->pend, 1, (size_t)(cut - s->pend), stdout);
                fflush(stdout);
            }
            s->pend_n = 0;
            return 1;
        }
        const size_t hold = strlen(s->stop);
        if (s->pend_n > hold) {
            const size_t emit = s->pend_n - hold;
            if (!s->quiet) { fwrite(s->pend, 1, emit, stdout); fflush(stdout); }
            memmove(s->pend, s->pend + emit, s->pend_n - emit + 1);
            s->pend_n -= emit;
        }
    }
    return 0;
}

/* ---- chat formatting -----------------------------------------------------
 *
 * An instruct model is trained on a conversation format, and feeding it a
 * bare line is asking it to continue text rather than to answer. HF ships
 * the format as a Jinja template; interpreting Jinja in C is a project of
 * its own, and the templates that matter reduce to a prefix and a suffix
 * per role, so that is what is read here. The converter copies the raw
 * .jinja into the container for tools that do interpret it.
 *
 * chat.json in the container, all fields optional:
 *   {"system":["<|im_start|>system\n","<|im_end|>\n"],
 *    "user":  ["<|im_start|>user\n","<|im_end|>\n"],
 *    "assistant":["<|im_start|>assistant\n","<|im_end|>\n"],
 *    "open":  "<|im_start|>assistant\n"}
 *
 * Without one the CLI says so and continues raw, which is honest: a
 * guessed format is worse than a visible absence. --raw forces it. */
typedef struct {
    char sys_p[128], sys_s[128];
    char usr_p[128], usr_s[128];
    char asst_p[128], asst_s[128];
    char open[128];
    int  have;
} chatfmt;

/* Closing quote of the JSON string whose opening quote is `p`, honouring
 * backslash escapes. Scanning with strchr instead treats the `\"` inside
 * `role=\"user\"` as a delimiter, which silently truncates every field of
 * an XTML chat template — the values that need embedded quotes are
 * exactly the ones this format is made of. */
static const char *jstr_end(const char *p)
{
    for (p++; *p; p++) {
        if (*p == '\\' && p[1]) { p++; continue; }
        if (*p == '"') return p;
    }
    return NULL;
}

static void jstr_field(const char *js, const char *key, int idx,
                       char *out, size_t cap)
{
    out[0] = 0;
    const char *k = strstr(js, key);
    if (!k) return;
    const char *p = strchr(k + strlen(key), '[');
    if (!p) return;
    for (int i = 0; i <= idx; i++) {
        p = strchr(p, '"');
        if (!p) return;
        if (i < idx) { p = jstr_end(p); if (!p) return; p++; }
    }
    p++;
    size_t n = 0;
    while (*p && *p != '"' && n + 1 < cap) {
        if (*p == '\\' && p[1]) {           /* \n and friends */
            p++;
            out[n++] = *p == 'n' ? '\n' : *p == 't' ? '\t' : *p;
        } else out[n++] = *p;
        p++;
    }
    out[n] = 0;
}

static int load_chatfmt(const char *model, chatfmt *f)
{
    memset(f, 0, sizeof *f);
    char path[1024];
    snprintf(path, sizeof path, "%s/chat.json", model);
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    char buf[8192];
    const size_t n = fread(buf, 1, sizeof buf - 1, fp);
    fclose(fp);
    buf[n] = 0;
    jstr_field(buf, "\"system\"", 0, f->sys_p, sizeof f->sys_p);
    jstr_field(buf, "\"system\"", 1, f->sys_s, sizeof f->sys_s);
    jstr_field(buf, "\"user\"", 0, f->usr_p, sizeof f->usr_p);
    jstr_field(buf, "\"user\"", 1, f->usr_s, sizeof f->usr_s);
    jstr_field(buf, "\"assistant\"", 0, f->asst_p, sizeof f->asst_p);
    jstr_field(buf, "\"assistant\"", 1, f->asst_s, sizeof f->asst_s);
    /* "open" is a bare string, not a pair; reuse the scanner on a fake array */
    const char *o = strstr(buf, "\"open\"");
    if (o) {
        const char *q = strchr(o + 6, '"');
        if (q) {
            size_t i = 0;
            for (q++; *q && *q != '"' && i + 1 < sizeof f->open; q++) {
                if (*q == '\\' && q[1]) { q++; f->open[i++] = *q == 'n' ? '\n' : *q == 't' ? '\t' : *q; }
                else f->open[i++] = *q;
            }
            f->open[i] = 0;
        }
    }
    f->have = f->usr_p[0] || f->asst_p[0] || f->open[0];
    return f->have;
}

/* ---- commands ----------------------------------------------------------- */

static int cmd_plan(int argc, char **argv)
{
    opts o; opts_init(&o);
    if (parse_opts(argc, argv, 2, &o)) return 2;
    if (o.n_pos < 1) { fprintf(stderr, "usage: waste plan MODEL [options]\n"); return 2; }

    waste_memplan p;
    const waste_status st = waste_plan_memory(o.pos[0], o.ctx, &p);
    if (st != WASTE_OK) return fail("plan", st);

    if (o.json) {
        printf("{\"ctx\":%u,\"trunk_bytes\":%llu,\"state_bytes\":%llu,"
               "\"scratch_bytes\":%llu,\"min_expert_cache\":%llu,"
               "\"floor_bytes\":%llu,\"recommended_bytes\":%llu",
               o.ctx, (unsigned long long)p.trunk_bytes,
               (unsigned long long)p.state_bytes,
               (unsigned long long)p.scratch_bytes,
               (unsigned long long)p.min_expert_cache,
               (unsigned long long)p.floor_bytes,
               (unsigned long long)p.recommended_bytes);
        /* The human form already says "machine N GB" and the JSON did not,
         * so anything reading this had to work out physical RAM for itself
         * — which on Windows means neither sysconf nor /proc exists and the
         * caller reimplements GlobalMemoryStatusEx. The engine already
         * knows; 0 when it cannot tell.
         *
         * Both figures, because in a container they differ and only the
         * second is the one the budget was sized against. A reader given
         * only physical_ram_bytes there would compute a ceiling the engine
         * never used, and conclude the engine had ignored its own rule. */
        printf(",\"physical_ram_bytes\":%llu,\"usable_ram_bytes\":%llu",
               (unsigned long long)waste_physical_ram(),
               (unsigned long long)waste_usable_ram());
        if (p.vision_bytes)
            printf(",\"vision_bytes\":%llu", (unsigned long long)p.vision_bytes);
        if (o.budget)
            printf(",\"budget_bytes\":%llu,\"expert_cache_bytes\":%lld",
                   (unsigned long long)o.budget,
                   o.budget < p.floor_bytes ? -1LL
                     : (long long)(o.budget - p.floor_bytes + p.min_expert_cache));
        printf("}\n");
        return o.budget && o.budget < p.floor_bytes ? 1 : 0;
    }

    char b[5][32];
    human(p.trunk_bytes, b[0], 32); human(p.state_bytes, b[1], 32);
    human(p.scratch_bytes, b[2], 32); human(p.min_expert_cache, b[3], 32);
    human(p.floor_bytes, b[4], 32);
    printf("memory plan for %s (ctx %u)\n\n", o.pos[0], o.ctx);
    printf("  resident trunk        %12s\n", b[0]);
    printf("  KDA state + KV cache  %12s\n", b[1]);
    printf("  scratch               %12s\n", b[2]);
    printf("  minimum expert cache  %12s\n", b[3]);
    printf("  ---------------------------------\n");
    printf("  FLOOR                 %12s\n", b[4]);
    human(p.recommended_bytes, b[0], 32);
    printf("  recommended           %12s   (floor + 3x a token's working set;\n"
           "                                      below that the cache keeps\n"
           "                                      nothing alive between tokens)\n", b[0]);
    if (p.vision_bytes) {
        char vb[32];
        human(p.vision_bytes, vb, 32);
        printf("\n  with --image           %12s more, taken from the expert\n"
               "                                      cache, not added to the budget\n", vb);
    }
    if (o.budget) {
        char bb[32];
        human(o.budget, bb, 32);
        if (o.budget < p.floor_bytes) printf("\n  budget %s is BELOW the floor — open would fail\n", bb);
        else {
            char cb[32];
            human(o.budget - p.floor_bytes + p.min_expert_cache, cb, 32);
            printf("\n  budget %s -> expert cache %s\n", bb, cb);
        }
    }
    return 0;
}

static int cmd_info(int argc, char **argv)
{
    opts o; opts_init(&o);
    if (parse_opts(argc, argv, 2, &o)) return 2;
    if (o.n_pos < 1) { fprintf(stderr, "usage: waste info MODEL [options]\n"); return 2; }
    waste_ctx *c;
    const waste_status st = open_model(o.pos[0], &o, &c);
    if (st != WASTE_OK) return fail("open", st);

    waste_model_info mi;
    waste_model_get_info(c, &mi);
    waste_memplan used;
    waste_memory_used(c, &used);
    if (o.json) {
        printf("{\"engine\":\"%s\",\"arch\":\"%s\",\"layers\":%u,"
               "\"experts\":%u,\"top_k\":%u,\"hidden\":%u,"
               "\"params_total\":%llu,\"params_active\":%llu,"
               "\"quantization\":\"%s\",\"expert_cache_bytes\":%llu}\n",
               waste_version(), mi.arch, mi.n_layers, mi.n_experts, mi.top_k,
               mi.hidden, (unsigned long long)mi.params_total,
               (unsigned long long)mi.params_active, mi.quant_summary,
               (unsigned long long)used.min_expert_cache);
        waste_close(c);
        return 0;
    }
    char cb[32];
    human(used.min_expert_cache, cb, 32);
    printf("%s\n\n", waste_build_info());
    printf("  arch          %s\n", mi.arch);
    printf("  layers        %u\n", mi.n_layers);
    printf("  experts       %u (top-%u)\n", mi.n_experts, mi.top_k);
    printf("  hidden        %u\n", mi.hidden);
    char pt[32], pa[32];
    params_human(mi.params_total, pt, sizeof pt);
    params_human(mi.params_active, pa, sizeof pa);
    printf("  parameters    %s total, %s active/token\n", pt, pa);
    printf("  quantization  %s\n", mi.quant_summary);
    printf("  expert cache  %s\n", cb);
    waste_close(c);
    return 0;
}

/* The prompt the tokenizer sees carries one placeholder per image, and the
 * engine expands each into as many positions as that image encodes to. The
 * wrapper tokens around it are the container's own names for the parts of a
 * media block; only the pad is load-bearing.
 *
 * Images go in front of the text. That is the convention every VLM prompt
 * follows and the only ordering a single --image flag can express; a caller
 * that needs a picture in the middle of a sentence should write the
 * placeholder itself and use the library. */
/* One media block is under 64 bytes, so this is the parser's limit spelled
 * in bytes: sizing it to a round 512 instead would have accepted 32 images
 * on the command line and refused the tenth here. */
/* Per image: the four markers are 58 bytes and `image 99999x99999` is
 * another 17, so 64 was exactly too small once the dimensions went in. */
#define MEDIA_HEAD_CAP (WASTE_MAX_IMAGES_CLI * 128 + 1)

static int prefix_images(waste_ctx *c, const opts *o, char *buf, size_t cap)
{
    size_t k = 0;
    for (int i = 0; i < o->n_image; i++) {
        size_t rows = 0;
        const waste_status st = waste_image_add(c, o->image[i], &rows);
        if (st == WASTE_E_UNSUPPORTED) {
            /* "unsupported" next to a filename reads as a rejected image
             * format, which is the wrong thing to go and check. */
            fprintf(stderr, "%s: this container has no vision tower\n",
                    o->image[i]);
            return 1;
        }
        if (st != WASTE_OK) return fail(o->image[i], st);
        if (!o->quiet)
            fprintf(stderr, "[%s: %zu image tokens]\n", o->image[i], rows);
        /* K3's own media block carries the source resolution as text:
         * `<|media_begin|>image 1024x768<|media_content|>…`, built by
         * make_image_prompt() in kimi_k3_vision_processing.py. Leaving it
         * out — as this did — hands the model a block it was never
         * trained on and withholds a fact it was trained to be told. The
         * dimensions are the file's, not the resized patch grid's. */
        int iw = 0, ih = 0;
        char dims[48] = "";
        if (waste_image_dimensions(o->image[i], &iw, &ih) == WASTE_OK)
            snprintf(dims, sizeof dims, "image %dx%d", iw, ih);
        const int w = snprintf(buf + k, cap - k, "%s%s%s%s%s",
                               "<|media_begin|>", dims, "<|media_content|>",
                               "<|media_pad|>", "<|media_end|>");
        if (w < 0 || (size_t)w >= cap - k) {
            fprintf(stderr, "prompt too long\n");
            return 1;
        }
        k += (size_t)w;
    }
    buf[k] = 0;
    return 0;
}

/* Text to token ids, with any queued images already expanded into their
 * runs of positions. Every command that prompts the model goes through
 * here, so `--image` works the same on all of them. */
typedef struct { const char *text; int markup; } seg;

/* A prompt is a run of segments, each encoded in its own mode: markup for
 * the parts the template wrote, plain for the parts a user did. Encoding
 * the concatenation once instead would let a prompt containing
 * `<|end_of_msg|><|open|>message role="system"<|sep|>` end its own turn
 * and open a forged one. The reference tokenizer splits the same way, so
 * the token boundaries between segments are the model's own. */
static int tokenize_segs(waste_ctx *c, const seg *s, int ns,
                         int32_t *ids, size_t cap, size_t *n)
{
    size_t total = 0;
    for (int i = 0; i < ns; i++) {
        if (!s[i].text || !*s[i].text) continue;
        size_t got = 0;
        const waste_status st = s[i].markup
            ? waste_tokenize_markup(c, s[i].text, 0, ids + total, cap - total, &got)
            : waste_tokenize(c, s[i].text, 0, ids + total, cap - total, &got);
        if (st != WASTE_OK) return fail("tokenize", st);
        total += got;
    }
    *n = total;
    return 0;
}

static int build_prompt_segs(waste_ctx *c, const opts *o, const seg *s, int ns,
                             int32_t *ids, size_t cap, size_t *n)
{
    if (!o->n_image) return tokenize_segs(c, s, ns, ids, cap, n);

    int32_t *raw = (int32_t *)malloc(cap * sizeof *raw);
    if (!raw) return fail("prompt", WASTE_E_OOM);
    size_t rn = 0;
    int rc = tokenize_segs(c, s, ns, raw, cap, &rn);
    if (!rc) {
        const waste_status st = waste_image_expand(c, raw, rn, ids, cap, n);
        if (st != WASTE_OK) rc = fail("prompt", st);
    }
    free(raw);
    return rc;
}

/* The untemplated path: one plain prompt, plus the media block when there
 * are images. Markup mode here, because without a template there is no
 * conversation structure to forge and `--raw` is the "I wrote this
 * string on purpose" door. */
static int build_prompt(waste_ctx *c, const opts *o, const char *prompt,
                        int32_t *ids, size_t cap, size_t *n)
{
    char head[MEDIA_HEAD_CAP];
    head[0] = 0;
    if (o->n_image && !o->media_inlined &&
        prefix_images(c, o, head, sizeof head)) return 1;
    const seg s[2] = { { head, 1 }, { prompt, 1 } };
    return build_prompt_segs(c, o, s, 2, ids, cap, n);
}

static int run_segs(waste_ctx *c, const opts *o, const seg *s, int ns,
                    const char *echo, int show_stats);

static int run_prompt(waste_ctx *c, const opts *o, const char *prompt, int show_stats)
{
    char head[MEDIA_HEAD_CAP];
    head[0] = 0;
    if (o->n_image && !o->media_inlined &&
        prefix_images(c, o, head, sizeof head)) return 1;
    const seg s[2] = { { head, 1 }, { prompt, 1 } };
    opts q = *o;
    q.media_inlined = 1;          /* prefix_images already ran, above */
    return run_segs(c, &q, s, 2, prompt, show_stats);
}

static int run_segs(waste_ctx *c, const opts *o, const seg *segs, int ns,
                    const char *echo, int show_stats)
{
    int32_t ids[MAXTOK];
    size_t n = 0;
    waste_status st;
    if (build_prompt_segs(c, o, segs, ns, ids, MAXTOK, &n)) return 1;

    waste_gen_params p;
    waste_gen_params_init(&p);
    p.max_tokens = o->max_tokens;
    p.temperature = o->temperature;
    p.top_p = o->top_p;
    p.top_k = o->top_k;
    p.seed = o->seed;

    sink s;
    memset(&s, 0, sizeof s);
    s.quiet = o->quiet;
    s.stop = o->stop;
    /* chat already showed the user what they typed, and with a
     * template the prompt is mostly role markers */
    if (!o->quiet && !o->no_echo && echo) fputs(echo, stdout);
    st = waste_generate(c, ids, n, &p, on_token, &s);
    if (!o->quiet && s.pend_n) fwrite(s.pend, 1, s.pend_n, stdout);
    free(s.pend);
    printf("\n");
    if (s.oom) return fail("stop matcher", WASTE_E_OOM);
    if (st != WASTE_OK && st != WASTE_E_CANCELLED) return fail_ctx("generate", st, c);
    /* A generation that ran out of context returns OK — the tokens it
     * produced are good — and stops mid-answer. Silence there reads as a
     * model that had nothing more to say, which is the wrong thing to
     * conclude. To stderr, so a piped run is still just the generation. */
    if (st == WASTE_OK) {
        const char *why = waste_error_detail(c);
        if (why) fprintf(stderr, "\nwaste: %s\n", why);
    }

    if (show_stats && s.n) {
        const double sec = s.ms / 1000.0;
        fprintf(stderr, "\n[%u tokens, %.2f s, %.2f tok/s | experts %llu hit / "
                        "%llu miss = %.0f%%]\n",
                s.n, sec, s.n / sec,
                (unsigned long long)s.hit, (unsigned long long)s.miss,
                100.0 * (double)s.hit / (double)(s.hit + s.miss ? s.hit + s.miss : 1));
    }
    return 0;
}

/* Without --budget the engine picks the ceiling itself, and since that
 * choice depends on the machine as well as the model, the same command on
 * two laptops runs under two different ceilings. Say which one. To stderr,
 * so a piped generation stays just the generation. */
static void show_chosen_budget(waste_ctx *c, const opts *o)
{
    if (o->budget || o->quiet) return;
    waste_memplan u;
    if (waste_memory_used(c, &u) != WASTE_OK) return;
    /* waste_memory_used reports the cache the engine really allocated, so
     * this is what is held, not what was allowed. */
    const uint64_t used = u.trunk_bytes + u.state_bytes + u.scratch_bytes +
                          u.min_expert_cache;
    /* Usable, not physical: inside a cgroup "using 20 GB of 256 GB" names a
     * machine this process does not have, and the number that explains the
     * choice is the one the choice was made from. */
    const uint64_t phys = waste_usable_ram();
    char ub[32], cb[32], pb[32];
    human(used, ub, 32);
    human(u.min_expert_cache, cb, 32);
    if (phys) {
        human(phys, pb, 32);
        fprintf(stderr, "waste: no --budget, using %s of %s (expert cache %s)\n",
                ub, pb, cb);
    } else {
        fprintf(stderr, "waste: no --budget, using %s (expert cache %s)\n", ub, cb);
    }
}

static int cmd_run(int argc, char **argv)
{
    opts o; opts_init(&o);
    if (parse_opts(argc, argv, 2, &o)) return 2;
    if (o.n_pos < 1) {
        fprintf(stderr, "usage: waste run MODEL [\"prompt\" | - | --file F] [options]\n");
        return 2;
    }
    char *prompt = read_prompt(&o);
    if (!prompt || !*prompt) {
        free(prompt);
        fprintf(stderr, "no prompt: give one as an argument, with --file, "
                        "or on stdin\n");
        return 2;
    }
    waste_ctx *c;
    const waste_status st = open_model(o.pos[0], &o, &c);
    if (st != WASTE_OK) { free(prompt); return fail("open", st); }
    show_chosen_budget(c, &o);

    /* A one-shot run is a question when the container knows how to ask
     * one. Without this, `run` was raw continuation even on a container
     * carrying chat.json, so the only way to get an answer rather than a
     * completion was the interactive `chat` — and every single-command
     * example had to be a completion. --raw restores the old behaviour,
     * which is what you want for a base model or to measure the prompt
     * the model actually sees. */
    chatfmt fmt;
    int r;
    if (!o.raw && load_chatfmt(o.pos[0], &fmt)) {
        char head[MEDIA_HEAD_CAP];
        head[0] = 0;
        if (o.n_image) {
            if (prefix_images(c, &o, head, sizeof head)) {
                waste_close(c); free(prompt); return 1;
            }
            o.media_inlined = 1;
        }
        /* Markup is the template's; the system text and the prompt are
         * the caller's and are encoded as plain text. The image block
         * goes inside the user turn, before the words. */
        const seg s[7] = {
            { o.system && *o.system ? fmt.sys_p : "", 1 },
            { o.system ? o.system : "",               0 },
            { o.system && *o.system ? fmt.sys_s : "", 1 },
            { fmt.usr_p,                              1 },
            { head,                                   1 },
            { prompt,                                 0 },
            { fmt.usr_s,                              1 },
        };
        seg all[8];
        memcpy(all, s, sizeof s);
        all[7] = (seg){ fmt.open, 1 };
        /* stop where the assistant's turn closes, so the marker does not
         * land in the user's output */
        if (!o.stop && fmt.asst_s[0]) o.stop = fmt.asst_s;
        o.no_echo = 1;              /* the prompt here is mostly markers */
        r = run_segs(c, &o, all, 8, NULL, !o.quiet);
    } else {
        r = run_prompt(c, &o, prompt, !o.quiet);
    }
    if (o.learn) waste_save_usage(c);
    waste_close(c);
    free(prompt);
    return r;
}

static int cmd_chat(int argc, char **argv)
{
    opts o; opts_init(&o);
    if (parse_opts(argc, argv, 2, &o)) return 2;
    if (o.n_pos < 1) { fprintf(stderr, "usage: waste chat MODEL [options]\n"); return 2; }
    waste_ctx *c;
    const waste_status st = open_model(o.pos[0], &o, &c);
    if (st != WASTE_OK) return fail("open", st);
    show_chosen_budget(c, &o);

    chatfmt fmt;
    const int templated = !o.raw && load_chatfmt(o.pos[0], &fmt);
    printf("%s\n", waste_build_info());
    if (templated)
        printf("chat format from %s/chat.json\n", o.pos[0]);
    else if (o.raw)
        printf("raw continuation (--raw)\n");
    else
        printf("no chat.json in the container: raw continuation, so an "
               "instruct model is being asked to continue text rather than "
               "to answer.\n  Copy one in — examples/ has K3's — or pass "
               "--raw to say you meant it\n");
    printf("/reset clears state, /save FILE and /load FILE persist it, "
           "/image FILE attaches a picture, /stats prints counters, "
           "Ctrl-D exits\n");

    /* The system turn goes in once, before anything else. */
    if (templated && o.system && *o.system) {
        opts q = o;
        q.n_image = 0;              /* a picture belongs to a user turn */
        const seg system_turn[3] = {
            { fmt.sys_p, 1 }, { o.system, 0 }, { fmt.sys_s, 1 },
        };
        int32_t ids[MAXTOK];
        size_t n = 0;
        if (build_prompt_segs(c, &q, system_turn, 3, ids, MAXTOK, &n)) {
            waste_close(c);
            return 1;
        }
        /* Prefill only.  Generating one token and resetting cannot remove
         * just that token: reset also erases the system turn itself. */
        const waste_status es = waste_eval(c, ids, n, NULL, NULL);
        if (es != WASTE_OK) {
            const int rc = fail_ctx("system prompt", es, c);
            waste_close(c);
            return rc;
        }
    }
    /* Attached paths live here rather than in strdup'd memory: image[]
     * otherwise mixes argv pointers with owned ones and /reset in a long
     * session would leak one path per attachment. */
    char attach[WASTE_MAX_IMAGES_CLI][512];
    char line[8192];
    while (1) {
        fputs("\n> ", stdout);
        fflush(stdout);
        if (!fgets(line, sizeof line, stdin)) break;
        size_t l = strlen(line);
        while (l && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = 0;
        if (!l) continue;
        if (!strcmp(line, "/reset")) {
            waste_state_reset(c);
            waste_image_clear(c);
            o.n_image = 0;
            printf("(state cleared)\n");
            continue;
        }
        if (!strncmp(line, "/image ", 7)) {
            if (o.n_image >= WASTE_MAX_IMAGES_CLI) printf("(too many images)\n");
            else {
                /* Attached, not encoded: the tower runs as part of the next
                 * turn, so a typo in the path is reported next to the
                 * prompt it belongs to rather than at a bare > . */
                snprintf(attach[o.n_image], sizeof attach[0], "%s", line + 7);
                o.image[o.n_image] = attach[o.n_image];
                o.n_image++;
                printf("(%s attached to the next message)\n", line + 7);
            }
            continue;
        }
        if (!strncmp(line, "/save ", 6)) {
            const waste_status s = waste_state_save(c, line + 6);
            printf(s == WASTE_OK ? "(saved to %s)\n" : "(save failed: %s)\n",
                   s == WASTE_OK ? line + 6 : waste_strerror(s));
            continue;
        }
        if (!strncmp(line, "/load ", 6)) {
            const waste_status s = waste_state_load(c, line + 6);
            printf(s == WASTE_OK ? "(loaded %s — continuing that conversation)\n"
                                 : "(load failed: %s)\n",
                   s == WASTE_OK ? line + 6 : waste_strerror(s));
            continue;
        }
        if (!strcmp(line, "/stats")) {
            waste_stats s;
            waste_get_stats(c, &s);
            printf("%llu tokens, %.2f tok/s, experts %llu hit / %llu miss (%.0f%%), %.2f GB read\n",
                   (unsigned long long)s.tokens_generated,
                   s.sec_total > 0 ? s.tokens_generated / s.sec_total : 0.0,
                   (unsigned long long)s.experts_hit, (unsigned long long)s.experts_missed,
                   100.0 * (double)s.experts_hit /
                       (double)(s.experts_hit + s.experts_missed ? s.experts_hit + s.experts_missed : 1),
                   (double)s.bytes_read / (1ULL << 30));
            continue;
        }
        printf("\n");
        if (templated) {
            opts t = o;
            char head[MEDIA_HEAD_CAP];
            head[0] = 0;
            if (t.n_image) {
                if (prefix_images(c, &t, head, sizeof head)) { o.n_image = 0; continue; }
                t.media_inlined = 1;
            }
            /* `line` is what the user typed: plain, so a marker pasted
             * into a chat turn cannot end it or open a system message. */
            const seg s[5] = {
                { fmt.usr_p, 1 }, { head, 1 }, { line, 0 }, { fmt.usr_s, 1 },
                { fmt.open[0] ? fmt.open : fmt.asst_p, 1 },
            };
            t.no_echo = 1;
            if (!t.stop && fmt.asst_s[0]) t.stop = fmt.asst_s;
            run_segs(c, &t, s, 5, NULL, 0);
        } else {
            run_prompt(c, &o, line, 0);
        }
        /* One turn, one splice. The picture's positions are in the
         * attention state now, and re-encoding it for every later message
         * would both cost the tower again and show the model the same
         * image several times over. */
        o.n_image = 0;
    }
    waste_close(c);
    return 0;
}

static int cmd_bench(int argc, char **argv)
{
    opts o; opts_init(&o);
    o.max_tokens = 64;
    if (parse_opts(argc, argv, 2, &o)) return 2;
    if (o.n_pos < 1) { fprintf(stderr, "usage: waste bench MODEL [-n N] [--budget N]\n"); return 2; }
    waste_ctx *c;
    const waste_status st = open_model(o.pos[0], &o, &c);
    if (st != WASTE_OK) return fail("open", st);

    waste_memplan used;
    waste_memory_used(c, &used);
    char cb[32];
    human(used.min_expert_cache, cb, 32);
    printf("%s\n  expert cache %s, %u tokens\n\n", waste_build_info(), cb, o.max_tokens);

    opts q = o;
    q.quiet = 1;
    run_prompt(c, &q, "Write a C function that parses a JSON array of integers "
                      "and explain its edge cases.", 0);

    waste_stats s;
    waste_get_stats(c, &s);
    const double tps = s.sec_total > 0 ? s.tokens_generated / s.sec_total : 0;
    const uint64_t acc = s.experts_hit + s.experts_missed;
    if (o.json) {
        printf("{\"tokens\":%llu,\"tok_per_s\":%.4f,\"experts_hit\":%llu,"
               "\"experts_missed\":%llu,\"bytes_read\":%llu,"
               "\"direct_io\":%s}\n",
               (unsigned long long)s.tokens_generated, tps,
               (unsigned long long)s.experts_hit,
               (unsigned long long)s.experts_missed,
               (unsigned long long)s.bytes_read,
               s.direct_io ? "true" : "false");
        waste_close(c);
        return 0;
    }
    printf("  %.2f tok/s (%.0f ms/token)\n", tps, 1000.0 / (tps > 0 ? tps : 1));
    if (!s.direct_io)
        printf("  note      page cache not bypassed on this filesystem — the hit\n"
               "            rate below is partly the kernel's, not the engine's\n");
    printf("  experts   %llu hit / %llu miss = %.1f%% hit\n",
           (unsigned long long)s.experts_hit, (unsigned long long)s.experts_missed,
           100.0 * (double)s.experts_hit / (double)(acc ? acc : 1));
    printf("  disk      %.2f GB total, %.3f GB/token\n",
           (double)s.bytes_read / (1ULL << 30),
           (double)s.bytes_read / (1ULL << 30) /
               (s.tokens_generated ? s.tokens_generated : 1));
    waste_close(c);
    return 0;
}

static char *detokenize_alloc(waste_ctx *c, const int32_t *ids, size_t n,
                              size_t *used, waste_status *status)
{
    size_t cap = n <= (SIZE_MAX - 64) / 4 ? n * 4 + 64 : 0;
    if (cap < 64) cap = 64;
    char *out = (char *)malloc(cap);
    if (!out) { *status = WASTE_E_OOM; return NULL; }
    for (;;) {
        size_t need = 0;
        *status = waste_detokenize(c, ids, n, out, cap, &need);
        if (*status == WASTE_OK) { *used = need; return out; }
        if (*status != WASTE_E_ARG || need >= SIZE_MAX - 1) {
            free(out);
            return NULL;
        }
        cap = need + 1;
        char *grown = (char *)realloc(out, cap);
        if (!grown) { free(out); *status = WASTE_E_OOM; return NULL; }
        out = grown;
    }
}

/* Tokenize and detokenize: the round trip the engine does on every prompt,
 * exposed because debugging a container without it means guessing. */
static int cmd_tokens(int argc, char **argv, int decode)
{
    opts o; opts_init(&o);
    if (parse_opts(argc, argv, 2, &o)) return 2;
    if (o.n_pos < 1) {
        fprintf(stderr, "usage: waste %s MODEL [text|ids | - | --file F]\n",
                decode ? "detokenize" : "tokenize");
        return 2;
    }
    char *text = read_prompt(&o);
    if (!text || !*text) { free(text); fprintf(stderr, "nothing to read\n"); return 2; }
    waste_ctx *c;
    const waste_status st = open_model(o.pos[0], &o, &c);
    if (st != WASTE_OK) { free(text); return fail("open", st); }

    int rc = 0;
    if (decode) {
        int32_t ids[MAXTOK];
        size_t n = 0;
        for (char *t = strtok(text, " ,\t\n"); t && n < MAXTOK; t = strtok(NULL, " ,\t\n"))
            ids[n++] = (int32_t)atoi(t);
        size_t used = 0;
        waste_status d = WASTE_OK;
        char *out = detokenize_alloc(c, ids, n, &used, &d);
        if (!out) rc = fail("detokenize", d);
        else { fwrite(out, 1, used, stdout); putchar('\n'); free(out); }
    } else {
        int32_t ids[MAXTOK];
        size_t n = 0;
        /* Markup mode: this command is how a chat template gets checked
         * marker by marker, so it has to show `<|sep|>` as the one token
         * it is. Prompts go through waste_tokenize instead, where the
         * same string is ordinary text. */
        const waste_status t = waste_tokenize_markup(c, text, 0, ids, MAXTOK, &n);
        if (t != WASTE_OK) rc = fail("tokenize", t);
        else if (o.json) {
            printf("{\"n\":%zu,\"ids\":[", n);
            for (size_t i = 0; i < n; i++) printf("%s%d", i ? "," : "", ids[i]);
            printf("]}\n");
        } else {
            for (size_t i = 0; i < n; i++) printf("%s%d", i ? " " : "", ids[i]);
            printf("\n");
        }
    }
    waste_close(c);
    free(text);
    return rc;
}

/* Next-token distribution without generating: what waste_eval is for, and
 * the only way to get a logit out of the CLI. */
static int cmd_eval(int argc, char **argv)
{
    opts o; opts_init(&o);
    if (parse_opts(argc, argv, 2, &o)) return 2;
    if (o.n_pos < 1) {
        fprintf(stderr, "usage: waste eval MODEL [\"prompt\" | - | --file F] "
                        "[--top-k N] [--json]\n");
        return 2;
    }
    char *prompt = read_prompt(&o);
    if (!prompt || !*prompt) { free(prompt); fprintf(stderr, "no prompt\n"); return 2; }
    waste_ctx *c;
    const waste_status st = open_model(o.pos[0], &o, &c);
    if (st != WASTE_OK) { free(prompt); return fail("open", st); }

    int32_t ids[MAXTOK];
    size_t n = 0;
    if (build_prompt(c, &o, prompt, ids, MAXTOK, &n)) {
        waste_close(c); free(prompt); return 1;
    }

    waste_status t;
    const float *lg = NULL;
    size_t vocab = 0;
    t = waste_eval(c, ids, n, &lg, &vocab);
    if (t != WASTE_OK) { const int rc = fail_ctx("eval", t, c);
                         waste_close(c); free(prompt); return rc; }

    /* softmax over the whole vocabulary, then report the top few */
    int k = o.top_k > 0 ? o.top_k : 10;
    if ((size_t)k > vocab) k = (int)vocab;
    float mx = lg[0];
    for (size_t v = 1; v < vocab; v++) if (lg[v] > mx) mx = lg[v];
    double sum = 0;
    for (size_t v = 0; v < vocab; v++) sum += exp((double)(lg[v] - mx));

    /* Its own array: borrowing the tail of `ids` would collide with the
     * prompt as soon as the prompt got long. */
    int32_t *picked = (int32_t *)malloc((size_t)k * sizeof *picked);
    if (!picked) { waste_close(c); free(prompt); return fail("eval", WASTE_E_OOM); }

    if (o.json) printf("{\"prompt_tokens\":%zu,\"vocab\":%zu,\"top\":[", n, vocab);
    for (int i = 0; i < k; i++) {
        size_t best = 0;
        float bv = -3.4e38f;
        for (size_t v = 0; v < vocab; v++) {
            int seen = 0;
            for (int j = 0; j < i; j++) if (picked[j] == (int32_t)v) seen = 1;
            if (!seen && lg[v] > bv) { bv = lg[v]; best = v; }
        }
        picked[i] = (int32_t)best;
        const double pr = exp((double)(bv - mx)) / sum;
        size_t used = 0;
        const int32_t one = (int32_t)best;
        waste_status ds = WASTE_OK;
        char *piece = detokenize_alloc(c, &one, 1, &used, &ds);
        if (o.json)
            printf("%s{\"id\":%zu,\"logit\":%.4f,\"prob\":%.6f}", i ? "," : "", best, bv, pr);
        else
            printf("  %6zu  %8.4f  %7.4f  %.*s\n", best, bv, pr,
                   piece ? (int)used : 0, piece ? piece : "");
        free(piece);
    }
    if (o.json) printf("]}\n");

    free(picked);
    waste_close(c);
    free(prompt);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        printf("waste %s — run large MoE models from a streaming container\n\n"
               "  waste run        MODEL \"prompt\"   generate\n"
               "  waste chat       MODEL            interactive, state kept\n"
               "  waste eval       MODEL \"prompt\"   next-token distribution\n"
               "  waste tokenize   MODEL \"text\"     text -> token ids\n"
               "  waste detokenize MODEL \"1 2 3\"    token ids -> text\n"
               "  waste bench      MODEL            throughput and cache stats\n"
               "  waste plan       MODEL            memory floor and budget\n"
               "  waste info       MODEL            container details\n"
               "  waste version\n\n"
               "The prompt may be an argument, a file (--file F), or stdin —\n"
               "given as - or simply piped in.\n\n"
               "options: --budget 8G  --ctx N  -n N  --temp F  --top-p F\n"
               "         --top-k N  --seed N  --threads N  --cpus LIST\n"
               "         --stop STR  --file F  --json  -q  --learn  --verify\n"
               "         --allow-concurrent-open\n"
         "  --stop  ends generation when the text appears\n"
         "  --json  machine-readable output for eval, tokenize, plan,\n"
         "          info and bench\n"
         "  --learn records which experts the run used, so the next open\n"
         "  starts with a warm cache instead of an empty one\n"
         "  --threads sets the compute pool; 0 (default) is one per core\n"
         "  --cpus restricts that pool to a cpu list — \"0-5\", \"0-2,6-8\" —\n"
         "  and then --threads 0 means one per CPU listed. Worth it where\n"
         "  the cores differ: on a two-die Ryzen, six threads on one die\n"
         "  measured 16-25%% faster than six split across both. Linux and\n"
         "  Windows; the default is to leave placement to the OS\n"
         "  --allow-concurrent-open opts out of the POSIX container lock\n"
         "  --verify checks each expert record's checksum as it is read,\n"
         "  for a container you have not read since copying it. Costs ~5%%\n"
         "  on Kimi-Linear, ~1%% on K3; off otherwise\n",
               waste_version());
        return argc < 2 ? 2 : 0;
    }
    if (!strcmp(argv[1], "version")) {
        printf("%s\n", waste_build_info());
        return 0;
    }
    if (!strcmp(argv[1], "run"))   return cmd_run(argc, argv);
    if (!strcmp(argv[1], "chat"))  return cmd_chat(argc, argv);
    if (!strcmp(argv[1], "bench")) return cmd_bench(argc, argv);
    if (!strcmp(argv[1], "plan"))  return cmd_plan(argc, argv);
    if (!strcmp(argv[1], "info"))  return cmd_info(argc, argv);
    if (!strcmp(argv[1], "eval"))  return cmd_eval(argc, argv);
    if (!strcmp(argv[1], "tokenize"))   return cmd_tokens(argc, argv, 0);
    if (!strcmp(argv[1], "detokenize")) return cmd_tokens(argc, argv, 1);
    fprintf(stderr, "unknown command '%s' (try --help)\n", argv[1]);
    return 2;
}
