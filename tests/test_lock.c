/* SPDX-License-Identifier: Apache-2.0 */
/* test_lock.c — container ownership is process-wide and leak-free. */
#include "../src/waste.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static int failed;

#define CHECK(expr, what) do {                                             \
    if (!(expr)) {                                                         \
        fprintf(stderr, "FAIL line %d: %s\n", __LINE__, (what));          \
        failed = 1;                                                        \
    }                                                                      \
} while (0)

static int copy_file(const char *src, const char *dst)
{
    const int in = open(src, O_RDONLY);
    if (in < 0) return -1;
    const int out = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (out < 0) { close(in); return -1; }
    char buf[16384];
    int rc = 0;
    for (;;) {
        ssize_t n;
        do n = read(in, buf, sizeof buf); while (n < 0 && errno == EINTR);
        if (n <= 0) { if (n < 0) rc = -1; break; }
        ssize_t off = 0;
        while (off < n) {
            ssize_t put;
            do put = write(out, buf + off, (size_t)(n - off));
            while (put < 0 && errno == EINTR);
            if (put <= 0) { rc = -1; break; }
            off += put;
        }
        if (rc) break;
    }
    if (close(out)) rc = -1;
    close(in);
    return rc;
}

static int write_text(const char *path, const char *text)
{
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return -1;
    const size_t n = strlen(text);
    const int rc = write(fd, text, n) == (ssize_t)n ? 0 : -1;
    return close(fd) ? -1 : rc;
}

/* A separately opened descriptor must not be able to take the directory's
 * flock while the library owns it. */
static int raw_lock_available(const char *model)
{
    const int fd = open(model, O_RDONLY);
    if (fd < 0) return 0;
    int rc;
    do rc = flock(fd, LOCK_EX | LOCK_NB); while (rc && errno == EINTR);
    if (!rc) (void)flock(fd, LOCK_UN);
    close(fd);
    return rc == 0;
}

static int probe(const char *model, int allow, uint64_t budget,
                 waste_status expected)
{
    waste_cfg cfg;
    waste_cfg_init(&cfg);
    cfg.allow_concurrent_open = allow;
    cfg.ram_budget_bytes = budget;
    waste_ctx *ctx = NULL;
    const waste_status got = waste_open(model, &cfg, &ctx);
    if (ctx) waste_close(ctx);
    if (got != expected) {
        fprintf(stderr, "probe: got %d (%s), expected %d (%s)\n",
                got, waste_strerror(got), expected, waste_strerror(expected));
        return 1;
    }
    return 0;
}

/* Exec makes this a genuinely separate library instance rather than relying
 * on fork's copied registry. It also verifies the lock descriptor is closed
 * across exec while the parent's descriptor continues to own the lock. */
static int child_probe(const char *self, const char *model, int allow,
                       uint64_t budget, waste_status expected)
{
    const pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) {
        char a[2], b[32], e[16];
        snprintf(a, sizeof a, "%d", allow);
        snprintf(b, sizeof b, "%llu", (unsigned long long)budget);
        snprintf(e, sizeof e, "%d", expected);
        execl(self, self, "--probe", model, a, b, e, (char *)NULL);
        _exit(127);
    }
    int ws;
    pid_t got;
    do got = waitpid(pid, &ws, 0); while (got < 0 && errno == EINTR);
    return got != pid || !WIFEXITED(ws) || WEXITSTATUS(ws) != 0;
}

static void remove_variant(const char *dir, int trunk)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/manifest.json", dir);
    (void)unlink(path);
    if (trunk) {
        snprintf(path, sizeof path, "%s/trunk.bin", dir);
        (void)unlink(path);
    }
    (void)rmdir(dir);
}

int main(int argc, char **argv)
{
    if (argc == 6 && !strcmp(argv[1], "--probe")) {
        const int allow = atoi(argv[3]);
        const uint64_t budget = (uint64_t)strtoull(argv[4], NULL, 10);
        const waste_status expected = (waste_status)strtol(argv[5], NULL, 10);
        return probe(argv[2], allow, budget, expected);
    }
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL SCRATCH-DIR\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const char *scratch = argv[2];

    waste_cfg cfg;
    waste_cfg_init(&cfg);
    CHECK(cfg.allow_concurrent_open == 0, "ownership must default on");
    CHECK(strstr(waste_strerror(WASTE_E_BUSY), "another process") != NULL,
          "WASTE_E_BUSY must explain the contention");
    waste_memplan plan;
    if (waste_plan_memory(model, cfg.ctx_tokens, &plan) != WASTE_OK) {
        fprintf(stderr, "cannot plan lock-test container\n");
        return 1;
    }
    cfg.ram_budget_bytes = plan.floor_bytes;

    waste_ctx *a = NULL, *b = NULL;
    CHECK(waste_open(model, &cfg, &a) == WASTE_OK, "first open");
    CHECK(a != NULL, "first context");
    if (!a) return 1;
    CHECK(!raw_lock_available(model), "open context must own directory");

    /* The second context shares the process entry instead of contending with
     * its own flock. Keeping it open exercises reference-counted release. */
    CHECK(waste_open(model, &cfg, &b) == WASTE_OK, "same-process second open");
    CHECK(b != NULL, "second context");
    CHECK(child_probe(argv[0], model, 0, 1, WASTE_E_BUSY) == 0,
          "competing process must receive WASTE_E_BUSY before budgeting");
    CHECK(child_probe(argv[0], model, 1, 1, WASTE_E_RAM_BUDGET) == 0,
          "explicit opt-out must bypass ownership");

    waste_close(a);
    a = NULL;
    CHECK(!raw_lock_available(model),
          "closing one context must retain the other context's ownership");
    CHECK(child_probe(argv[0], model, 0, 1, WASTE_E_BUSY) == 0,
          "remaining same-process reference must exclude competitors");

    waste_close(b);
    b = NULL;
    CHECK(raw_lock_available(model), "last close must release ownership");
    CHECK(child_probe(argv[0], model, 0, 1, WASTE_E_RAM_BUDGET) == 0,
          "a new process must pass ownership after normal close");

    /* Every return after acquisition must release the OS lock. */
    CHECK(probe(model, 0, 1, WASTE_E_RAM_BUDGET) == 0,
          "budget failure status");
    CHECK(raw_lock_available(model), "budget failure must release ownership");

    char bad_plan[1024], bad_load[1024], src[1024], dst[1024];
    snprintf(bad_plan, sizeof bad_plan, "%s/lock-bad-plan-%ld", scratch,
             (long)getpid());
    CHECK(mkdir(bad_plan, 0700) == 0, "create malformed variant");
    snprintf(dst, sizeof dst, "%s/manifest.json", bad_plan);
    CHECK(write_text(dst, "{") == 0, "write malformed manifest");
    waste_ctx *ctx = NULL;
    const waste_status plan_st = waste_open(bad_plan, &cfg, &ctx);
    CHECK(plan_st == WASTE_E_FORMAT && ctx == NULL, "planning failure status");
    CHECK(raw_lock_available(bad_plan), "planning failure must release ownership");
    remove_variant(bad_plan, 0);

    /* Copy enough for planning and trunk allocation, then omit codebooks so
     * model loading fails after partial construction. */
    snprintf(bad_load, sizeof bad_load, "%s/lock-bad-load-%ld", scratch,
             (long)getpid());
    CHECK(mkdir(bad_load, 0700) == 0, "create partial-load variant");
    snprintf(src, sizeof src, "%s/manifest.json", model);
    snprintf(dst, sizeof dst, "%s/manifest.json", bad_load);
    CHECK(copy_file(src, dst) == 0, "copy partial manifest");
    snprintf(src, sizeof src, "%s/trunk.bin", model);
    snprintf(dst, sizeof dst, "%s/trunk.bin", bad_load);
    CHECK(copy_file(src, dst) == 0, "copy partial trunk");
    ctx = NULL;
    const waste_status load_st = waste_open(bad_load, &cfg, &ctx);
    CHECK(load_st != WASTE_OK && load_st != WASTE_E_BUSY && ctx == NULL,
          "partial model-load failure status");
    CHECK(raw_lock_available(bad_load),
          "partial model-load failure must release ownership");
    remove_variant(bad_load, 1);

    if (failed) return 1;
    puts("PASS model-container ownership lock");
    return 0;
}

#else
int main(void)
{
    waste_cfg cfg;
    waste_cfg_init(&cfg);
    if (cfg.allow_concurrent_open != 0) return 1;
    puts("PASS model-container ownership lock (not used on this platform)");
    return 0;
}
#endif
