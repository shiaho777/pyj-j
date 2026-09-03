/*
 * vendor/app.c -- the downstream-application template.
 *
 * This is what shipping the kernel looks like: ~100 lines of C, no
 * Python, no numpy, no BLAS, no jsource. The app links the prebuilt
 * libj next to it, loads the two kernel scripts, trains a classifier
 * entirely inside the engine, then serves in-process predictions.
 *
 * The only data that crosses the boundary is scalar results read back
 * with JGetM. The app never touches a matrix.
 *
 * Build: see Makefile (one line). Run from this directory:
 *   DYLD_LIBRARY_PATH=. ./vendor_demo     (macOS)
 *   LD_LIBRARY_PATH=.    ./vendor_demo    (Linux)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void* JS;
extern JS JInit2(char*);
extern void JSM(JS, void**);
extern int JDo(JS, char*);
extern int JGetM(JS, char*, long long*, long long*, long long**, long long**);
extern int JFree(JS);

/* Engine output -> stdout; blank result lines are dropped. */
static void app_output(void* jt, int type, char* s)
{
  (void)jt;
  if (type == 5) return;                 /* MTYOEXIT */
  if (s && *s && strspn(s, " \t\n") != strlen(s)) {
    fputs(s, stdout); fputc('\n', stdout);
  }
}

static JS jt = NULL;

static int jdo(const char* sentence)
{
  int rc = JDo(jt, (char*)sentence);
  if (rc != 0) fprintf(stderr, "app: JDo failed rc=%d: %s\n", rc, sentence);
  return rc;
}

/* Read a scalar FL noun; returns 0 on success. */
static int jget_scalar(const char* name, double* out)
{
  long long jtype = 0, jrank = 0;
  long long *jshape = NULL, *jdata = NULL;
  if (JGetM(jt, (char*)name, &jtype, &jrank, &jshape, &jdata) != 0 ||
      jrank != 0 || !(jtype & 8LL /* FL */)) {
    fprintf(stderr, "app: %s is not a scalar float (rank %lld type %llx)\n",
            name, jrank, jtype);
    return -1;
  }
  *out = *(double*)jdata;
  return 0;
}

int main(void)
{
  char sentence[512];
  double acc, p0, p1;

  jt = JInit2((char*)".");               /* engine + libgmp live here */
  if (!jt) { fprintf(stderr, "app: JInit2 failed (run from the vendor dir)\n"); return 1; }
  void* callbacks[5] = {app_output, 0, 0, 0, (void*)3L};
  JSM(jt, callbacks);

  if (jdo("0!:0 <'kernel/ad.ijs'")) return 1;
  if (jdo("0!:0 <'kernel/train.ijs'")) return 1;

  puts("vendor: training two-spiral classifier inside libj...");
  if (jdo("sp_seed 42")) return 1;
  if (jdo("sp_data 384")) return 1;
  if (jdo("sp_build 0")) return 1;
  if (jdo("sp_train 1500")) return 1;
  if (jdo("sp_acc 0")) return 1;
  if (jget_scalar("spacc", &acc)) return 1;
  printf("vendor: train accuracy %.4f\n", acc);

  /* Serve predictions: pick points, run the forward pass, read logits.
     The point lives J-side; the app just names it. */
  const char* probes[] = { "0.1 0.1", "0.7 0.5", "0.5 0.7", "0.9 0.05" };
  for (int i = 0; i < 4; i++) {
    snprintf(sentence, sizeof sentence, "sp_probe %s", probes[i]);
    if (jdo(sentence)) return 1;
    /* sp_plogits is a 2-element FL vector: pull each element by name */
    if (jdo("sp_lg0=: 0 { sp_plogits")) return 1;
    if (jdo("sp_lg1=: 1 { sp_plogits")) return 1;
    if (jget_scalar("sp_lg0", &p0)) return 1;
    if (jget_scalar("sp_lg1", &p1)) return 1;
    printf("vendor: probe %-9s -> class %d  (logits %.3f %.3f)\n",
           probes[i], p1 > p0 ? 1 : 0, p0, p1);
  }

  int ok = acc >= 0.95;
  printf("vendor: %s\n", ok ? "PASS" : "FAIL");
  JFree(jt);
  return ok ? 0 : 1;
}
