/*
 * host/train.c -- a non-Python host driving the J engine as an embeddable
 * tensor kernel.
 *
 * Everything numeric -- dataset, MLP forward, backprop, the SGD updates --
 * happens inside the J engine (ad.ijs + host/train.ijs). The C program only
 * sequences the ABI:
 *
 *   JInit2(libpath)   load the engine
 *   JSM(callbacks)    register the output callback
 *   JDo               load ad.ijs, load train.ijs, sp_data, sp_build,
 *                     sp_train (one call runs 1500 GD steps J-side),
 *                     sp_acc
 *   JGetM("spacc")    read the scalar accuracy back
 *
 * Build (see build.sh): cc -O2 host/train.c -o host/train -I<libdir> -L<libdir> -lj
 * Run:                  DYLD_LIBRARY_PATH=... ./host/train jlibrary/bin pyj
 *
 * The host keeps zero math of its own: no numpy, no Python, no BLAS. This
 * is the "ship story" workload from VISION.md -- an application vendor
 * links libj (~5 MB) and gets a trained classifier from a 100-line host.
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

/* Output callback: pass engine text through to stdout (errors included). */
static void host_output(void* jt, int type, char* s)
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
  if (rc != 0)
    fprintf(stderr, "host: JDo failed rc=%d: %s\n", rc, sentence);
  return rc;
}

static double jget_scalar(const char* name)
{
  long long jtype = 0, jrank = 0;
  long long *jshape = NULL, *jdata = NULL;
  if (JGetM(jt, (char*)name, &jtype, &jrank, &jshape, &jdata) != 0 ||
      jrank != 0 || !(jtype & 8LL /* FL */)) {
    fprintf(stderr, "host: %s is not a scalar float\n", name);
    return -1.0;
  }
  return *(double*)jdata;
}

int main(int argc, char* argv[])
{
  const char* libpath = (argc > 1) ? argv[1] : "jlibrary/bin";
  const char* scriptdir = (argc > 2) ? argv[2] : "pyj";
  char sentence[1024];

  jt = JInit2((char*)libpath);
  if (!jt) { fprintf(stderr, "host: JInit2 failed\n"); return 1; }
  void* callbacks[5] = {host_output, 0, 0, 0, (void*)3L};
  JSM(jt, callbacks);

  /* Paths are relative to the repo root; the host takes no part in the
     numerics, so it also does no path massaging beyond these loads. */
  snprintf(sentence, sizeof sentence, "0!:0 <'%s/ad.ijs'", scriptdir);
  if (jdo(sentence)) return 1;
  snprintf(sentence, sizeof sentence, "0!:0 <'%s/host/train.ijs'", scriptdir);
  if (jdo(sentence)) return 1;

  if (jdo("sp_seed 42")) return 1;       /* reproducible runs */
  if (jdo("sp_data 384")) return 1;      /* 768 two-spiral points */
  if (jdo("sp_build 0")) return 1;       /* fresh 2->16->16->2 MLP */

  if (jdo("sp_train 1500")) return 1;    /* all 1500 GD steps inside J */
  if (jdo("sp_acc 0")) return 1;

  double acc = jget_scalar("spacc");
  printf("host: final train accuracy %.4f\n", acc);
  int ok = acc >= 0.95;
  printf("host: %s (threshold 0.95)\n", ok ? "PASS" : "FAIL");

  JFree(jt);
  return ok ? 0 : 1;
}
