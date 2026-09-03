# vendor/ -- what a downstream application receives

This folder is the ship story from VISION.md: an application that embeds
the J engine as its tensor kernel, with **no jsource checkout, no Python,
and no engine build step**.

Everything in here is what the repo's build.sh produces at the end of a
normal build: the engine binary, the two kernel scripts it needs, and one
demo app. `vendor/Makefile` shows how an app builds against a prebuilt
libj with one line.

    libj.dylib / libj.so   the J engine (~5 MB; macOS copy also ships libgmp)
    profile.ijs            engine bootstrap script (required by JInit2 lookups)
    kernel/ad.ijs          tape AD core (pure J, no stdlib)
    kernel/train.ijs       spiral dataset + MLP trainer (pure J)
    app.c                  demo: train in-process, then serve predictions
    Makefile               `make && ./vendor_demo` -- that is the whole install

Run:

    make
    DYLD_LIBRARY_PATH=. ./vendor_demo        # macOS
    LD_LIBRARY_PATH=.    ./vendor_demo       # Linux

Expected output ends with `vendor: PASS`. The app trains the classifier
(1500 GD steps, ~1 s) and then runs a few in-process predictions, reading
results back through JGetM. All numerics stay inside libj; app.c is ~100
lines of C that never sees a matrix.
