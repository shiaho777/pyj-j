# vendor/Makefile -- the whole "install" for a downstream application.
# The repo's build.sh `vendor` target fills this directory (libj, profile,
# kernel scripts); then `make` builds the app and `./vendor_demo` runs it.

CC ?= cc
CFLAGS = -O2

UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
  RUNENV = DYLD_LIBRARY_PATH=.
else
  RUNENV = LD_LIBRARY_PATH=.
endif

vendor_demo: app.c
	$(CC) $(CFLAGS) app.c -o vendor_demo -I. -L. -lj

run: vendor_demo
	$(RUNENV) ./vendor_demo

clean:
	rm -f vendor_demo

.PHONY: run clean
