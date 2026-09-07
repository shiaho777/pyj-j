# Tally — repo entry points. Run from the pyj/ directory.
# Engine env vars are needed by every target that touches libj.

export PYJ_LIBPATH := $(CURDIR)/jlibrary/bin
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  export DYLD_LIBRARY_PATH := $(PYJ_LIBPATH)
else
  export LD_LIBRARY_PATH := $(PYJ_LIBPATH)
endif

.PHONY: tally-test demo shootout cards vendor host all

# Full Tally test suite: unit + MCP stdio integration (73 checks)
tally-test:
	python3 test_tally.py

# Money-checker demo over a real MCP handshake
demo:
	python3 demo_tally.py

# Regenerate the float64-vs-exact case table
shootout:
	python3 recon/shootout.py

# Regenerate the caught-in-the-act cards (README/issue content)
cards:
	python3 recon/cards.py

# Route 2: C host driving libj through the raw ABI
host:
	cc -O2 host/train.c -o host/train -Ijlibrary/bin -Ljlibrary/bin -lj
	cd .. && DYLD_LIBRARY_PATH=pyj/jlibrary/bin ./pyj/host/train jlibrary/bin pyj

# Route 3: zero-build vendor demo
vendor:
	$(MAKE) -C vendor run

# Full gate: tests + shootout + cards + host + vendor
all: tally-test shootout cards host vendor
