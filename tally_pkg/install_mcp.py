"""
tally-mcp-install -- wire Tally into an MCP host with one command.

  tally-mcp-install claude      # Claude Desktop
  tally-mcp-install cursor      # Cursor
  tally-mcp-install generic     # print the JSON snippet, touch nothing

Writes (or merges) the `tally` server entry into the host's config.
The engine is bundled in the wheel, so no env vars are needed.
Existing config is preserved; a timestamped backup is made before writing.
"""
import json
import os
import shutil
import sys
import time


def _server_entry():
    # absolute path to the console script installed alongside this package
    exe = shutil.which("tally-mcp")
    if exe is None:
        exe = os.path.join(os.path.dirname(sys.executable), "tally-mcp")
    return {"command": exe, "args": []}


CONFIG_PATHS = {
    "claude": {
        "darwin": "~/Library/Application Support/Claude/claude_desktop_config.json",
        "linux": "~/.config/Claude/claude_desktop_config.json",
        "win32": "~/AppData/Roaming/Claude/claude_desktop_config.json",
    },
    "cursor": {
        "darwin": "~/.cursor/mcp.json",
        "linux": "~/.cursor/mcp.json",
        "win32": "~/.cursor/mcp.json",
    },
}


def _snippet():
    return {"mcpServers": {"tally": _server_entry()}}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    host = argv[0] if argv else "generic"

    if host == "generic":
        print(json.dumps(_snippet(), indent=2))
        print("\n# merge the above into your MCP host's config file",
              file=sys.stderr)
        return 0

    if host not in CONFIG_PATHS:
        print(f"unknown host {host!r}; choose from: "
              f"{', '.join(CONFIG_PATHS)}, generic", file=sys.stderr)
        return 2

    plat = sys.platform
    if plat not in CONFIG_PATHS[host]:
        plat = "linux"
    path = os.path.expanduser(CONFIG_PATHS[host][plat])

    cfg = {}
    if os.path.isfile(path):
        try:
            with open(path) as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"cannot parse existing config {path}: {e}", file=sys.stderr)
            return 1
        backup = f"{path}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(path, backup)
        print(f"backup: {backup}", file=sys.stderr)

    servers = cfg.setdefault("mcpServers", {})
    if "tally" in servers:
        print("tally already configured; updating entry", file=sys.stderr)
    servers["tally"] = _server_entry()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"wrote {path}", file=sys.stderr)
    print("restart the host to load Tally", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())