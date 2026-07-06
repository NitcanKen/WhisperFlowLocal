"""Entrypoint: python -m whisperflow_local [--selftest]

--selftest constructs the menu-bar app (menu, config, history DB) without
entering the GUI run loop or loading the ASR model, prints the menu tree,
and exits 0. Used by CI-style checks; normal runs take no arguments.
"""
import os
import sys

if "--selftest" in sys.argv:
    os.environ["WFL_SELFTEST"] = "1"

from .app import WhisperFlowApp, main  # noqa: E402


def selftest() -> int:
    app = WhisperFlowApp()
    print(f"app title: {app.title!r}")
    print("menu items:")
    for key in app.menu.keys():
        print(f"  - {key}")
    print("selftest OK: rumps app constructed with full menu")
    return 0


if __name__ == "__main__":
    if os.environ.get("WFL_SELFTEST"):
        sys.exit(selftest())
    main()
