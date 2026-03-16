"""
Entry point for: python -m ppm_preprocessing.webapp

On Windows the default file-encoding is cp1252 which cannot handle
characters like U+2713 that libraries (FLAML, sklearn) may write.
We fix this by re-launching the interpreter with  -X utf8  so that
ALL open() / print() / to_csv() calls default to UTF-8.
"""
import os
import sys


def _running_utf8() -> bool:
    """Return True if the interpreter is already in UTF-8 mode."""
    return getattr(sys.flags, "utf8_mode", 0) == 1


if sys.platform == "win32" and not _running_utf8():
    # Re-exec ourselves under UTF-8 mode.
    # Guard with an env-var so we don't loop forever.
    if not os.environ.get("_PPM_UTF8_RESTARTED"):
        os.environ["_PPM_UTF8_RESTARTED"] = "1"
        import subprocess

        rc = subprocess.call(
            [sys.executable, "-X", "utf8", "-m", "ppm_preprocessing.webapp"]
            + sys.argv[1:]
        )
        sys.exit(rc)

# Either we're on Linux/Mac, or we've been re-launched with -X utf8.
from ppm_preprocessing.webapp.app import main  # noqa: E402

main()
