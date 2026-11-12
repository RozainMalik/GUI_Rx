import subprocess
import sys

from benchev import ROOT


def test_help_lists_subcommands():
    out = subprocess.run([sys.executable, "-m", "benchev.cli", "-h"], capture_output=True, text=True, cwd=ROOT).stdout
    for c in ("run", "ingest", "report"):
        assert c in out


def test_expand_benches():
    from benchev.cli import expand_benches
    assert expand_benches("core") == ["ssv2", "sspro", "medspot"]
    assert len(expand_benches("suite")) == 8 and expand_benches("oswg") == ["oswg"]
