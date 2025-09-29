import subprocess
import sys
from pathlib import Path


def test_run_cli_module_without_install(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("Hello world. Test document.\nAnother line.", encoding="utf-8")

    # Run the module with a single interaction: ask a question then exit
    cmd = [
        sys.executable,
        "-m",
        "src.rag.cli.main",
        "chat",
        "--file",
        str(sample),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        out, _ = proc.communicate(input="hello?\n/exit\n", timeout=15)
    finally:
        proc.kill()

    assert "Indexing file:" in out
    assert "Top Context" in out
