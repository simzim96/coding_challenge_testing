import os
import subprocess
import sys
from pathlib import Path


def test_run_chat_script(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text(
        "This is a tiny document about arcs and curves.", encoding="utf-8"
    )

    cmd = [sys.executable, "scripts/chat.py", "--file", str(sample)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [env.get("PYTHONPATH", ""), str(Path.cwd())])
    )
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    try:
        out, _ = proc.communicate(input="what is this?\n/exit\n", timeout=15)
    finally:
        proc.kill()

    assert "Indexing file:" in out
    assert "Top Context" in out
