"""Run Phase 1 with one model server alive at a time (fits a 16GB GPU)."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import hydra
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[2]


def wait_ready(proc: subprocess.Popen, url: str, timeout: int = 600) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Model server exited with code {proc.returncode}; check its log.")
        try:
            with urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Model server not ready: {url}")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


@hydra.main(config_path="../cfg", config_name="recons", version_base=None)
def main(cfg: DictConfig) -> None:
    for phase in ("gsam", "hamer"):
        port = urlparse(cfg.servers[phase]).port
        log_path = ROOT / "omnidexgrasp" / "log" / f"{phase}-sequential.log"
        print(f"[{phase}] starting server (log: {log_path})", flush=True)
        with log_path.open("a") as log:
            proc = subprocess.Popen(
                ["bash", str(ROOT / "scripts/run_reconstruction.sh"), phase, f"server.port={port}"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        try:
            wait_ready(proc, f"http://127.0.0.1:{port}/openapi.json")
            subprocess.run(
                [sys.executable, "-m", "recons.client", *sys.argv[1:], f"phase={phase}"],
                cwd=ROOT / "omnidexgrasp", check=True,
            )
        finally:
            stop_server(proc)
            print(f"[{phase}] server stopped", flush=True)


if __name__ == "__main__":
    main()
