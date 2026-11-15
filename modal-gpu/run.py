"""Run any command from this repo on a Modal GPU and copy its outputs back.

    modal run modal-gpu/run.py --workdir myproto --cmd "python train.py"

See modal-gpu/README.md for setup and options.
"""

import io
import os
import tarfile
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/repo"

# Add packages your code needs here. torch from PyPI ships with CUDA.
PACKAGES = [
    "torch",
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "numba",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*PACKAGES)
    .env({"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        REPO,
        REMOTE_REPO,
        ignore=[
            ".git/**",
            "**/.venv/**",
            "venv/**",
            "**/__pycache__/**",
            "**/node_modules/**",
            "**/*.fbx",
        ],
    )
)

app = modal.App("junction-signs-gpu", image=image)


@app.function(gpu="T4", timeout=60 * 60)
def run(cmd: str, workdir: str, outputs: list[str]) -> bytes:
    import subprocess

    cwd = Path(REMOTE_REPO) / workdir
    env = {**os.environ, "PYTHONPATH": str(cwd)}
    subprocess.run(["nvidia-smi", "-L"], check=False)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=env)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in outputs:
            if (cwd / d).exists():
                tar.add(cwd / d, arcname=d)
    return buf.getvalue()


@app.local_entrypoint()
def main(cmd: str, workdir: str = ".", gpu: str = "T4", outputs: str = "models,results"):
    out_list = [o.strip() for o in outputs.split(",") if o.strip()]
    fn = run if gpu == "T4" else run.with_options(gpu=gpu)
    payload = fn.remote(cmd, workdir, out_list)

    dest = REPO / workdir
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        names = sorted({n.split("/")[0] for n in tar.getnames()})
        tar.extractall(dest, filter="data")
    print(f"Copied back into {dest}: {names or 'nothing'}")
