"""Run any vgp command on a Modal GPU and copy models/ and results/ back.

    modal run modal_gpu.py --cmd "python scripts/train.py"
    modal run modal_gpu.py --cmd "python scripts/eval.py" --gpu L4

The container mirrors the repo layout (/repo/data, /repo/vgp), so
signrec.data.REPO_ROOT resolves the same way it does locally. Code and data are
mounted fresh on every run (no rebuild needed after edits). Whatever the command
writes to vgp/models/ or vgp/results/ is copied back into the local vgp/ folder.
"""

import io
import tarfile
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUTPUT_DIRS = ("models", "results")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "numba",
    )
    .env({"PYTHONPATH": "/repo/vgp", "MPLBACKEND": "Agg"})
    .workdir("/repo/vgp")
    .add_local_dir(DATA, "/repo/data", ignore=["Test/**", "**/*.fbx"])
    .add_local_dir(
        HERE,
        "/repo/vgp",
        ignore=[".venv/**", "**/__pycache__/**", "models/**", "results/**", ".git/**"],
    )
)

app = modal.App("vgp-signrec", image=image)


@app.function(gpu="T4", timeout=60 * 60)
def run(cmd: str) -> bytes:
    import subprocess

    subprocess.run(["nvidia-smi", "-L"], check=False)
    subprocess.run(cmd, shell=True, check=True)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in OUTPUT_DIRS:
            if Path(d).exists():
                tar.add(d)
    return buf.getvalue()


@app.local_entrypoint()
def main(cmd: str, gpu: str = "T4"):
    fn = run if gpu == "T4" else run.with_options(gpu=gpu)
    payload = fn.remote(cmd)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        names = tar.getnames()
        tar.extractall(HERE, filter="data")
    print(f"Copied back into {HERE}: {sorted({n.split('/')[0] for n in names}) or 'nothing'}")
