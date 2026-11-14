# Training on a Modal GPU

`run.py` runs any command from this repo on a cloud GPU and copies the outputs back to your machine. Everyone at Junction gets $100 of Modal credit, and a T4 costs well under $1 an hour.

## One-time setup (about 5 minutes)

1. Sign up at <https://modal.com>.
2. Redeem the Junction credits with code `VVX-2QX-8HW`: <https://modal.fillout.com/t/qMXCmRGseUus?code=VVX-2QX-8HW>
3. Add a payment method under Settings → Billing in the Modal dashboard. **GPU runs fail without one** ("Please add a payment method to use T4 GPU functions"), even when you have credits. Usage is drawn from the credits first.
4. Install the CLI and log in:

   ```sh
   uv tool install modal      # or: pipx install modal / pip install modal
   modal setup                # opens a browser to log in
   ```

   If `uv tool` puts it in `~/.local/bin` and that isn't on your PATH, call `~/.local/bin/modal`.

## Running

From the repo root:

```sh
modal run modal-gpu/run.py --workdir <your-folder> --cmd "python train.py"
```

| Option | Default | Meaning |
|---|---|---|
| `--cmd` | (required) | Shell command to run |
| `--workdir` | `.` | Folder (relative to the repo root) to run the command in. It's also put on `PYTHONPATH`. |
| `--gpu` | `T4` | `T4`, `L4`, `A10`, `L40S`, `A100`, `H100`… A T4 or L4 is plenty for small models |
| `--outputs` | `models,results` | Comma-separated folders inside `--workdir` to copy back afterwards |

How it works:

- The whole repo (without `.git`, `.venv` and the `.fbx` files) is uploaded to `/repo` in the container, so relative paths such as `../data` work as they do locally. Only changed files are re-uploaded.
- Your code needs `device = "cuda" if torch.cuda.is_available() else "cpu"` to actually use the GPU.
- Packages come from `PACKAGES` in `run.py`. Add what you need there. The first run, or the first after a package change, builds the image in a few minutes. Later runs start in seconds.
- The time limit is 1 hour per run. Change `timeout` in `run.py` if you need more.
- Each person uses their own Modal account and credits. See the dashboard at <https://modal.com/apps> for logs and spend.
