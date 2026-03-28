import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError


def _fmt_size(path: Path) -> str:
    size = path.stat().st_size
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size >= 1024 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.2f}{units[i]}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and place tokenizer.model for a local HF snapshot."
    )
    parser.add_argument(
        "--snap",
        type=Path,
        required=True,
        help="Path to snapshot directory, e.g. .../snapshots/<commit>",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Hugging Face repo id.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="HF access token. If omitted, use HF_TOKEN env var.",
    )
    args = parser.parse_args()

    snap = args.snap.resolve()
    snap.mkdir(parents=True, exist_ok=True)

    dst = snap / "tokenizer.model"
    if dst.exists():
        print(f"tokenizer.model already exists: {dst}")
    else:
        try:
            local_file = hf_hub_download(
                repo_id=args.repo_id,
                filename="original/tokenizer.model",
                local_dir=str(snap),
                token=args.token,
            )
        except GatedRepoError as exc:
            raise SystemExit(
                "Cannot access gated repo. Run `huggingface-cli login` with an account "
                "that has access, or pass `--token <HF_TOKEN>`."
            ) from exc

        src = Path(local_file)
        shutil.copy2(src, dst)

    print(f"Copied tokenizer to: {dst}")
    print(f"tokenizer.model: {_fmt_size(dst)}")

    model_path = snap / "model.pth"
    if model_path.exists():
        print(f"model.pth: {_fmt_size(model_path)}")
    else:
        print(f"model.pth not found: {model_path}")


if __name__ == "__main__":
    main()
