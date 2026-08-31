"""
Download FT datasets: LatentQA + Classification datasets.

Usage:
    python download_ft.py --out-dir tmp --hf-cache tmp/huggingface
"""

import argparse
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------#
# Git-clone datasets
# ---------------------------------------------------------------------------#

def download_latentqa(out_dir: Path):
    _git_clone("https://github.com/aypan17/latentqa", out_dir / "latentqa")

def download_geometry_of_truth(out_dir: Path):
    _git_clone("https://github.com/saprmarks/geometry-of-truth", out_dir / "geometry-of-truth")

def download_relations(out_dir: Path):
    _git_clone("https://github.com/evandez/relations", out_dir / "relations")


def _git_clone(url: str, dest: Path):
    if dest.exists():
        print(f"Already exists: {dest}, skipping.")
        return
    subprocess.run(["git", "clone", "--depth=1", url, str(dest)], check=True)
    print(f"Cloned {url} → {dest}")


# --------------------------------------------------------------------------- #
# HuggingFace datasets
# --------------------------------------------------------------------------- #

def download_hf_datasets(hf_cache: Path):
    from datasets import load_dataset

    # ner and tense use local files in experiments/baseline/activation_oracles/datasets/
    hf_datasets = [
        ("stanfordnlp/sst2",          {}),
        ("facebook/md_gender_bias",    {"name": "funpedia", "trust_remote_code": True}),
        ("stanfordnlp/snli",           {}),
    ]
    for name, kwargs in hf_datasets:
        print(f"Downloading {name} ...")
        try:
            load_dataset(name, cache_dir=str(hf_cache), **kwargs)
            print(f"  Done: {name}")
        except Exception as e:
            print(f"  FAILED: {name}: {e}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",   default="tmp")
    parser.add_argument("--hf-cache",  default="tmp/huggingface")
    args = parser.parse_args()

    out_dir  = Path(args.out_dir)
    hf_cache = Path(args.hf_cache)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Git repos
    download_latentqa(out_dir)
    download_geometry_of_truth(out_dir)
    download_relations(out_dir)

    # HuggingFace
    download_hf_datasets(hf_cache)
