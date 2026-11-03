# ABOUTME: Downloads the base chat model and the Taboo LoRA adapter into the shared HF cache.
# ABOUTME: Run once before the first audit trial.
import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

REPOS = ["google/gemma-2-9b-it", "bcywinski/gemma-2-9b-it-taboo-gold"]

for repo in REPOS:
    print(f"downloading {repo} ...", flush=True)
    path = snapshot_download(repo_id=repo, token=os.environ["HF_TOKEN"])
    print(f"  -> {path}", flush=True)
