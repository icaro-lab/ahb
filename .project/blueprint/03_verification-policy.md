# Verification Policy

- Verification modality must match the type of change.
- Repository/documentation changes require file inventory review and Python syntax checks for scripts.
- Prompt-template changes require JSON validity checks.
- Dataset export changes require running `python scripts/build_hf_dataset.py --source ../adv-humanities-bench/benchmark/benchmark.csv` and inspecting `hf_dataset/dataset_summary.json`.
- Hugging Face publication changes require `hf auth whoami`, upload verification, and a `datasets.load_dataset` smoke check after publication.
- Browser evidence is not required unless a future website or rendered documentation surface is added.
