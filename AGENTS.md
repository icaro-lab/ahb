# AHB AGENTS Guide

Scope: applies to this repository.

## Repo Model

- This is the public release repository for the Adversarial Humanities Benchmark.
- Keep the committed surface close to the HLE public benchmark pattern: README, license, citation, requirements, images, evaluation scripts, and concise public prompt-template examples.
- Do not commit raw private experiment runs, model responses, judge outputs, local credentials, or generated Hugging Face staging payloads.
- `hf_dataset/` is local-only staging for the Hugging Face dataset repository and is ignored by git.

## Dataset Publication

- Hugging Face dataset target: `icaro-lab/ahb`.
- Public split name: `test`.
- Default public cap: at most 600 prompts per method.
- Enforce prompt row caps and word limits before upload.

## Verification

- For repo/documentation changes, run markdown/file checks and script syntax checks.
- For dataset export changes, run `python scripts/build_hf_dataset.py --source <source benchmark.csv>` and inspect `hf_dataset/dataset_summary.json`.
- For Hugging Face publication, verify `hf auth whoami`, upload dry-run where practical, then verify `datasets.load_dataset("icaro-lab/ahb", split="test")` after publication.
