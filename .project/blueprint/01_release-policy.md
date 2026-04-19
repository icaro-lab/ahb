# Release Policy

- `dev` is the integration branch.
- `main` is the stable branch.
- Implementation PRs merge to `dev`.
- Release PRs merge from `dev` to `main`.
- GitHub release readiness requires script syntax checks and README/HF link review.
- Hugging Face release readiness requires a fresh `hf_dataset/` export, row-cap validation, `hf auth whoami`, upload, and a post-upload `datasets.load_dataset("icaro-lab/ahb", split="test")` smoke check.
- Branch protection is pending until the public GitHub repo exists and CI names are stable.
