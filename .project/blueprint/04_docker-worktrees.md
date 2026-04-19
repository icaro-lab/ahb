# Docker Worktrees

- This repo does not require Docker for normal development or publication.
- If a future workflow adds Docker, default to one active local stack across worktrees.
- Any future parallel Docker mode must define Compose project names, host port isolation, env/override strategy, and service/container naming rules before use.
