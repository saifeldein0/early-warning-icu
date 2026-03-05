# Public Release Notes

This repository is a public-safe packaging of the project for portfolio and fellowship review.

## Removed from the public release

- real clinical datasets and exports
- private uploads and spreadsheet extracts
- trained model weights and preprocessing binaries
- internal deployment notes, credentials, and integration details
- manuscript build outputs and internal-only paper assets
- repetitive figure directories, slide-support material, and exploratory cleanup scripts

## Kept in the public release

- the core research and experimentation scripts
- aggregate result summaries and public-safe figures
- the dashboard source code
- a runnable synthetic-data demo for the website
- a curated `results/` folder that collects the retained public artifacts in one place

## Expected limitations

- Many training and validation scripts require private data that is not distributed here.
- The website demo uses a deterministic heuristic scoring function instead of the private trained models.
- Paths and operational instructions are intentionally simplified for public use.
- Some scripts still reference legacy output paths from the private environment; the committed public artifacts are organized under `results/` instead.
