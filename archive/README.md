# archive/ — Phase A indexes

Generated against git HEAD `97b6668` on branch `cleanup/2026-05-13-archive`.
Phase A is read-only: no source files were moved, deleted, or edited.

## Files

- `runs.csv` — one row per ledger exp + orphan run dirs (filesystem ∖ ledger).
- `comparison_baselines.csv` — comparison methods (BatVision/EchoNet/EchoDiffusion family).
- `delete_candidates.csv` — items proposed for later pruning, with `safe_to_delete` flag.
- `artifacts.csv` — orphan checkpoints/results/logs not matched to any ledger row.
- `_phase_a.py` — generator (temporary; itself listed in delete_candidates).

## Run row counts

- total rows: 484 (ledger=371, orphan run-dirs=113)
- status: keep=113 archive=21 delete_candidate=228 deprecated=122
- era:    radial=26 pre-radial=458 unknown=0
- delete_candidates rows: total=930 safe_yes=118 safe_no=812
- artifacts rows: 261

## Keep-as-code paths (never proposed for deletion)

- `models/unet_foa.py`
- `models/bin_based/**`
- `models/batvision/**`
- `models/echonet/**`
- `models/echodiffusion/**`
- `models/pretrain/**`

## kept-by-ledger-perf

Reserved for unet_foa-family variants that the ledger shows beat the unet_foa baseline.
Phase A does not auto-detect these; populate during Phase B review.

## Status decision order

1. Status starts with `PLANNED / NOT REPORTED` → `deprecated`.
2. comparison_baseline + radial-era → `keep`.
3. pre-radial era → `delete_candidate` (user rule: non-radial 실험은 전부 삭제).
4. ledger Keep? contains `KEEP` + radial-era → `archive`.
5. radial w/ explicit Failure Mode text → `delete_candidate`.
6. fallback → `keep` (conservative).

## Notes

- `layout/` is out of scope and never indexed.
- `.gitignore` excludes checkpoints/, results/, eval/, outputs/, wandb/, *.pt, *.pth.
- Pre-radial run artifacts are listed with `safe_to_delete=no` pending per-row review.
