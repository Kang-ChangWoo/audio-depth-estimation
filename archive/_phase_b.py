#!/usr/bin/env python3
"""Phase B: move meaningful runs to archive/runs/<run_id>/.

Reads archive/runs.csv. For each row with status == 'archive':
- creates archive/runs/<run_id>/ with manifest.yaml, config.yaml, logs/, checkpoints/,
  results/, qualitative/, code_ref/
- moves tracked files (configs, logs) via `git mv`
- moves ignored files (checkpoints, results, qual dirs) via plain `mv`
"""
from __future__ import annotations
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/root/storage/implementation/shared_audio/baseline")
ARCHIVE_RUNS = REPO / "archive" / "runs"
MODELS = REPO / "models"

ARCHIVE_RUNS.mkdir(parents=True, exist_ok=True)


def run(cmd, check=True, **kw):
    return subprocess.run(cmd, cwd=str(REPO), check=check, capture_output=True, text=True, **kw)


def head_sha():
    return run(["git", "rev-parse", "HEAD"]).stdout.strip()


def yaml_quote(s: str) -> str:
    s = (s or "")
    # Use double-quoted form; escape backslash and quote
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def yaml_list(items):
    if not items:
        return "[]"
    return "[" + ", ".join(yaml_quote(i) for i in items) + "]"


def extract_metrics(notes: str) -> str:
    # notes is e.g. "reason | metrics=RMSE=...,AbsRel=..." — take portion after `metrics=`
    m = re.search(r"metrics=(.*)$", notes or "")
    return m.group(1).strip() if m else ""


# Map model_name -> models/ subpackage
def model_dir_for(model_name: str) -> Path | None:
    mn = (model_name or "").lower()
    # try to find subpackage name token in the model_name
    candidates = []
    for d in sorted(MODELS.iterdir()):
        if not d.is_dir() or d.name.startswith("__"):
            continue
        if d.name.lower() in mn:
            candidates.append(d)
    if candidates:
        # pick longest name match (most specific)
        candidates.sort(key=lambda p: len(p.name), reverse=True)
        return candidates[0]
    # try a few specific tokens
    if "echodiff" in mn or "echodiffusion" in mn:
        return MODELS / "echodiffusion"
    if "echonet" in mn:
        return MODELS / "echonet"
    if "batvision" in mn:
        return MODELS / "batvision"
    if "renew" in mn:
        return MODELS / "renew"
    if "echorange" in mn or "bin_based" in mn or "range_head" in mn:
        return MODELS / "bin_based"
    return None


def gather_model_sources(model_dir: Path | None) -> list[str]:
    if model_dir is None or not model_dir.exists():
        return []
    out = []
    for p in sorted(model_dir.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        out.append(str(p.relative_to(REPO)))
    return out


def is_tracked(path: Path) -> bool:
    if not path.exists():
        return False
    res = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(REPO))],
        cwd=str(REPO), capture_output=True, text=True,
    )
    return res.returncode == 0


def safe_git_mv(src: Path, dst: Path):
    """git mv with mkdir of parent dir."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel_src = src.relative_to(REPO)
    rel_dst = dst.relative_to(REPO)
    r = subprocess.run(["git", "mv", str(rel_src), str(rel_dst)], cwd=str(REPO), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git mv failed: {rel_src} -> {rel_dst}: {r.stderr}")


def safe_plain_mv(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # already moved; skip
        return
    shutil.move(str(src), str(dst))


def write_manifest(run_dir: Path, row: dict, extra_configs: list[str], missing_paths: list[str], source_files: list[str]):
    metrics = extract_metrics(row.get("notes", ""))
    # ledger_status_keep: best-effort - look at the part of notes before metrics
    reason = (row.get("notes") or "").split("|", 1)[0].strip()
    lines = []
    lines.append(f"run_id: {yaml_quote(row['run_id'])}")
    lines.append(f"model_name: {yaml_quote(row['model_name'])}")
    lines.append(f"role: {yaml_quote(row['role'])}")
    lines.append(f"era: {yaml_quote(row['era'])}")
    lines.append(f"git_commit: {yaml_quote(row['git_commit'])}")
    lines.append(f"ledger_status_keep: {yaml_quote(reason)}")
    lines.append(f"metrics_raw: {yaml_quote(metrics)}")
    lines.append(f"notes: {yaml_quote(row.get('notes',''))}")
    lines.append(f"extra_configs: {yaml_list(extra_configs)}")
    lines.append(f"missing_paths: {yaml_list(missing_paths)}")
    (run_dir / "manifest.yaml").write_text("\n".join(lines) + "\n")


def archive_row(row: dict, head: str) -> dict:
    """Returns stats dict."""
    rid = row["run_id"]
    run_dir = ARCHIVE_RUNS / rid
    # If multiple rows share rid (we saw exp410 twice), append _b
    if run_dir.exists():
        suffix = "b"
        while (ARCHIVE_RUNS / f"{rid}_{suffix}").exists():
            suffix = chr(ord(suffix) + 1)
        run_dir = ARCHIVE_RUNS / f"{rid}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "code_ref").mkdir(exist_ok=True)

    missing_paths: list[str] = []
    extra_configs: list[str] = []

    # Config(s)
    cfg_field = (row.get("config_path") or "").strip()
    cfg_list = [c.strip() for c in cfg_field.split(";") if c.strip()]
    moved_first_config = False
    for i, cfg_rel in enumerate(cfg_list):
        src = REPO / cfg_rel
        if not src.exists():
            missing_paths.append(cfg_rel)
            continue
        if i == 0:
            dst = run_dir / "config.yaml"
            if is_tracked(src):
                safe_git_mv(src, dst)
            else:
                safe_plain_mv(src, dst)
            moved_first_config = True
        else:
            extra_configs.append(cfg_rel)
            dst = run_dir / Path(cfg_rel).name
            try:
                if is_tracked(src):
                    safe_git_mv(src, dst)
                else:
                    safe_plain_mv(src, dst)
            except RuntimeError:
                missing_paths.append(cfg_rel)

    # Logs
    log_field = (row.get("log_path") or "").strip()
    for lp in [x.strip() for x in log_field.split(";") if x.strip()]:
        src = REPO / lp
        if not src.exists():
            missing_paths.append(lp)
            continue
        dst = run_dir / "logs" / Path(lp).name
        # Avoid filename collision
        if dst.exists():
            base = dst.stem
            ext = dst.suffix
            i = 1
            while (run_dir / "logs" / f"{base}__{i}{ext}").exists():
                i += 1
            dst = run_dir / "logs" / f"{base}__{i}{ext}"
        if is_tracked(src):
            safe_git_mv(src, dst)
        else:
            safe_plain_mv(src, dst)

    # Checkpoints (.gitignored - plain mv)
    ck_field = (row.get("checkpoint_path") or "").strip()
    ck_dir = run_dir / "checkpoints"
    for cp in [x.strip() for x in ck_field.split(";") if x.strip()]:
        src = REPO / cp
        if not src.exists():
            missing_paths.append(cp)
            continue
        ck_dir.mkdir(exist_ok=True)
        dst = ck_dir / Path(cp).name
        if dst.exists():
            i = 1
            while (ck_dir / f"{Path(cp).name}__{i}").exists():
                i += 1
            dst = ck_dir / f"{Path(cp).name}__{i}"
        safe_plain_mv(src, dst)

    # Results
    res_field = (row.get("result_path") or "").strip()
    res_dir = run_dir / "results"
    for rp in [x.strip() for x in res_field.split(";") if x.strip()]:
        src = REPO / rp
        if not src.exists():
            missing_paths.append(rp)
            continue
        res_dir.mkdir(exist_ok=True)
        dst = res_dir / Path(rp).name
        if dst.exists():
            i = 1
            while (res_dir / f"{Path(rp).name}__{i}").exists():
                i += 1
            dst = res_dir / f"{Path(rp).name}__{i}"
        safe_plain_mv(src, dst)

    # Qualitative
    q_field = (row.get("qualitative_path") or "").strip()
    q_dir = run_dir / "qualitative"
    for qp in [x.strip() for x in q_field.split(";") if x.strip()]:
        src = REPO / qp
        if not src.exists():
            missing_paths.append(qp)
            continue
        q_dir.mkdir(exist_ok=True)
        dst = q_dir / Path(qp).name
        if dst.exists():
            i = 1
            while (q_dir / f"{Path(qp).name}__{i}").exists():
                i += 1
            dst = q_dir / f"{Path(qp).name}__{i}"
        safe_plain_mv(src, dst)

    # code_ref
    git_commit = (row.get("git_commit") or "").strip() or head
    (run_dir / "code_ref" / "git_commit.txt").write_text(git_commit + "\n")
    model_dir = model_dir_for(row.get("model_name", ""))
    sources = gather_model_sources(model_dir)
    (run_dir / "code_ref" / "source_files.txt").write_text("\n".join(sources) + ("\n" if sources else ""))

    write_manifest(run_dir, row, extra_configs, missing_paths, sources)

    return {
        "run_id": rid,
        "run_dir": str(run_dir.relative_to(REPO)),
        "moved_config": moved_first_config,
        "missing_paths": missing_paths,
    }


def main():
    head = head_sha()
    with open(REPO / "archive" / "runs.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    archive_rows = [r for r in rows if r["status"] == "archive"]
    print(f"archive rows to process: {len(archive_rows)}")
    stats = []
    all_missing = []
    for row in archive_rows:
        s = archive_row(row, head)
        stats.append(s)
        if s["missing_paths"]:
            all_missing.extend([(s["run_id"], p) for p in s["missing_paths"]])
    print(f"created {len(stats)} run dirs")
    print(f"total missing paths: {len(all_missing)}")
    # Identify rows with ALL paths missing (no logs, configs, ckpts, results moved)
    fully_missing = []
    for s, row in zip(stats, archive_rows):
        rd = REPO / s["run_dir"]
        has_logs = any(rd.joinpath("logs").iterdir()) if rd.joinpath("logs").exists() else False
        has_ckpts = rd.joinpath("checkpoints").exists() and any(rd.joinpath("checkpoints").iterdir())
        has_results = rd.joinpath("results").exists() and any(rd.joinpath("results").iterdir())
        has_config = (rd / "config.yaml").exists()
        if not (has_logs or has_ckpts or has_results or has_config):
            fully_missing.append(row["run_id"])
    print(f"rows with NO artifacts moved: {len(fully_missing)}: {fully_missing}")


if __name__ == "__main__":
    main()
