#!/usr/bin/env python3
"""Phase A index generator. Read-only over the repo; writes 5 files under archive/.

Generates:
  - runs.csv               (one row per ledger experiment + orphan run dirs)
  - comparison_baselines.csv
  - delete_candidates.csv
  - artifacts.csv
  - README.md
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path("/root/storage/implementation/shared_audio/baseline")
ARCHIVE = REPO / "archive"
LEDGER = REPO / "audio_depth_experiment_ledger.csv"

CHECKPOINTS = REPO / "checkpoints"
RESULTS = REPO / "results"
LOGS = REPO / "logs"
CONFIG = REPO / "config"
MODELS = REPO / "models"
QUAL = REPO / "qual"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def short_head():
    out = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).decode().strip()
    return out


def du_sh(path: Path) -> str:
    if not path.exists():
        return "0"
    try:
        out = subprocess.check_output(["du", "-sh", str(path)], stderr=subprocess.DEVNULL).decode()
        return out.split("\t")[0].strip()
    except subprocess.CalledProcessError:
        return "?"


def trim(s: str, n: int) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Discover filesystem run-dirs
# ---------------------------------------------------------------------------
EXP_RE = re.compile(r"(?:^|[_/])(?:r2_)?(exp\d+)(?:[_/]|$)")


def list_top(p: Path):
    if not p.exists():
        return []
    return sorted([x for x in p.iterdir() if x.is_dir()])


def list_log_files(p: Path):
    out = []
    if not p.exists():
        return out
    for root, dirs, files in os.walk(p):
        # Skip code_snapshot heavy dir for log mapping but keep one entry
        for f in files:
            if f.endswith((".log", ".out")):
                out.append(Path(root) / f)
    return out


CKPT_DIRS = list_top(CHECKPOINTS)
RESULT_DIRS = list_top(RESULTS)
LOG_DIRS = list_top(LOGS)
LOG_FILES = list_log_files(LOGS)

# Map exp_id -> list of paths
ckpt_by_exp: dict[str, list[Path]] = defaultdict(list)
res_by_exp: dict[str, list[Path]] = defaultdict(list)
log_by_exp: dict[str, list[Path]] = defaultdict(list)
qual_by_exp: dict[str, list[Path]] = defaultdict(list)


def index_dirs(dirs, table):
    for d in dirs:
        m = re.search(r"exp(\d+)", d.name)
        if m:
            eid = "exp" + m.group(1)
            table[eid].append(d)


index_dirs(CKPT_DIRS, ckpt_by_exp)
index_dirs(RESULT_DIRS, res_by_exp)

for f in LOG_FILES:
    m = re.search(r"exp(\d+)", f.name)
    if m:
        log_by_exp["exp" + m.group(1)].append(f)

if QUAL.exists():
    for q in QUAL.iterdir():
        m = re.search(r"exp(\d+)", q.name)
        if m:
            qual_by_exp["exp" + m.group(1)].append(q)


# ---------------------------------------------------------------------------
# Config matching
# ---------------------------------------------------------------------------
CONFIG_FILES = sorted([p for p in CONFIG.glob("*.yaml")])
CONFIG_STEMS = {p.stem: p for p in CONFIG_FILES}


def best_config_match(model_field: str, exp_name: str) -> Path | None:
    """Best-effort config filename match for a ledger row."""
    text = f"{model_field} {exp_name}".lower()
    # Look for "config=xxx" hint
    m = re.search(r"config=([a-z0-9_]+)", text)
    if m and m.group(1) in CONFIG_STEMS:
        return CONFIG_STEMS[m.group(1)]
    # Try direct stem match
    for stem, p in CONFIG_STEMS.items():
        if stem in text:
            return p
    # Try removing common suffixes from exp_name
    name = exp_name.lower()
    # Drop trailing lr/bs/lsh tokens
    name_tokens = re.split(r"[_/]", name)
    # Build candidates by progressively dropping trailing tokens
    for k in range(len(name_tokens), 1, -1):
        cand = "_".join(name_tokens[:k])
        if cand in CONFIG_STEMS:
            return CONFIG_STEMS[cand]
    return None


# ---------------------------------------------------------------------------
# Era / role classification
# ---------------------------------------------------------------------------
RADIAL_HINT = re.compile(r"\bradial\b|\bbin[-_ ]?based\b|\bechorange\b|\bhaz\b|\brange_head\b|R[34]_\d", re.I)
COMPARISON_PREFIXES = (
    "batvision", "echonet", "echodiffusion", "echodiff",  # echodiff_sh_side too
)
PRETRAIN_HINT = re.compile(r"pretrain|backbone", re.I)
ABLATION_HINT = re.compile(r"ablation|oracle|UB|upper bound|nokl", re.I)


def classify_era(model_field: str, exp_name: str, target_field: str, config_path: Path | None) -> str:
    blob = " ".join([model_field or "", exp_name or "", target_field or "", config_path.name if config_path else ""])
    if RADIAL_HINT.search(blob):
        return "radial"
    return "pre-radial"


def classify_role(model_field: str, exp_name: str, purpose: str) -> str:
    text = f"{model_field} {exp_name} {purpose}".lower()
    if any(pre in text for pre in ("batvision",)):
        return "comparison_baseline"
    if "echonet" in text:
        return "comparison_baseline"
    if "echodiff" in text or "echodiffusion" in text:
        # echodiffusion variants are comparison_baselines per task spec
        return "comparison_baseline"
    if PRETRAIN_HINT.search(text):
        return "pretrained_backbone"
    if ABLATION_HINT.search(text):
        return "ablation"
    if "echorange" in text or "bin_based" in text or "range_head" in text:
        return "comparison_baseline"
    return "ours"


# ---------------------------------------------------------------------------
# Parse ledger
# ---------------------------------------------------------------------------
ledger_rows: list[dict] = []
parse_errors: list[str] = []
with LEDGER.open(newline="") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        try:
            ledger_rows.append(row)
        except Exception as e:
            parse_errors.append(f"row {i}: {e}")

# Extract exp_id from "Exp ID / Name" field
def extract_exp_id(s: str) -> tuple[str, str]:
    """Returns (exp_id_or_empty, full_name)."""
    s = (s or "").strip()
    m = re.match(r"\s*(exp\d+)\b", s)
    eid = m.group(1) if m else ""
    # full name = after the slash if present, else s
    if "/" in s:
        full = s.split("/", 1)[1].strip()
    else:
        full = s
    return eid, full


# ---------------------------------------------------------------------------
# Build runs.csv rows
# ---------------------------------------------------------------------------
git_commit = short_head()
ledger_exp_ids: set[str] = set()

runs_rows = []
for r in ledger_rows:
    raw_id = r.get("Exp ID / Name", "")
    eid, full_name = extract_exp_id(raw_id)
    if not eid:
        # Fallback: try to extract anywhere
        m = re.search(r"exp(\d+)", raw_id)
        eid = ("exp" + m.group(1)) if m else raw_id.strip().split()[0]
    if eid.startswith("exp"):
        ledger_exp_ids.add(eid)
    status_cell = (r.get("Status") or "").strip()
    purpose = (r.get("Purpose") or "").strip()
    target = (r.get("Target") or "").strip()
    model_field = (r.get("Model / Change") or "").strip()
    keep_cell = (r.get("Keep?") or "").strip()
    fail_cell = (r.get("Failure Mode") or "").strip()
    metrics = (r.get("Metrics") or "").strip()

    cfg = best_config_match(model_field, full_name)
    era = classify_era(model_field, full_name, target, cfg)
    role = classify_role(model_field, full_name, purpose)

    # Status decision tree
    if status_cell.startswith("PLANNED / NOT REPORTED") or status_cell.startswith("PLANNED"):
        run_status = "deprecated"
        reason = "ledger PLANNED/NOT REPORTED"
    elif role == "comparison_baseline" and era == "radial":
        run_status = "keep"
        reason = "radial-era comparison baseline (representative TBD)"
    elif era == "pre-radial":
        run_status = "delete_candidate"
        reason = "pre-radial era; user rule: non-radial 실험은 전부 삭제"
    elif "KEEP" in keep_cell.upper() and era == "radial":
        run_status = "archive"
        reason = "ledger KEEP + radial-era"
    elif (
        era == "radial"
        and len(fail_cell) > 3
        and fail_cell.lower() not in ("none reported", "unknown", "n/a", "na")
        and "not completed" not in fail_cell.lower()
    ):
        run_status = "delete_candidate"
        reason = f"radial w/ explicit failure: {trim(fail_cell, 60)}"
    else:
        run_status = "keep"
        reason = "conservative default"

    ckpts = ckpt_by_exp.get(eid, [])
    res = res_by_exp.get(eid, [])
    logs = log_by_exp.get(eid, [])
    quals = qual_by_exp.get(eid, [])

    runs_rows.append(
        {
            "run_id": eid,
            "model_name": trim(model_field, 60),
            "role": role,
            "era": era,
            "config_path": str(cfg.relative_to(REPO)) if cfg else "",
            "log_path": ";".join(str(p.relative_to(REPO)) for p in logs),
            "checkpoint_path": ";".join(str(p.relative_to(REPO)) for p in ckpts),
            "result_path": ";".join(str(p.relative_to(REPO)) for p in res),
            "qualitative_path": ";".join(str(p.relative_to(REPO)) for p in quals),
            "git_commit": git_commit,
            "status": run_status,
            "notes": trim(f"{reason} | metrics={metrics}", 240),
        }
    )

# Append orphan run dirs (exp ids found on disk but not in ledger)
disk_exp_ids = set(ckpt_by_exp.keys()) | set(res_by_exp.keys()) | set(log_by_exp.keys())
orphan_ids = sorted(disk_exp_ids - ledger_exp_ids)
for eid in orphan_ids:
    ckpts = ckpt_by_exp.get(eid, [])
    res = res_by_exp.get(eid, [])
    logs = log_by_exp.get(eid, [])
    # Try to guess era from path names
    blob = " ".join([p.name for p in (ckpts + res + logs)])
    era = "radial" if RADIAL_HINT.search(blob) else ("pre-radial" if blob else "unknown")
    role = "unknown"
    runs_rows.append(
        {
            "run_id": eid,
            "model_name": "",
            "role": role,
            "era": era,
            "config_path": "",
            "log_path": ";".join(str(p.relative_to(REPO)) for p in logs),
            "checkpoint_path": ";".join(str(p.relative_to(REPO)) for p in ckpts),
            "result_path": ";".join(str(p.relative_to(REPO)) for p in res),
            "qualitative_path": "",
            "git_commit": git_commit,
            "status": "keep",
            "notes": "orphan run dir; not in ledger (conservative keep)",
        }
    )


# Write runs.csv
def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow(r)


RUN_FIELDS = [
    "run_id", "model_name", "role", "era", "config_path", "log_path",
    "checkpoint_path", "result_path", "qualitative_path", "git_commit",
    "status", "notes",
]
write_csv(ARCHIVE / "runs.csv", RUN_FIELDS, runs_rows)


# ---------------------------------------------------------------------------
# comparison_baselines.csv
# ---------------------------------------------------------------------------
COMPARISON_METHODS = [
    {
        "method_name": "BatVision",
        "code_path": "models/batvision/batvision.py",
        "config_glob": "batvision*",
    },
    {
        "method_name": "EchoNet",
        "code_path": "models/echonet/echonet.py",
        "config_glob": "echonet*",
    },
    {
        "method_name": "EchoDiffusion",
        "code_path": "models/echodiffusion/echodiffusion.py",
        "config_glob": "echodiffusion.yaml",
    },
    {
        "method_name": "EchoDiffusion-Ambi",
        "code_path": "models/echodiffusion/echodiffusion_ambi.py",
        "config_glob": "echodiffusion_ambi*",
    },
    {
        "method_name": "EchoDiffusion-Ambi-SH",
        "code_path": "models/echodiffusion/echodiffusion_ambi_sh.py",
        "config_glob": "echodiffusion_ambi_sh*",
    },
]


def find_related(prefix_tokens: list[str], base: Path) -> list[Path]:
    if not base.exists():
        return []
    out = []
    for d in base.iterdir():
        n = d.name.lower()
        for t in prefix_tokens:
            if n.startswith(t):
                out.append(d)
                break
    return out


comp_rows = []
for m in COMPARISON_METHODS:
    glob = m["config_glob"]
    cfgs = sorted(CONFIG.glob(glob))
    # Exclude superset matches: e.g. echodiffusion.yaml from echodiffusion_ambi*
    if m["method_name"] == "EchoDiffusion":
        cfgs = [c for c in cfgs if c.stem == "echodiffusion"]
    if m["method_name"] == "EchoDiffusion-Ambi":
        cfgs = [c for c in cfgs if c.stem.startswith("echodiffusion_ambi") and not c.stem.startswith("echodiffusion_ambi_sh")]
    if m["method_name"] == "EchoDiffusion-Ambi-SH":
        cfgs = [c for c in cfgs if c.stem.startswith("echodiffusion_ambi_sh")]

    # Tokens for matching artifact dir prefixes
    name_lower = m["method_name"].lower().replace("-", "_")
    tokens = []
    if name_lower == "batvision":
        tokens = ["batvision"]
    elif name_lower == "echonet":
        tokens = ["echonet"]
    elif name_lower == "echodiffusion":
        tokens = ["echodiffusion_soundspaces"]  # the bare model
    elif name_lower == "echodiffusion_ambi":
        tokens = ["echodiffusion_ambi_soundspaces", "echodiffusion_ambi_cide"]
    elif name_lower == "echodiffusion_ambi_sh":
        tokens = ["echodiffusion_ambi_sh", "echodiff_sh_side_plus", "echodiff_sh_side"]

    rel_logs = []
    for ld in LOG_DIRS:
        n = ld.name.lower()
        for t in tokens:
            if t.startswith(n) or n.startswith(t.split("_soundspaces")[0]):
                rel_logs.append(ld)
                break
    # Also search log files by tokens for echodiff
    rel_ckpts = find_related(tokens, CHECKPOINTS)
    rel_res = find_related(tokens, RESULTS)

    comp_rows.append({
        "method_name": m["method_name"],
        "code_path": m["code_path"],
        "config_path": ";".join(str(c.relative_to(REPO)) for c in cfgs),
        "related_logs": ";".join(str(p.relative_to(REPO)) for p in rel_logs) or f"count={len(rel_logs)}",
        "related_checkpoints": f"count={len(rel_ckpts)}",
        "related_results": f"count={len(rel_res)}",
        "status": "keep",
    })

write_csv(
    ARCHIVE / "comparison_baselines.csv",
    ["method_name", "code_path", "config_path", "related_logs", "related_checkpoints", "related_results", "status"],
    comp_rows,
)


# ---------------------------------------------------------------------------
# delete_candidates.csv
# ---------------------------------------------------------------------------
KEEP_AS_CODE_PREFIXES = [
    "models/unet_foa.py",
    "models/bin_based/",
    "models/batvision/",
    "models/echonet/",
    "models/echodiffusion/",
    "models/pretrain/",
]


def is_keep_as_code(rel: str) -> bool:
    for p in KEEP_AS_CODE_PREFIXES:
        if rel == p or rel.startswith(p):
            return True
    return False


del_rows = []

# 1. Pre-radial run artifacts (status=delete_candidate from runs)
for r in runs_rows:
    if r["status"] != "delete_candidate":
        continue
    rid = r["run_id"]
    for p_field, ptype in [("checkpoint_path", "checkpoint"), ("result_path", "result"), ("log_path", "log")]:
        v = r[p_field]
        if not v:
            continue
        for p in v.split(";"):
            del_rows.append({
                "path": p,
                "type": ptype,
                "reason": f"pre-radial era artifact for {rid}",
                "replacement_or_canonical_file": "",
                "safe_to_delete": "no",
                "notes": trim(r["notes"], 120),
            })

# 2. __pycache__ dirs (always safe)
pycache_dirs = []
for dirpath, dirnames, filenames in os.walk(REPO):
    rel = Path(dirpath).relative_to(REPO).as_posix()
    if rel.startswith("layout") or rel.startswith("intent-contract") or rel.startswith(".git"):
        dirnames[:] = []
        continue
    if "__pycache__" in dirnames:
        pycache_dirs.append((Path(dirpath) / "__pycache__").relative_to(REPO).as_posix())
    # Don't descend into pycache
    if "__pycache__" in dirnames:
        dirnames.remove("__pycache__")
for pc in pycache_dirs:
    del_rows.append({
        "path": pc,
        "type": "cache",
        "reason": "Python bytecode cache; always regenerable",
        "replacement_or_canonical_file": "",
        "safe_to_delete": "yes",
        "notes": "",
    })

# 3. Empty checkpoint dirs
for d in CKPT_DIRS:
    try:
        entries = list(d.iterdir())
    except OSError:
        continue
    is_empty = len(entries) == 0 or not any(p.is_file() and p.stat().st_size > 0 for p in d.rglob("*") if p.is_file())
    if is_empty:
        del_rows.append({
            "path": str(d.relative_to(REPO)),
            "type": "checkpoint",
            "reason": "empty checkpoint dir (no nonzero file)",
            "replacement_or_canonical_file": "",
            "safe_to_delete": "yes",
            "notes": "",
        })

# 4. Model .py files not imported by train.py / test.py / utils/train_utils.py
imp_grep_targets = [REPO / "train.py", REPO / "test.py", REPO / "utils" / "train_utils.py", REPO / "utils" / "test_utils.py"]
imp_text = ""
for p in imp_grep_targets:
    if p.exists():
        imp_text += p.read_text(errors="ignore") + "\n"

# Read all yaml config text too
cfg_text = ""
for c in CONFIG_FILES:
    cfg_text += c.read_text(errors="ignore") + "\n"

models_py: list[Path] = []
for p in MODELS.rglob("*.py"):
    if "__pycache__" in p.parts:
        continue
    if p.name == "__init__.py":
        continue
    models_py.append(p)

# Check if module path or stem is referenced
for p in models_py:
    rel = p.relative_to(REPO).as_posix()
    if is_keep_as_code(rel):
        continue
    stem = p.stem
    # Reference patterns
    dotted = p.relative_to(REPO).with_suffix("").as_posix().replace("/", ".")
    pkg_path = ".".join(p.relative_to(MODELS).with_suffix("").parts)  # e.g. "n9_0424.n9_0424"
    referenced = False
    for needle in (stem, dotted, pkg_path):
        if needle and needle in imp_text:
            referenced = True
            break
        if needle and needle in cfg_text:
            referenced = True
            break
    # Also check via models/__init__.py exports
    if not referenced:
        init_text = (MODELS / "__init__.py").read_text(errors="ignore")
        if stem in init_text or pkg_path in init_text:
            referenced = True
    if not referenced:
        del_rows.append({
            "path": rel,
            "type": "code",
            "reason": "no reference found in train/test/utils/configs/__init__",
            "replacement_or_canonical_file": "",
            "safe_to_delete": "no",
            "notes": "manual review required",
        })

# 5. Config YAML whose model.name is in a delete-candidate run list
# Gather model.name values referenced by delete_candidate runs
delete_model_names: set[str] = set()
for r in runs_rows:
    if r["status"] != "delete_candidate":
        continue
    if r["config_path"]:
        # parse model.name from that config
        cfgp = REPO / r["config_path"]
        if cfgp.exists():
            try:
                txt = cfgp.read_text(errors="ignore")
                m = re.search(r"model:\s*\n(?:[^\n]*\n)*?\s*name:\s*([A-Za-z0-9_\-]+)", txt)
                if m:
                    delete_model_names.add(m.group(1))
            except OSError:
                pass

# Mark config YAMLs whose model.name appears in delete list AND not in any keep run
keep_configs: set[str] = set()
for r in runs_rows:
    if r["status"] in ("keep", "archive") and r["config_path"]:
        keep_configs.add(r["config_path"])

for c in CONFIG_FILES:
    rel = c.relative_to(REPO).as_posix()
    if rel in keep_configs:
        continue
    try:
        txt = c.read_text(errors="ignore")
        m = re.search(r"model:\s*\n(?:[^\n]*\n)*?\s*name:\s*([A-Za-z0-9_\-]+)", txt)
        if not m:
            continue
        mn = m.group(1)
        if mn in delete_model_names:
            del_rows.append({
                "path": rel,
                "type": "config",
                "reason": f"model.name={mn} only used by delete_candidate runs",
                "replacement_or_canonical_file": "",
                "safe_to_delete": "no",
                "notes": "needs review",
            })
    except OSError:
        continue

# 6. The phase A script itself
del_rows.append({
    "path": "archive/_phase_a.py",
    "type": "code",
    "reason": "temporary scaffolding from Phase A; remove after index files are finalized",
    "replacement_or_canonical_file": "",
    "safe_to_delete": "no",
    "notes": "remove only when Phase B/C/D no longer needs to regenerate the indexes",
})

write_csv(
    ARCHIVE / "delete_candidates.csv",
    ["path", "type", "reason", "replacement_or_canonical_file", "safe_to_delete", "notes"],
    del_rows,
)


# ---------------------------------------------------------------------------
# artifacts.csv (orphan artifacts not matched to any ledger exp)
# ---------------------------------------------------------------------------
def has_ledger_match(name: str) -> bool:
    m = re.search(r"exp(\d+)", name)
    if not m:
        return False
    return ("exp" + m.group(1)) in ledger_exp_ids


orphan_paths: list[tuple[Path, str]] = []
for d in CKPT_DIRS:
    if not has_ledger_match(d.name):
        orphan_paths.append((d, "checkpoint"))
for d in RESULT_DIRS:
    if not has_ledger_match(d.name):
        orphan_paths.append((d, "result"))
for d in LOG_DIRS:
    # Logs are bins of multiple exp files; mark as orphan only if NONE of the files match
    if not d.is_dir():
        continue
    found = False
    for root, _, files in os.walk(d):
        for f in files:
            if has_ledger_match(f):
                found = True
                break
        if found:
            break
    if not found:
        orphan_paths.append((d, "log"))
# Also include top-level loose log files
for p in LOGS.iterdir() if LOGS.exists() else []:
    if p.is_file() and not has_ledger_match(p.name):
        orphan_paths.append((p, "log"))

# Batch du
artifact_rows = []
if orphan_paths:
    paths_str = [str(p) for p, _ in orphan_paths]
    # Run du in chunks of 200
    sizes = {}
    CHUNK = 200
    for i in range(0, len(paths_str), CHUNK):
        chunk = paths_str[i:i + CHUNK]
        try:
            out = subprocess.check_output(["du", "-sh", *chunk], stderr=subprocess.DEVNULL).decode()
            for line in out.splitlines():
                if "\t" in line:
                    sz, pth = line.split("\t", 1)
                    sizes[pth.strip()] = sz.strip()
        except subprocess.CalledProcessError as e:
            # Try one by one
            for ch in chunk:
                try:
                    line = subprocess.check_output(["du", "-sh", ch], stderr=subprocess.DEVNULL).decode().strip()
                    if "\t" in line:
                        sz, pth = line.split("\t", 1)
                        sizes[pth.strip()] = sz.strip()
                except Exception:
                    sizes[ch] = "?"
    for p, ptype in orphan_paths:
        m = re.search(r"exp(\d+)", p.name)
        matched = ("exp" + m.group(1)) if m else ""
        artifact_rows.append({
            "path": str(p.relative_to(REPO)),
            "type": ptype,
            "size_estimate": sizes.get(str(p), "?"),
            "matched_run_id": "",
            "notes": f"no ledger row matches (extracted exp_id={matched or 'none'})",
        })

write_csv(
    ARCHIVE / "artifacts.csv",
    ["path", "type", "size_estimate", "matched_run_id", "notes"],
    artifact_rows,
)


# ---------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------
status_counts = defaultdict(int)
era_counts = defaultdict(int)
for r in runs_rows:
    status_counts[r["status"]] += 1
    era_counts[r["era"]] += 1

safe_yes = sum(1 for r in del_rows if r["safe_to_delete"] == "yes")
safe_no = sum(1 for r in del_rows if r["safe_to_delete"] == "no")

readme_lines = [
    "# archive/ — Phase A indexes",
    "",
    f"Generated against git HEAD `{git_commit}` on branch `cleanup/2026-05-13-archive`.",
    "Phase A is read-only: no source files were moved, deleted, or edited.",
    "",
    "## Files",
    "",
    "- `runs.csv` — one row per ledger exp + orphan run dirs (filesystem ∖ ledger).",
    "- `comparison_baselines.csv` — comparison methods (BatVision/EchoNet/EchoDiffusion family).",
    "- `delete_candidates.csv` — items proposed for later pruning, with `safe_to_delete` flag.",
    "- `artifacts.csv` — orphan checkpoints/results/logs not matched to any ledger row.",
    "- `_phase_a.py` — generator (temporary; itself listed in delete_candidates).",
    "",
    "## Run row counts",
    "",
    f"- total rows: {len(runs_rows)} (ledger={len(ledger_rows)}, orphan run-dirs={len(orphan_ids)})",
    f"- status: keep={status_counts['keep']} archive={status_counts['archive']} delete_candidate={status_counts['delete_candidate']} deprecated={status_counts['deprecated']}",
    f"- era:    radial={era_counts['radial']} pre-radial={era_counts['pre-radial']} unknown={era_counts['unknown']}",
    f"- delete_candidates rows: total={len(del_rows)} safe_yes={safe_yes} safe_no={safe_no}",
    f"- artifacts rows: {len(artifact_rows)}",
    "",
    "## Keep-as-code paths (never proposed for deletion)",
    "",
    "- `models/unet_foa.py`",
    "- `models/bin_based/**`",
    "- `models/batvision/**`",
    "- `models/echonet/**`",
    "- `models/echodiffusion/**`",
    "- `models/pretrain/**`",
    "",
    "## kept-by-ledger-perf",
    "",
    "Reserved for unet_foa-family variants that the ledger shows beat the unet_foa baseline.",
    "Phase A does not auto-detect these; populate during Phase B review.",
    "",
    "## Status decision order",
    "",
    "1. Status starts with `PLANNED / NOT REPORTED` → `deprecated`.",
    "2. comparison_baseline + radial-era → `keep`.",
    "3. pre-radial era → `delete_candidate` (user rule: non-radial 실험은 전부 삭제).",
    "4. ledger Keep? contains `KEEP` + radial-era → `archive`.",
    "5. radial w/ explicit Failure Mode text → `delete_candidate`.",
    "6. fallback → `keep` (conservative).",
    "",
    "## Notes",
    "",
    "- `layout/` is out of scope and never indexed.",
    "- `.gitignore` excludes checkpoints/, results/, eval/, outputs/, wandb/, *.pt, *.pth.",
    "- Pre-radial run artifacts are listed with `safe_to_delete=no` pending per-row review.",
]
(ARCHIVE / "README.md").write_text("\n".join(readme_lines) + "\n")

# ---------------------------------------------------------------------------
# Print summary for the human running this
# ---------------------------------------------------------------------------
print("git_commit:", git_commit)
print("runs.csv rows:", len(runs_rows), "(ledger rows:", len(ledger_rows), "; orphan run-dirs:", len(orphan_ids), ")")
print("comparison_baselines.csv rows:", len(comp_rows))
print("delete_candidates.csv rows:", len(del_rows), "safe_yes:", safe_yes, "safe_no:", safe_no)
print("artifacts.csv rows:", len(artifact_rows))
print("status dist:", dict(status_counts))
print("era dist:", dict(era_counts))
if parse_errors:
    print("parse errors:", parse_errors[:10])
