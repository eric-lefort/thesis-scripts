#!/usr/bin/env python3
"""
make_cluster_ckpts_and_export.py
--------------------------------
• Split a DIG checkpoint into per-cluster mini-ckpts.
• Copy config.yml and patch *only* its ``output_dir:`` so it equals
  the cluster folder.
• Export each cluster with ``ns-export gaussian-splat`` in its own
  subprocess (frees GPU RAM between clusters).
"""

import os, subprocess, shutil, torch, numpy as np
from collections import defaultdict
from pathlib import Path

# ─── user paths ───────────────────────────────────────────────────────
CKPT_PATH   = Path("outputs/cooler_5/dig/2025-04-03_094939/nerfstudio_models/step-000014999.ckpt")
CONFIG_PATH = Path("outputs/cooler_5/dig/2025-04-03_094939/config.yml")
STATE_PATH  = Path("outputs/cooler_5/state.pt")            # has cluster_labels & means
OUT_ROOT    = Path("clusters")                             # clusters/cluster_<id>/
CUDA_DEVICE = "0"                                          # "" to force CPU

# constants copied from original config (used to build ckpt tree)
EXP_NAME    = "cooler_5"
METHOD_NAME = "dig"
TIMESTAMP   = "2025-04-03_094939"
# ──────────────────────────────────────────────────────────────────────


# ---------- helpers to edit YAML text without parsing -----------------
def _to_posix_block(path: Path, indent: str) -> list[str]:
    parts = ["/"] + [p for p in path.parts if p != "/"]
    block = [f"{indent}!!python/object/apply:pathlib.PosixPath"]
    block += [f"{indent}- {p}" for p in parts]
    return block


def _replace_block(lines: list[str], key: str, new_block: list[str]) -> list[str]:
    """Replace the YAML list that follows `key:` with new_block."""
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith(f"{key}:"):
            indent = line[: line.index(key[0])]
            out.append(f"{indent}{key}:")
            # skip old list lines
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("-"):
                i += 1
            # add replacement block
            out.extend(new_block)
            continue
        out.append(line)
        i += 1
    return out


def patch_output_dir(cfg_path: Path, new_dir: Path):
    """In-place set ``output_dir: <new_dir>`` (PosixPath style)."""
    lines = cfg_path.read_text().splitlines()
    block = _to_posix_block(new_dir, indent="  ")
    patched = _replace_block(lines, "output_dir", block)
    cfg_path.write_text("\n".join(patched) + "\n")
# ----------------------------------------------------------------------


def extract_gauss(ckpt):
    out, pre = {}, "_model.gauss_params."
    for k, v in ckpt["pipeline"].items():
        if k.startswith(pre):
            out[k[len(pre):]] = v
    return out


def exact_lut(means_full: torch.Tensor):
    lut = defaultdict(list)
    for i, xyz in enumerate(means_full.cpu().numpy()):
        lut[tuple(xyz)].append(i)
    return lut


def main():
    OUT_ROOT.mkdir(exist_ok=True)

    ckpt_big  = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    gauss_big = extract_gauss(ckpt_big)

    state     = torch.load(STATE_PATH, map_location="cpu", weights_only=False)
    labels_sm = state["cluster_labels"].cpu().numpy().astype(np.int64)
    means_sm  = state["means"]

    lut       = exact_lut(gauss_big["means"])
    idx_full  = torch.tensor([lut[tuple(m.tolist())].pop() for m in means_sm])

    cluster_full = -torch.ones(len(gauss_big["means"]), dtype=torch.long)
    cluster_full[idx_full] = torch.from_numpy(labels_sm)
    clusters = torch.unique(cluster_full[cluster_full >= 0]).tolist()

    for cid in clusters:
        cid_idx = torch.where(cluster_full == cid)[0]
        print(f"Cluster {cid}: {cid_idx.numel()} gaussians")

        # -- write mini-checkpoint ----------------------------------
        ckpt_c = ckpt_big.copy()
        for k in gauss_big:
            ckpt_c["pipeline"][f"_model.gauss_params.{k}"] = gauss_big[k][cid_idx]

        cluster_dir = OUT_ROOT / f"cluster_{cid}"
        ckpt_subdir = cluster_dir / EXP_NAME / METHOD_NAME / TIMESTAMP / "nerfstudio_models"
        ckpt_subdir.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt_c, ckpt_subdir / "step-000014999.ckpt")
        del ckpt_c

        # -- copy & patch config.yml --------------------------------
        cfg_out = cluster_dir / "config.yml"
        shutil.copy(CONFIG_PATH, cfg_out)
        patch_output_dir(cfg_out, cluster_dir.resolve())

        # -- export --------------------------------------------------
        env = {"CUDA_VISIBLE_DEVICES": CUDA_DEVICE, **os.environ}
        cmd = [
            "ns-export", "marching-cubes",
            "--load-config", str(cfg_out),
            "--output-dir",  str(cluster_dir),
            # "--output-filename", "splat.ply",
            # "--ply-color-mode", "rgb",
        ]
        print("  →", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

    print("All clusters exported ✔")


if __name__ == "__main__":
    main()
