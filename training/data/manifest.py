"""
Split manifest + data-leakage verification.

write_manifest(): records every ORIGINAL image with its label and partition
(train / val / test) as CSV and JSON, together with a SHA-256 content hash.

verify_no_overlap(): scans prepared split folders (train, train_aug, val, test)
and fails if any image -- or any augmented variant of an image -- appears in
more than one of {train, val, test}. Checked two ways:
  1. by source identifier: augmented files are named "<stem>_aug<i>.png", so
     "<stem>" identifies the original they came from;
  2. by pixel content: SHA-256 of decoded pixels for every original image.

Usage (after run_full_training.py has prepared datasets/prepared):
    python training/data/manifest.py --prepared_dir datasets/prepared
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2

EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
AUG_SUFFIX = re.compile(r"_aug\d+$")


def _pixel_hash(path: str) -> str:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    data = img.tobytes() if img is not None else open(path, "rb").read()
    return hashlib.sha256(data).hexdigest()


def _list_images(root: Path) -> List[Path]:
    files = []
    for ext in EXTS:
        files.extend(root.rglob(ext))
    return sorted(f for f in files if "_mask" not in f.name.lower())


def source_id(path: Path) -> str:
    """Identifier of the ORIGINAL image a file comes from (class + stem without _augN)."""
    return f"{path.parent.name}/{AUG_SUFFIX.sub('', path.stem)}"


def write_manifest(splits: Dict[str, List[Dict]], output_dir: str) -> str:
    """Write split_manifest.csv / .json listing every original image and its partition."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for partition in ("train", "val", "test"):
        for s in splits.get(partition, []):
            rows.append({
                "image_id": f"{s['class_name']}/{Path(s['image_path']).stem}",
                "file": os.path.basename(s["image_path"]),
                "class_name": s["class_name"],
                "label": s["label"],
                "partition": partition,
                "has_mask": bool(s.get("mask_path")),
                "source_dataset": s.get("source", ""),
                "sha256": _pixel_hash(s["image_path"]),
            })
    csv_path = os.path.join(output_dir, "split_manifest.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = defaultdict(lambda: defaultdict(int))
    for r in rows:
        summary[r["partition"]][r["class_name"]] += 1
    with open(os.path.join(output_dir, "split_manifest.json"), "w") as f:
        json.dump({"summary": summary, "images": rows}, f, indent=2)
    print(f"[Manifest] {len(rows)} images written to {csv_path}")
    for part, counts in summary.items():
        print(f"  {part:5s}: {dict(counts)}  total={sum(counts.values())}")
    return csv_path


def verify_no_overlap(prepared_dir: str) -> bool:
    """
    Check that train (incl. train_aug), val and test share no image or augmented variant.
    Returns True if clean; prints every conflict otherwise.
    """
    root = Path(prepared_dir)
    groups = {"train": ["train", "train_aug"], "val": ["val"], "test": ["test"]}
    ids, hashes = {}, {}
    counts = {}
    for part, folders in groups.items():
        id_set, hash_set, n = set(), {}, 0
        for folder in folders:
            if not (root / folder).exists():
                continue
            for p in _list_images(root / folder):
                n += 1
                id_set.add(source_id(p))
                if not AUG_SUFFIX.search(p.stem):  # hash originals only
                    hash_set[_pixel_hash(str(p))] = str(p)
        ids[part], hashes[part], counts[part] = id_set, hash_set, n

    ok = True
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared_ids = ids[a] & ids[b]
        shared_px = set(hashes[a]) & set(hashes[b])
        if shared_ids or shared_px:
            ok = False
            print(f"[LEAK] {a} <-> {b}: {len(shared_ids)} shared source ids, "
                  f"{len(shared_px)} pixel-identical images")
            for sid in sorted(shared_ids)[:10]:
                print(f"    id: {sid}")
            for h in sorted(shared_px)[:10]:
                print(f"    duplicate: {hashes[a][h]}  ==  {hashes[b][h]}")
    print(f"[Overlap check] image files: {counts}")
    print("[Overlap check] PASSED - no image or augmented variant is shared between "
          "train/val/test." if ok else "[Overlap check] FAILED")
    return ok


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Verify that prepared splits do not overlap")
    p.add_argument("--prepared_dir", default="datasets/prepared")
    a = p.parse_args()
    sys.exit(0 if verify_no_overlap(a.prepared_dir) else 1)
