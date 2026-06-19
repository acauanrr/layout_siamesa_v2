#!/usr/bin/env python3
"""Dry-run auditor: detect human RED annotations (rectangles, ellipses/circles,
arrows, hand-drawn strokes, red tags) on images in errors_dataset/.

READ-ONLY. Does not modify or delete any image. Writes a CSV report only.

Heuristic pipeline (numpy + PIL + scipy only):
 1. Load RGB. Optionally downscale very large images for speed (mask is scale-
    invariant via fractions; geometry thresholds use the working resolution).
 2. Build a "strong red" mask:
      R high, R - max(G,B) large, and G,B both low (excludes skin/orange/pink).
 3. Connected components (scipy.ndimage.label, 8-connectivity).
 4. Discard tiny noise components.
 5. For surviving components compute geometry: bbox, fill ratio (area / bbox area),
    elongation, perimeter-ish thinness. Flag a component as a likely human mark when:
      - it is LARGE (big bbox spanning a good fraction of the image), OR
      - it is an OUTLINE/RING (large bbox but low fill ratio -> drawn rectangle/ellipse), OR
      - it is a THIN ELONGATED stroke that is reasonably LONG (arrow / underline / line).
    Solid, small, compact blobs (badges, icons, buttons, error text) are NOT flagged.
 6. Image is flagged if any component qualifies. Reason string explains why.

The thresholds below were tuned by visually inspecting calibration images
(boundbox arrows/ellipses/rectangles vs. red notification badges, red app icons,
red 'end call' buttons).
"""

import os
import csv
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

BASE = "/home/acauan/iats/layout_siamesa_v2/data/input/errors_dataset"
OUT_CSV = "/home/acauan/iats/layout_siamesa_v2/artifacts/reports/red_marks_audit.csv"

CATEGORIES = [
    "black bars",
    "disordered layout",
    "distortion",
    "empty space",
    "orientation",
    "overlay",
]

# ---- red-mask thresholds ----
R_MIN = 110          # red channel must be reasonably bright
RED_DOMINANCE = 70   # R - max(G,B) : how much red dominates the other channels
GB_MAX = 120         # both G and B must be below this (kills skin/orange/pink/magenta)

# ---- working resolution (downscale long side to this for speed/consistency) ----
WORK_LONG_SIDE = 900

# ---- component noise / geometry thresholds (in WORKING-resolution pixels) ----
MIN_COMP_PIXELS = 60         # ignore specks smaller than this

# "LARGE HOLLOW" mark: a big red shape that is mostly empty inside -> a drawn
# loop / ellipse / rectangle / scribble spanning a large area. We REQUIRE low
# fill here: solid filled red regions of this size are almost always UI content
# (red banners, ads, red clothing, video stills) and are NOT human annotations.
LARGE_BBOX_AREA_FRAC = 0.08  # bbox covers >=8% of the image -> big drawing
LARGE_MIN_PIXELS = 1200
LARGE_FILL_MAX = 0.45        # must be hollow-ish; rejects solid red banners/clothing

# "OUTLINE / RING" (drawn rectangle or ellipse): big bbox but mostly hollow
OUTLINE_BBOX_AREA_FRAC = 0.03   # bbox at least 3% of image
OUTLINE_FILL_MAX = 0.35         # filled <=35% of its bbox -> hollow outline
OUTLINE_MIN_BBOX_SIDE = 60      # bbox not trivially small (working px)

# "THIN ELONGATED stroke" (arrow / underline / hand line)
STROKE_MIN_LEN = 110            # longest bbox side at least this (working px)
STROKE_ASPECT = 3.0             # elongated: long side / short side
STROKE_FILL_MAX = 0.45          # thin: low fill within its bbox


def build_red_mask(rgb):
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    maxgb = np.maximum(g, b)
    mask = (
        (r >= R_MIN)
        & ((r - maxgb) >= RED_DOMINANCE)
        & (g <= GB_MAX)
        & (b <= GB_MAX)
    )
    return mask


def analyze(path):
    """Return dict with metrics + flag + reason for one image."""
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:  # corrupt / unreadable
        return {
            "flagged": 0,
            "red_area_frac": 0.0,
            "n_components": 0,
            "max_component_area": 0,
            "reason": f"READ_ERROR:{e}",
        }

    w0, h0 = im.size
    long_side = max(w0, h0)
    if long_side > WORK_LONG_SIDE:
        scale = WORK_LONG_SIDE / long_side
        im = im.resize((max(1, int(w0 * scale)), max(1, int(h0 * scale))), Image.BILINEAR)
    rgb = np.asarray(im)
    H, W = rgb.shape[:2]
    img_area = float(H * W)

    mask = build_red_mask(rgb)
    red_frac = float(mask.sum()) / img_area

    # close tiny gaps so dashed/anti-aliased strokes connect, then label
    mask_c = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=1)
    labels, n = ndimage.label(mask_c, structure=np.ones((3, 3)))

    flagged = 0
    reasons = []
    max_comp_area = 0
    n_real = 0

    if n > 0:
        sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
        objs = ndimage.find_objects(labels)
        for i, sl in enumerate(objs):
            comp_pixels = int(sizes[i])
            if comp_pixels < MIN_COMP_PIXELS:
                continue
            n_real += 1
            ys, xs = sl
            bh = ys.stop - ys.start
            bw = xs.stop - xs.start
            bbox_area = float(bh * bw)
            bbox_area_frac = bbox_area / img_area
            fill = comp_pixels / bbox_area if bbox_area > 0 else 1.0
            long_b = max(bh, bw)
            short_b = max(1, min(bh, bw))
            aspect = long_b / short_b
            if comp_pixels > max_comp_area:
                max_comp_area = comp_pixels

            # rule 1: large HOLLOW red shape (drawn loop/ellipse/rect/scribble).
            # Requires low fill so solid red banners/clothing/video-stills are rejected.
            if (
                bbox_area_frac >= LARGE_BBOX_AREA_FRAC
                and comp_pixels >= LARGE_MIN_PIXELS
                and fill <= LARGE_FILL_MAX
            ):
                flagged = 1
                reasons.append(
                    f"LARGE_HOLLOW(bboxfrac={bbox_area_frac:.3f},px={comp_pixels},fill={fill:.2f})"
                )
                continue

            # rule 2: outline / ring -> drawn rectangle or ellipse
            if (
                bbox_area_frac >= OUTLINE_BBOX_AREA_FRAC
                and fill <= OUTLINE_FILL_MAX
                and min(bh, bw) >= OUTLINE_MIN_BBOX_SIDE
            ):
                flagged = 1
                reasons.append(
                    f"OUTLINE(bboxfrac={bbox_area_frac:.3f},fill={fill:.2f},{bw}x{bh})"
                )
                continue

            # rule 3: thin elongated stroke -> arrow / underline / hand line
            if (
                long_b >= STROKE_MIN_LEN
                and aspect >= STROKE_ASPECT
                and fill <= STROKE_FILL_MAX
            ):
                flagged = 1
                reasons.append(
                    f"STROKE(len={long_b},aspect={aspect:.1f},fill={fill:.2f})"
                )
                continue

    reason = ";".join(reasons) if reasons else (
        "no_qualifying_red_component" if n_real else "no_strong_red"
    )
    return {
        "flagged": flagged,
        "red_area_frac": round(red_frac, 6),
        "n_components": n_real,
        "max_component_area": max_comp_area,
        "reason": reason,
    }


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    rows = []
    per_cat = {c: {"total": 0, "flagged": 0, "examples": []} for c in CATEGORIES}

    for cat in CATEGORIES:
        d = os.path.join(BASE, cat)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            path = os.path.join(d, fn)
            if not os.path.isfile(path):
                continue
            res = analyze(path)
            per_cat[cat]["total"] += 1
            if res["flagged"]:
                per_cat[cat]["flagged"] += 1
                if len(per_cat[cat]["examples"]) < 6:
                    per_cat[cat]["examples"].append(fn)
            rows.append({
                "category": cat,
                "filename": fn,
                "flagged": res["flagged"],
                "red_area_frac": res["red_area_frac"],
                "n_components": res["n_components"],
                "max_component_area": res["max_component_area"],
                "reason": res["reason"],
            })

    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "category", "filename", "flagged", "red_area_frac",
                "n_components", "max_component_area", "reason",
            ],
        )
        wr.writeheader()
        wr.writerows(rows)

    total = len(rows)
    total_flag = sum(r["flagged"] for r in rows)
    print(f"WROTE {OUT_CSV}")
    print(f"TOTAL images={total} flagged={total_flag}")
    for cat in CATEGORIES:
        c = per_cat[cat]
        print(f"  {cat}: {c['flagged']}/{c['total']} flagged | examples: {c['examples']}")
    # emit a compact machine-readable line for the caller
    import json
    print("JSON_SUMMARY=" + json.dumps({
        "total": total, "total_flagged": total_flag,
        "per_cat": {k: {"total": v["total"], "flagged": v["flagged"], "examples": v["examples"]}
                    for k, v in per_cat.items()},
    }))


if __name__ == "__main__":
    sys.exit(main())
