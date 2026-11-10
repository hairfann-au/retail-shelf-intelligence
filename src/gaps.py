"""Shelf-row and gap detection from product bounding boxes.

A gap is an empty horizontal stretch between two neighbouring products in the same
shelf row that is wider than `gap_factor` x the row's median product width.

Steps:
1. Group boxes into shelf rows with DBSCAN on (x-centre x a small weight, bottom edge).
   Products stand on a shelf, so their bottom edges line up. The small x term lets a
   row chain across the image even when the photo is tilted.
2. Re-join row pieces that a wide gap split apart: if one piece's bottom-edge line,
   extended sideways, runs through the other piece, they are the same shelf.
3. Skip "stacked" rows: rows whose items mostly sit directly on top of other boxes
   (e.g. boxes stacked two high). The space above an unstacked product is not a gap.
4. Within each remaining row, sort boxes left to right and measure the space between
   neighbours. Keep a gap only if other boxes cover at most `max_cover` of it.
5. Remove "structural" gaps: gaps that line up vertically across most rows of the image.
   These are fixtures (fridge door frames, shelf-unit uprights, peg-board columns),
   while out-of-stocks are scattered. A product stocked in a vertical block that sells
   out across several shelves can also line up, so a gap must line up in at least
   `structural_share` of the rows (and at least 4, in images with 5+ rows) to be treated
   as a fixture. With fewer rows, chance alignments of missed detections are too likely.
6. Flag the image for a human check if more than `flag_gaps_per_row` gaps remain per row:
   real out-of-stocks are occasional, so gaps everywhere usually mean an unusual layout.
Spaces at the start or end of a row are ignored: the shelf may simply end there.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from boxes import occupancy_mask, rect_coverage

GAP_COLUMNS = ["row", "x1", "y1", "x2", "y2", "width_ratio"]


def _merge_split_rows(boxes, raw, eps):
    """Join DBSCAN clusters that are side-by-side pieces of the same tilted shelf."""
    ids = [r for r in np.unique(raw) if r >= 0]
    if len(ids) < 2:
        return raw
    xc = (boxes[:, 0] + boxes[:, 2]) / 2
    y2 = boxes[:, 3]
    slopes = [np.polyfit(xc[raw == r], y2[raw == r], 1)[0] for r in ids if (raw == r).sum() >= 4]
    s_global = float(np.median(slopes)) if slopes else 0.0

    fits = {}
    for r in ids:
        m = raw == r
        s = np.polyfit(xc[m], y2[m], 1)[0] if m.sum() >= 4 else s_global
        fits[r] = (s, np.median(y2[m] - s * xc[m]), xc[m].min(), xc[m].max(), m.sum())

    parent = {r: r for r in ids}

    def root(r):
        while parent[r] != r:
            r = parent[r]
        return r

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sa, ca, lo_a, hi_a, na = fits[a]
            sb, cb, lo_b, hi_b, nb = fits[b]
            if min(hi_a, hi_b) > max(lo_a, lo_b):  # pieces overlap sideways: different rows
                continue
            s, c, other = (sa, ca, b) if na >= nb else (sb, cb, a)
            m = raw == other
            if np.median(np.abs(y2[m] - (s * xc[m] + c))) < eps:
                parent[root(b)] = root(a)

    merged = raw.copy()
    for r in ids:
        merged[raw == r] = root(r)
    return merged


def find_rows(boxes, x_weight=0.05, eps_factor=0.3, min_samples=2):
    """Return a shelf-row label per box, numbered top to bottom (-1 = no row)."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    labels = np.full(len(boxes), -1)
    if len(boxes) < min_samples:
        return labels
    eps = eps_factor * np.median(boxes[:, 3] - boxes[:, 1])
    feats = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2 * x_weight, boxes[:, 3]]
    raw = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feats)
    raw = _merge_split_rows(boxes, raw, eps)
    ids = sorted((r for r in np.unique(raw) if r >= 0),
                 key=lambda r: np.median(boxes[raw == r, 3]))
    for new, r in enumerate(ids):
        labels[raw == r] = new
    return labels


def _supported_share(row_boxes, all_boxes, tol):
    """Share of a row's boxes resting directly on top of another box."""
    supported = 0
    for x1, _, x2, y2 in row_boxes:
        touching = np.abs(all_boxes[:, 1] - y2) <= tol
        overlap = np.minimum(x2, all_boxes[:, 2]) - np.maximum(x1, all_boxes[:, 0])
        narrower = np.minimum(x2 - x1, all_boxes[:, 2] - all_boxes[:, 0])
        supported += bool(np.any(touching & (overlap >= 0.5 * narrower)))
    return supported / len(row_boxes)


def layout_features(boxes, rows=None, stacked_rows=()):
    """Measurements that separate shelf layouts from hanging (peg-hook) displays.

    top_misalign / bottom_misalign: median difference in top / bottom edge between each
    product and its right-hand neighbour, as a share of the typical product height.
    On shelves, bottoms line up (products stand on the shelf); on peg hooks, tops line up.
    """
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    keys = ["top_misalign", "bottom_misalign", "bottom_top_ratio",
            "share_in_rows", "n_rows", "n_stacked_rows", "stacked_share"]
    if len(boxes) < 5:
        return dict.fromkeys(keys, float("nan"))
    if rows is None:
        rows = find_rows(boxes)
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    med_h = np.median(h)

    d_top, d_bot = [], []
    for i in range(len(boxes)):
        right = (boxes[:, 0] >= boxes[i, 0] + 0.5 * w[i]) & (boxes[:, 0] <= boxes[i, 2] + w[i])
        v_overlap = np.minimum(boxes[i, 3], boxes[:, 3]) - np.maximum(boxes[i, 1], boxes[:, 1])
        cand = np.where(right & (v_overlap > 0.5 * np.minimum(h[i], h)))[0]
        if len(cand):
            j = cand[np.argmin(boxes[cand, 0])]
            d_top.append(abs(boxes[i, 1] - boxes[j, 1]) / med_h)
            d_bot.append(abs(boxes[i, 3] - boxes[j, 3]) / med_h)

    labels, counts = np.unique(rows[rows >= 0], return_counts=True)
    usable = [r for r, c in zip(labels, counts) if c >= 3]
    top_m = float(np.median(d_top)) if d_top else float("nan")
    bot_m = float(np.median(d_bot)) if d_bot else float("nan")
    return {
        "top_misalign": top_m,
        "bottom_misalign": bot_m,
        "bottom_top_ratio": bot_m / max(top_m, 1e-6),
        "share_in_rows": float(np.isin(rows, [r for r in usable if r not in stacked_rows]).mean()),
        "n_rows": len(usable),
        "n_stacked_rows": len(stacked_rows),
        "stacked_share": len(stacked_rows) / max(len(usable), 1),
    }


def _structural(gap_df, n_rows, share):
    """Boolean mask of gaps that line up vertically with gaps in most other rows."""
    if len(gap_df) == 0 or n_rows < 5:  # with few rows, chance alignments are too likely
        return np.zeros(len(gap_df), bool)
    x1, x2, r = gap_df.x1.to_numpy(), gap_df.x2.to_numpy(), gap_df.row.to_numpy()
    overlap = np.minimum(x2[:, None], x2[None, :]) - np.maximum(x1[:, None], x1[None, :])
    narrower = np.minimum((x2 - x1)[:, None], (x2 - x1)[None, :])
    aligned = (overlap >= 0.5 * narrower) & (r[:, None] != r[None, :])
    rows_lined_up = np.array([len(set(r[aligned[i]])) + 1 for i in range(len(r))])
    return rows_lined_up >= max(4, int(np.ceil(share * n_rows)))


def detect_gaps(boxes, img_w, img_h, gap_factor=1.0, min_row_boxes=3, max_cover=0.3,
                stacked_share=0.5, structural_share=0.6, flag_gaps_per_row=2.0,
                x_weight=0.05, eps_factor=0.3):
    """Find gaps in one image.

    Returns a dict with:
      rows            - shelf-row label per box (-1 = no row)
      stacked_rows    - row labels skipped because they sit on top of other products
      gaps            - DataFrame of likely out-of-stock gaps (row, x1, y1, x2, y2, width_ratio)
      structural_gaps - gaps removed because they line up with fixtures
      score           - total gap width / total width of the analysed rows (0 = full shelves)
      gaps_per_row    - remaining gaps per analysed row
      check_layout    - True if gaps are everywhere, so the score needs a human check
    """
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    rows = find_rows(boxes, x_weight, eps_factor)
    if len(boxes) == 0:
        return {"rows": rows, "stacked_rows": [], "gaps": pd.DataFrame(columns=GAP_COLUMNS),
                "structural_gaps": pd.DataFrame(columns=GAP_COLUMNS), "score": 0.0,
                "gaps_per_row": 0.0, "check_layout": False}

    mask = occupancy_mask(boxes, img_w, img_h)
    med_h = np.median(boxes[:, 3] - boxes[:, 1])
    gaps, stacked, span_total, n_analysed = [], [], 0.0, 0

    for r in np.unique(rows[rows >= 0]):
        rb = boxes[rows == r]
        if len(rb) < min_row_boxes:
            continue
        if _supported_share(rb, boxes[rows != r], 0.1 * med_h) > stacked_share:
            stacked.append(int(r))
            continue
        n_analysed += 1
        rb = rb[np.argsort(rb[:, 0])]
        med_w = np.median(rb[:, 2] - rb[:, 0])
        span_total += rb[:, 2].max() - rb[:, 0].min()

        left = rb[0]
        right_edge = left[2]
        for b in rb[1:]:
            width = b[0] - right_edge
            if width > gap_factor * med_w:
                rect = (right_edge, min(left[1], b[1]), b[0], max(left[3], b[3]))
                if rect_coverage(mask, rect) <= max_cover:
                    gaps.append((int(r), *rect, width / med_w))
            if b[2] > right_edge:
                right_edge, left = b[2], b

    all_gaps = pd.DataFrame(gaps, columns=GAP_COLUMNS)
    fixture = _structural(all_gaps, n_analysed, structural_share)
    gap_df = all_gaps[~fixture].reset_index(drop=True)
    score = float((gap_df.x2 - gap_df.x1).sum() / span_total) if span_total else 0.0
    per_row = len(gap_df) / n_analysed if n_analysed else 0.0
    return {"rows": rows, "stacked_rows": stacked, "gaps": gap_df,
            "structural_gaps": all_gaps[fixture].reset_index(drop=True), "score": score,
            "gaps_per_row": per_row, "check_layout": per_row > flag_gaps_per_row}
