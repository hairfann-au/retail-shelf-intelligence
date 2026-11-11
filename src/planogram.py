"""Simulated planogram: which Instacart product sits where on a shelf photo.

SKU-110K photos have no product names, so this project assigns them a store layout:
- Each photo is one shelf bay. A bay is given one Instacart aisle, chosen by a stable hash
  of its file name among the best-selling aisles (the same photo always gets the same aisle).
- Each bay has up to `n_shelves` shelves (detected rows, numbered top to bottom) and
  `n_positions` positions across each shelf.
- The aisle's best sellers go at eye level first (shelves 3, 2, 4, 1, ...), as in real
  stores, and positions fill left to right.

This is a stand-in for a real planogram, which a retailer would supply. Say so wherever
results that depend on it are reported.
"""
import zlib

import numpy as np
import pandas as pd

EYE_LEVEL_ORDER = [3, 2, 4, 1, 5, 0, 6, 7, 8, 9]


def build_planogram(product_orders, n_aisles=60, n_shelves=8, n_positions=5):
    """Build the layout table.

    product_orders: DataFrame with product_id, product_name, aisle_id, aisle, orders
                    (orders = how many orders contain the product).
    Returns one row per (aisle_slot, shelf, position) with the product placed there.
    """
    aisle_sales = product_orders.groupby(["aisle_id", "aisle"])["orders"].sum().nlargest(n_aisles)
    shelf_order = [s for s in EYE_LEVEL_ORDER if s < n_shelves]
    shelf_order += [s for s in range(n_shelves) if s not in shelf_order]
    n_slots = n_shelves * n_positions

    rows = []
    for aisle_slot, (aisle_id, aisle) in enumerate(aisle_sales.index):
        best = (product_orders[product_orders.aisle_id == aisle_id]
                .nlargest(n_slots, "orders").reset_index(drop=True))
        for k in range(n_slots):
            p = best.iloc[k % len(best)]  # small aisles repeat their products
            shelf, position = shelf_order[k // n_positions], k % n_positions
            rows.append((aisle_slot, aisle_id, aisle, shelf, position,
                         int(p.product_id), p.product_name))
    cols = ["aisle_slot", "aisle_id", "aisle", "shelf", "position", "product_id", "product_name"]
    return pd.DataFrame(rows, columns=cols).sort_values(["aisle_slot", "shelf", "position"],
                                                        ignore_index=True)


def bay_aisle_slot(bay_name, n_aisles):
    """Stable aisle assignment for a photo (Python's hash() changes between runs; crc32 doesn't)."""
    return zlib.crc32(str(bay_name).encode()) % n_aisles


def lookup(planogram, bay_name, shelf, x_frac):
    """Product placed at a shelf row and horizontal position (0 = left edge, 1 = right edge)."""
    n_aisles = planogram["aisle_slot"].nunique()
    n_shelves = planogram["shelf"].max() + 1
    n_positions = planogram["position"].max() + 1
    slot = bay_aisle_slot(bay_name, n_aisles)
    shelf = int(min(max(shelf, 0), n_shelves - 1))
    position = int(min(max(np.floor(x_frac * n_positions), 0), n_positions - 1))
    hit = planogram[(planogram.aisle_slot == slot) & (planogram.shelf == shelf)
                    & (planogram.position == position)]
    return hit.iloc[0]
