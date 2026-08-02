#!/usr/bin/env python3
"""Generate a Kalibr-style Aprilgrid (tag36h11) as a print-exact PDF + YAML.

Layout follows Kalibr's aprilgrid convention: tags of edge `tagsize` separated
by gaps of `tagspacing * tagsize`, IDs row-major starting at 0. The PDF is
emitted at true physical scale for the given page size and includes a 100 mm
scale bar — print at 100% (no "fit to page") and verify the bar with a ruler.

Usage: make_aprilgrid.py [--rows 6] [--cols 6] [--page letter|a4] [--out dir]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

PAGES_IN = {"letter": (8.5, 11.0), "a4": (8.27, 11.69)}
MARGIN_IN = 0.5
DPI = 300
SPACING_RATIO = 0.3  # kalibr default: gap = 0.3 * tagsize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--page", choices=PAGES_IN, default="letter")
    ap.add_argument("--out", type=Path, default=Path("target"))
    args = ap.parse_args()

    page_w, page_h = PAGES_IN[args.page]
    usable_w = page_w - 2 * MARGIN_IN
    usable_h = page_h - 2 * MARGIN_IN - 0.6  # reserve strip for label + scale bar

    # tag edge s such that cols*s + (cols-1)*0.3s fits width (same for height)
    s_w = usable_w / (args.cols + (args.cols - 1) * SPACING_RATIO)
    s_h = usable_h / (args.rows + (args.rows - 1) * SPACING_RATIO)
    tag_in = min(s_w, s_h)
    tag_m = tag_in * 0.0254
    gap_in = SPACING_RATIO * tag_in

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_px = int(round(tag_in * DPI))
    # 36h11 marker bitmap is 10x10 cells incl. border; render at cell multiple
    cells = 10
    render_px = (tag_px // cells) * cells or cells

    page = np.full((int(page_h * DPI), int(page_w * DPI)), 255, np.uint8)
    grid_w_in = args.cols * tag_in + (args.cols - 1) * gap_in
    grid_h_in = args.rows * tag_in + (args.rows - 1) * gap_in
    x0_in = (page_w - grid_w_in) / 2
    y0_in = MARGIN_IN

    for r in range(args.rows):
        for c in range(args.cols):
            tag_id = r * args.cols + c
            img = cv2.aruco.generateImageMarker(dictionary, tag_id, render_px)
            img = cv2.resize(img, (tag_px, tag_px), interpolation=cv2.INTER_NEAREST)
            # kalibr ID 0 is bottom-left; image row 0 is top -> flip row order
            y_in = y0_in + (args.rows - 1 - r) * (tag_in + gap_in)
            x_in = x0_in + c * (tag_in + gap_in)
            y, x = int(y_in * DPI), int(x_in * DPI)
            page[y:y + tag_px, x:x + tag_px] = img

    # 100 mm scale bar + label
    bar_px = int(round(100 / 25.4 * DPI))
    by = int((page_h - MARGIN_IN - 0.25) * DPI)
    bx = int(x0_in * DPI)
    page[by:by + 8, bx:bx + bar_px] = 0
    label = (f"aprilgrid {args.rows}x{args.cols} tag36h11  "
             f"tagsize {tag_m:.4f} m  spacing {SPACING_RATIO}  "
             f"| bar = 100.0 mm | print at 100%")
    cv2.putText(page, label, (bx, by - 12), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, 0, 2, cv2.LINE_AA)

    args.out.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out / f"aprilgrid_{args.rows}x{args.cols}_{args.page}.pdf"
    # write PDF at exact physical size via PIL (300 dpi resolution tag)
    from PIL import Image
    Image.fromarray(page).save(pdf_path, "PDF", resolution=DPI)

    yaml_path = args.out / "aprilgrid.yaml"
    yaml_path.write_text(
        "target_type: 'aprilgrid'\n"
        f"tagCols: {args.cols}\n"
        f"tagRows: {args.rows}\n"
        f"tagSize: {tag_m:.6f}   # meters, edge of black square — verify after printing!\n"
        f"tagSpacing: {SPACING_RATIO}   # gap = tagSpacing * tagSize\n")

    print(f"wrote {pdf_path}")
    print(f"wrote {yaml_path}")
    print(f"tag size: {tag_m * 1000:.2f} mm, gap: {gap_in * 25.4:.2f} mm, "
          f"grid: {grid_w_in * 25.4:.0f} x {grid_h_in * 25.4:.0f} mm")


if __name__ == "__main__":
    main()
