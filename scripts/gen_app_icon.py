"""
gen_app_icon.py — Generate Engineer Companion launcher PNG icons at all mipmap densities.

Brand: black background (#0F0F0F), yellow glyph (#F5C518), red accent (#E5382B).
Design: dark rounded-square (or circle for round) background with a chat-bubble outline
        containing a 4-point spark, plus a small red dot accent.

Densities:
  mdpi     48 x 48
  hdpi     72 x 72
  xhdpi    96 x 96
  xxhdpi  144 x 144
  xxxhdpi 192 x 192

Output: android/app/src/main/res/mipmap-<density>/ic_launcher.png (square)
        android/app/src/main/res/mipmap-<density>/ic_launcher_round.png (circle bg)
"""

import math
import os
import sys

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Pillow not found. Install with: pip install Pillow", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DENSITIES = {
    "mdpi":     48,
    "hdpi":     72,
    "xhdpi":    96,
    "xxhdpi":  144,
    "xxxhdpi": 192,
}

BG_COLOR    = (15, 15, 15, 255)         # #0F0F0F
YELLOW      = (245, 197, 24, 255)       # #F5C518
RED         = (229, 56, 43, 255)        # #E5382B
TRANSPARENT = (0, 0, 0, 0)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RES_DIR     = os.path.join(PROJECT_DIR, "android", "app", "src", "main", "res")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_rounded_rect(draw: ImageDraw.ImageDraw, xy, radius, fill=None, outline=None, width=1):
    """Draw a rounded rectangle using pieslice + rectangle approach."""
    x0, y0, x1, y1 = xy
    r = radius
    if fill:
        draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
        draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
        draw.pieslice([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=fill)
        draw.pieslice([x1 - 2*r, y0, x1, y0 + 2*r], 270, 360, fill=fill)
        draw.pieslice([x0, y1 - 2*r, x0 + 2*r, y1], 90, 180, fill=fill)
        draw.pieslice([x1 - 2*r, y1 - 2*r, x1, y1], 0, 90, fill=fill)
    if outline:
        # Draw outline arcs + lines
        lw = width
        draw.arc([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=outline, width=lw)
        draw.arc([x1 - 2*r, y0, x1, y0 + 2*r], 270, 360, fill=outline, width=lw)
        draw.arc([x0, y1 - 2*r, x0 + 2*r, y1], 90, 180, fill=outline, width=lw)
        draw.arc([x1 - 2*r, y1 - 2*r, x1, y1], 0, 90, fill=outline, width=lw)
        # top, bottom, left, right lines
        draw.line([(x0 + r, y0), (x1 - r, y0)], fill=outline, width=lw)
        draw.line([(x0 + r, y1), (x1 - r, y1)], fill=outline, width=lw)
        draw.line([(x0, y0 + r), (x0, y1 - r)], fill=outline, width=lw)
        draw.line([(x1, y0 + r), (x1, y1 - r)], fill=outline, width=lw)


def draw_chat_bubble(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, stroke: float):
    """
    Draw a chat-bubble outline (rounded rect + tail) centered around (cx, cy-offset).
    size: icon size in px. The bubble occupies roughly 60% of the icon width.
    """
    # bubble body: occupies 62% width, 50% height, vertically offset slightly upward
    bw = size * 0.62
    bh = size * 0.50
    # center the bubble at cy - 5% of size (leave room for tail below)
    bx0 = cx - bw / 2
    by0 = cy - bh / 2 - size * 0.05
    bx1 = bx0 + bw
    by1 = by0 + bh

    r = size * 0.10   # corner radius
    sw = max(1, stroke)

    # Draw filled rounded rect background (slightly lighter than bg to give depth)
    draw_rounded_rect(draw, (bx0, by0, bx1, by1), r, fill=(25, 25, 25, 255))
    # Draw outline
    draw_rounded_rect(draw, (bx0, by0, bx1, by1), r, outline=YELLOW, width=int(sw))

    # Tail: a small triangle pointing down-left from bottom-left of bubble
    tail_tip_x = bx0 + size * 0.04
    tail_tip_y = by1 + size * 0.10
    tail_base_x1 = bx0 + size * 0.01
    tail_base_y  = by1 - 1
    tail_base_x2 = bx0 + size * 0.15
    tail_base_y2 = by1 - 1

    # Draw filled tail
    draw.polygon(
        [(tail_tip_x, tail_tip_y), (tail_base_x1, tail_base_y), (tail_base_x2, tail_base_y2)],
        fill=(25, 25, 25, 255)
    )
    # Outline the tail edges (left side + right side, not the base which merges with bubble)
    draw.line([(tail_base_x1, tail_base_y), (tail_tip_x, tail_tip_y)], fill=YELLOW, width=int(sw))
    draw.line([(tail_tip_x, tail_tip_y), (tail_base_x2, tail_base_y2)], fill=YELLOW, width=int(sw))

    return bx0, by0, bx1, by1


def draw_spark(draw: ImageDraw.ImageDraw, cx: float, cy: float, arm: float, stroke: float):
    """
    Draw a 4-point spark (asterisk with 8 arms: 4 cardinal + 4 diagonal) centered at (cx, cy).
    arm: half-length of each arm in px.
    """
    sw = max(1, int(stroke))
    diag = arm * 0.75  # diagonal arms slightly shorter for visual balance

    # Cardinal arms
    draw.line([(cx, cy - arm), (cx, cy + arm)], fill=YELLOW, width=sw)          # vertical
    draw.line([(cx - arm, cy), (cx + arm, cy)], fill=YELLOW, width=sw)          # horizontal

    # Diagonal arms
    d = diag / math.sqrt(2)
    draw.line([(cx - d, cy - d), (cx + d, cy + d)], fill=YELLOW, width=sw)      # NW-SE
    draw.line([(cx + d, cy - d), (cx - d, cy + d)], fill=YELLOW, width=sw)      # NE-SW


def draw_red_dot(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float):
    """Draw a filled red dot of radius r centered at (cx, cy)."""
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RED)


def render_icon(size: int, round_bg: bool = False) -> Image.Image:
    """
    Render the Engineer Companion icon at `size` x `size` pixels.
    round_bg: if True, use a circle background (for *_round variants).
    """
    # Work at 4x supersampling for smooth anti-aliasing, then downscale
    scale = 4
    ss = size * scale
    img = Image.new("RGBA", (ss, ss), TRANSPARENT)
    draw = ImageDraw.Draw(img)

    cx = ss / 2
    cy = ss / 2
    s = ss  # alias

    # --- Background ---
    margin = s * 0.02  # tiny margin so bg doesn't touch edges
    if round_bg:
        draw.ellipse([margin, margin, s - margin, s - margin], fill=BG_COLOR)
    else:
        # Rounded square, corner radius ~22% of size
        r_bg = s * 0.22
        draw_rounded_rect(draw, (margin, margin, s - margin, s - margin), r_bg, fill=BG_COLOR)

    # --- Chat bubble ---
    stroke = max(2, s * 0.032)   # stroke scales with icon size
    bx0, by0, bx1, by1 = draw_chat_bubble(draw, cx, cy, s, stroke)

    # --- Spark inside bubble ---
    # Spark center: horizontal center of bubble, vertical center of bubble body
    spark_cx = (bx0 + bx1) / 2
    spark_cy = (by0 + by1) / 2
    spark_arm = min(bx1 - bx0, by1 - by0) * 0.28   # arm length ~28% of bubble dim
    spark_stroke = max(2, s * 0.028)
    draw_spark(draw, spark_cx, spark_cy, spark_arm, spark_stroke)

    # --- Red accent dot at NE tip of spark ---
    dot_r = max(2, s * 0.040)
    diag = spark_arm * 0.75 / math.sqrt(2)
    dot_cx = spark_cx + diag + dot_r * 0.3
    dot_cy = spark_cy - diag - dot_r * 0.3
    draw_red_dot(draw, dot_cx, dot_cy, dot_r)

    # --- Downscale with LANCZOS for crisp result ---
    img = img.resize((size, size), Image.LANCZOS)
    return img


def main():
    generated = []

    for density, size in DENSITIES.items():
        mipmap_dir = os.path.join(RES_DIR, f"mipmap-{density}")
        os.makedirs(mipmap_dir, exist_ok=True)

        # Square icon
        square_path = os.path.join(mipmap_dir, "ic_launcher.png")
        img_sq = render_icon(size, round_bg=False)
        img_sq.save(square_path, "PNG")
        file_size = os.path.getsize(square_path)
        print(f"  [OK] {square_path}  ({size}x{size}, {file_size} bytes)")
        generated.append(square_path)

        # Round icon
        round_path = os.path.join(mipmap_dir, "ic_launcher_round.png")
        img_rnd = render_icon(size, round_bg=True)
        img_rnd.save(round_path, "PNG")
        file_size = os.path.getsize(round_path)
        print(f"  [OK] {round_path}  ({size}x{size}, {file_size} bytes)")
        generated.append(round_path)

    print(f"\nGenerated {len(generated)} icon files.")


if __name__ == "__main__":
    main()
