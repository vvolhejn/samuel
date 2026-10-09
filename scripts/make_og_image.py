"""Build the webapp's social card and Apple touch icon.

Run with: uv run --with pillow python scripts/make_og_image.py
"""

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "samuel-screenshot.png"
TARGET = ROOT / "webapp" / "public" / "og.png"
ICON_SOURCE = ROOT / "webapp" / "app" / "icon.png"
ICON_TARGET = ROOT / "webapp" / "app" / "apple-icon.png"

WIDTH, HEIGHT = 1200, 630
# Target side for the icon inside the 180px touch icon, before rounding to an
# integer scale factor.
ICON_BOX = 140
HIGHLIGHT = "#f92672"
FOREGROUND = "#1c1520"
MUTED = "#57534e"

TITLE = "Samuel"
LINES = [
    "A model that learns to control",
    "a silly mouth to mimic speech.",
]
SUBTITLE = "samuel.vvolhejn.com"


def font(size: int, index: int) -> ImageFont.FreeTypeFont:
    """Helvetica Neue face by collection index (1 = bold, 0 = regular)."""
    return ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", size, index=index)


def trimmed_box(image: Image.Image) -> tuple[int, int, int, int]:
    """Bounding box of everything that isn't the white page background."""
    diff = ImageChops.difference(image.convert("RGB"), Image.new("RGB", image.size, "white"))
    return diff.convert("L").point(lambda v: 255 if v > 8 else 0).getbbox()


def make_apple_icon() -> None:
    """iOS ignores transparency and offers no padding, so bake in both."""
    icon = Image.open(ICON_SOURCE).convert("RGBA")
    # The source is pixel art: an integer nearest-neighbour scale keeps the
    # edges hard instead of smearing them.
    scale = max(1, round(ICON_BOX / max(icon.width, icon.height)))
    icon = icon.resize((icon.width * scale, icon.height * scale), Image.NEAREST)
    canvas = Image.new("RGB", (180, 180), "white")
    canvas.paste(icon, ((180 - icon.width) // 2, (180 - icon.height) // 2), icon)
    canvas.save(ICON_TARGET, optimize=True)
    print(f"wrote {ICON_TARGET}")


def main() -> None:
    make_apple_icon()

    card = Image.new("RGB", (WIDTH, HEIGHT), "white")

    tract = Image.open(SOURCE).convert("RGBA")
    # Crop off the voicebox sliders: they read as UI chrome at card size.
    tract = tract.crop((0, 0, tract.width, int(tract.height * 0.70)))
    tract = tract.crop(trimmed_box(tract))
    scale = min((HEIGHT - 30) / tract.height, 620 / tract.width)
    tract = tract.resize(
        (round(tract.width * scale), round(tract.height * scale)), Image.LANCZOS
    )
    card.paste(tract, (WIDTH - tract.width - 30, (HEIGHT - tract.height) // 2), tract)

    draw = ImageDraw.Draw(card)
    x = 72
    draw.text((x, 190), TITLE, font=font(96, 1), fill=HIGHLIGHT)
    y = 320
    for line in LINES:
        draw.text((x, y), line, font=font(34, 0), fill=FOREGROUND)
        y += 48
    draw.text((x, y + 24), SUBTITLE, font=font(26, 0), fill=MUTED)

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    card.save(TARGET, optimize=True)
    print(f"wrote {TARGET} ({TARGET.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
