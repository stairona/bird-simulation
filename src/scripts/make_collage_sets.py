from PIL import Image
import os

INPUT_DIR = "../outputs/monthly-annotated-maps"

# Layout: 3 columns x 2 rows (6 months per collage)
COLS = 3
ROWS = 2
BORDER = 6  # thin white border

def find_month_file(prefix: str, month: int) -> str:
    # Matches: visited_01_Jan_eco.png, whole_12_Dec_eco.png, etc.
    # Robust: find first file starting with prefix_XX_ and ending .png
    start = f"{prefix}_{month:02d}_"
    for fname in sorted(os.listdir(INPUT_DIR)):
        if fname.startswith(start) and fname.endswith(".png"):
            return os.path.join(INPUT_DIR, fname)
    raise FileNotFoundError(f"Missing file for {prefix} month {month:02d} in {INPUT_DIR}")

def load_months(prefix: str, months: list[int]) -> list[Image.Image]:
    imgs = []
    for m in months:
        path = find_month_file(prefix, m)
        imgs.append(Image.open(path))
    return imgs

def create_collage(images: list[Image.Image], out_name: str):
    if len(images) != COLS * ROWS:
        raise ValueError(f"Expected {COLS*ROWS} images, got {len(images)}")

    w, h = images[0].size
    # Ensure all same size
    for im in images:
        if im.size != (w, h):
            raise ValueError("Not all images are the same size. Regenerate months with consistent settings.")

    total_w = COLS * w + (COLS + 1) * BORDER
    total_h = ROWS * h + (ROWS + 1) * BORDER

    collage = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    idx = 0
    for r in range(ROWS):
        for c in range(COLS):
            x = BORDER + c * (w + BORDER)
            y = BORDER + r * (h + BORDER)
            collage.paste(images[idx], (x, y))
            idx += 1

    collage.save(out_name)
    print(f"Saved: {out_name}")

def main():
    # Month groups
    first_half = [1, 2, 3, 4, 5, 6]   # Jan–Jun
    second_half = [7, 8, 9, 10, 11, 12]  # Jul–Dec

    # VISITED
    create_collage(load_months("visited", first_half), "../outputs/summary-plots/collage_visited_Jan-Jun.png")
    create_collage(load_months("visited", second_half), "../outputs/summary-plots/collage_visited_Jul-Dec.png")

    # WHOLE
    create_collage(load_months("whole", first_half), "../outputs/summary-plots/collage_whole_Jan-Jun.png")
    create_collage(load_months("whole", second_half), "../outputs/summary-plots/collage_whole_Jul-Dec.png")

if __name__ == "__main__":
    main()
