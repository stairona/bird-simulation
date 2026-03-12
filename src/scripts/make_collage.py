from PIL import Image
import os

# Folder where the 24 images were generated
INPUT_DIR = "../outputs/monthly-annotated-maps"

# Output files
OUT_VISITED = "../outputs/summary-plots/collage_visited.png"
OUT_WHOLE = "../outputs/summary-plots/collage_whole.png"

# Layout: 4 columns x 3 rows
COLS = 4
ROWS = 3
BORDER = 6  # thin white border thickness


def load_images(prefix):
    images = []
    for i in range(1, 13):
        for fname in sorted(os.listdir(INPUT_DIR)):
            if fname.startswith(f"{prefix}_{i:02d}_") and fname.endswith(".png"):
                images.append(Image.open(os.path.join(INPUT_DIR, fname)))
                break
    return images


def create_collage(images, output_name):
    if len(images) != 12:
        print("Error: Did not find 12 images.")
        return

    w, h = images[0].size

    total_w = COLS * w + (COLS + 1) * BORDER
    total_h = ROWS * h + (ROWS + 1) * BORDER

    collage = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    index = 0
    for row in range(ROWS):
        for col in range(COLS):
            x = BORDER + col * (w + BORDER)
            y = BORDER + row * (h + BORDER)
            collage.paste(images[index], (x, y))
            index += 1

    collage.save(output_name)
    print(f"Saved {output_name}")


def main():
    visited_imgs = load_images("visited")
    whole_imgs = load_images("whole")

    create_collage(visited_imgs, OUT_VISITED)
    create_collage(whole_imgs, OUT_WHOLE)


if __name__ == "__main__":
    main()
