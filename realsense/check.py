import os
from pathlib import Path

if __name__ == "__main__":
    input_dir = Path("tmpws")

    cams = sorted(os.listdir(input_dir))

    print(f"Number of cameras found: {len(cams)}")

    images = set()

    for i, cam in enumerate(cams):
        current = set(os.listdir(input_dir / cam))
        if i == 0:
            images |= current
        else:
            images &= current

        print(f"Number of images found for cam{i}: {len(current)}")

    print(f"Number of images: {len(images)}")

    for cam in cams:
        current = set(os.listdir(input_dir / cam))
        for image in current - images:
            os.remove(input_dir / cam / image)
