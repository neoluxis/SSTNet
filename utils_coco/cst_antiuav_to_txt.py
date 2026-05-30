"""
Convert CST_AntiUAV/frhybrid YOLO labels into COCO-style txt files.

Each output line follows the same format used by coco_train_IRDST.txt:

    /abs/path/to/image.jpg x1,y1,x2,y2,class x1,y1,x2,y2,class ...

The script reads the image size for every frame, converts normalized YOLO
labels into absolute corner coordinates, and writes one line per image.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def find_label_files(labels_root: Path) -> List[Path]:
    """Collect all per-frame label files under a split's labels directory."""
    return sorted(
        path
        for path in labels_root.rglob("*.txt")
        if path.is_file() and path.name != "classes.txt"
    )


def image_path_from_label(label_path: Path, split_root: Path, image_ext: str) -> Path:
    """Map .../labels/seq/000001.txt -> .../images/seq/000001.jpg."""
    rel_path = label_path.relative_to(split_root / "labels")
    return (split_root / "images" / rel_path).with_suffix(image_ext)


def load_yolo_boxes(label_path: Path, image_size: Tuple[int, int]) -> List[Tuple[int, int, int, int, int]]:
    """Convert YOLO normalized labels to absolute COCO-style corner boxes."""
    width, height = image_size
    boxes: List[Tuple[int, int, int, int, int]] = []

    if not label_path.exists():
        return boxes

    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return boxes

    for line in content.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        box_width = float(parts[3]) * width
        box_height = float(parts[4]) * height

        x_min = int(round(x_center - box_width / 2.0))
        y_min = int(round(y_center - box_height / 2.0))
        x_max = int(round(x_center + box_width / 2.0))
        y_max = int(round(y_center + box_height / 2.0))

        x_min = max(0, min(x_min, width))
        y_min = max(0, min(y_min, height))
        x_max = max(0, min(x_max, width))
        y_max = max(0, min(y_max, height))

        if x_max > x_min and y_max > y_min:
            boxes.append((x_min, y_min, x_max, y_max, cls_id))

    return boxes


def convert_split(split_root: Path, output_path: Path, image_ext: str = ".jpg") -> None:
    """Convert one split, such as train/val/test, into a COCO txt file."""
    labels_root = split_root / "labels"
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_root}")

    label_files = find_label_files(labels_root)
    lines: List[str] = []

    for label_path in label_files:
        image_path = image_path_from_label(label_path, split_root, image_ext)
        if not image_path.exists():
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        boxes = load_yolo_boxes(label_path, (width, height))
        line = str(image_path.resolve())
        for x_min, y_min, x_max, y_max, cls_id in boxes:
            line += f" {x_min},{y_min},{x_max},{y_max},{cls_id}"
        lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Wrote {len(lines)} lines to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CST_AntiUAV/frhybrid to COCO txt format")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/CST_AntiUAV/frhybrid"),
        help="Path to the frhybrid dataset root",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to convert, for example: train val test",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where coco_*.txt files will be written",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="coco",
        help="Output file prefix, e.g. coco -> coco_train.txt",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="CST",
        help="Dataset suffix used in output filenames, e.g. CST -> coco_train_CST.txt",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default=".jpg",
        help="Image extension used inside the images directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()

    for split in args.splits:
        split_root = dataset_root / split
        output_name = f"{args.prefix}_{split}_{args.suffix}.txt"
        output_path = output_dir / output_name
        convert_split(split_root, output_path, image_ext=args.image_ext)


if __name__ == "__main__":
    main()