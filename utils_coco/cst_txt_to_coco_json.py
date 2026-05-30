"""
Convert CST_AntiUAV frhybrid COCO-style txt annotations into a COCO json file.

The generated json can be consumed by pycocotools.COCO / COCOeval to compute
AP, precision, recall, and PR curves.

Input line format:
    /abs/path/to/image.jpg x1,y1,x2,y2,class ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CST txt annotations to COCO json")
    parser.add_argument(
        "--txt-path",
        type=Path,
        default=Path("coco_val_CST.txt"),
        help="Path to the COCO-style txt annotation file",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("json/CST_instances_val2017.json"),
        help="Output COCO json path",
    )
    parser.add_argument(
        "--category-name",
        type=str,
        default="targ",
        help="Single category name used in the generated json",
    )
    parser.add_argument(
        "--category-id",
        type=int,
        default=1,
        help="Category id used in the generated json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.txt_path.exists():
        raise FileNotFoundError(f"Txt file not found: {args.txt_path}")

    images = []
    annotations = []
    annotation_id = 1

    lines = [line.strip() for line in args.txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for image_id, line in enumerate(lines, start=1):
        parts = line.split()
        image_path = Path(parts[0])
        with Image.open(image_path) as image:
            width, height = image.size

        images.append(
            {
                "id": image_id,
                "file_name": str(image_path),
                "width": width,
                "height": height,
            }
        )

        for box_info in parts[1:]:
            x_min, y_min, x_max, y_max, cls_id = map(float, box_info.split(","))
            box_width = x_max - x_min
            box_height = y_max - y_min
            if box_width <= 0 or box_height <= 0:
                continue

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(cls_id) + 1,
                    "bbox": [x_min, y_min, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": args.category_id,
                "name": args.category_name,
                "supercategory": "object",
            }
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(coco_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(images)} images and {len(annotations)} annotations to {args.output_json}")


if __name__ == "__main__":
    main()