import argparse
import colorsys
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from nets.Network import Network
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image, show_config
from utils.utils_bbox import decode_outputs, non_max_suppression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CST_AntiUAV and plot PR curve")
    parser.add_argument("--coco-gt", type=Path, default=Path("json/CST_instances_val2017.json"))
    parser.add_argument("--model-path", type=Path, default=Path("model_data/pre_trained.pth"))
    parser.add_argument("--classes-path", type=Path, default=Path("model_data/classes.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("map_out/cst_eval"))
    parser.add_argument("--input-shape", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--phi", type=str, default="s")
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.65)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_false", dest="cuda")
    parser.add_argument("--num-frame", type=int, default=5)
    parser.add_argument("--save-name", type=str, default="pr_curve.png")
    return parser.parse_args()


def _resolve_frame_path(image_dir: Path, frame_index: int, stem_width: int, suffix: str) -> Path:
    suffixes = [suffix] if suffix else [".jpg", ".jpeg", ".png", ".bmp"]
    for ext in suffixes:
        padded = image_dir / f"{frame_index:0{stem_width}d}{ext}"
        if padded.exists():
            return padded
        plain = image_dir / f"{frame_index}{ext}"
        if plain.exists():
            return plain
    raise FileNotFoundError(f"No frame found for index {frame_index} under {image_dir}")


def _get_min_frame_index(image_dir: Path, suffix: str) -> int:
    suffixes = [suffix] if suffix else [".jpg", ".jpeg", ".png", ".bmp"]
    frame_ids = []
    for ext in suffixes:
        for candidate in image_dir.glob(f"*{ext}"):
            if candidate.stem.isdigit():
                frame_ids.append(int(candidate.stem))
    return min(frame_ids) if frame_ids else 0


def get_history_imgs(image_path: str, num_frame: int) -> list[str]:
    image_file = Path(image_path)
    image_dir = image_file.parent
    stem_width = len(image_file.stem)
    suffix = image_file.suffix
    image_id = int(image_file.stem)
    min_frame_index = _get_min_frame_index(image_dir, suffix)

    history = []
    for offset in range(num_frame - 1, -1, -1):
        frame_index = max(image_id - offset, min_frame_index)
        history.append(str(_resolve_frame_path(image_dir, frame_index, stem_width, suffix)))
    return history


class MAPVid(object):
    def __init__(self, **kwargs):
        self.model_path = kwargs.get("model_path")
        self.classes_path = kwargs.get("classes_path")
        self.input_shape = kwargs.get("input_shape")
        self.phi = kwargs.get("phi")
        self.confidence = kwargs.get("confidence")
        self.nms_iou = kwargs.get("nms_iou")
        self.letterbox_image = True
        self.cuda = kwargs.get("cuda")
        self.num_frame = kwargs.get("num_frame", 5)

        self.class_names, self.num_classes = get_classes(str(self.classes_path))
        hsv_tuples = [(x / self.num_classes, 1.0, 1.0) for x in range(self.num_classes)]
        self.colors = [tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*x)) for x in hsv_tuples]
        self.generate()

    def generate(self, onnx: bool = False):
        self.net = Network(self.num_classes, num_frame=self.num_frame)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dict = self.net.state_dict()
        pretrained_dict = torch.load(self.model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}

        for k, v in pretrained_dict.items():
            key = k[7:] if k.startswith("module.") else k
            if key in model_dict.keys() and np.shape(model_dict[key]) == np.shape(v):
                temp_dict[key] = v
                load_key.append(key)
            else:
                no_load_key.append(k)

        model_dict.update(temp_dict)
        self.net.load_state_dict(model_dict)
        self.net = self.net.eval()
        print(f"{self.model_path} model, and classes loaded.")
        print("Successful Load Key Num:", len(load_key))
        print("Fail To Load Key Num:", len(no_load_key))
        if not onnx and self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image_id, images, results, cat_ids):
        image_shape = np.array(np.shape(images[0])[0:2])
        images = [cvtColor(image) for image in images]
        image_data = [resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image) for image in images]
        image_data = [np.transpose(preprocess_input(np.array(image, dtype="float32")), (2, 0, 1)) for image in image_data]
        image_data = np.stack(image_data, axis=1)
        image_data = np.expand_dims(image_data, 0)

        with torch.no_grad():
            batch = torch.from_numpy(image_data)
            if self.cuda:
                batch = batch.cuda()
            outputs = self.net(batch)
            outputs = decode_outputs(outputs, self.input_shape)
            outputs = non_max_suppression(
                outputs,
                self.num_classes,
                self.input_shape,
                image_shape,
                self.letterbox_image,
                conf_thres=self.confidence,
                nms_thres=self.nms_iou,
            )

            if outputs[0] is None:
                return results

            top_label = np.array(outputs[0][:, 6], dtype="int32")
            top_conf = outputs[0][:, 4] * outputs[0][:, 5]
            top_boxes = outputs[0][:, :4]

        for i, c in enumerate(top_label):
            top, left, bottom, right = top_boxes[i]
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(cat_ids[c]),
                    "bbox": [float(left), float(top), float(right - left), float(bottom - top)],
                    "score": float(top_conf[i]),
                }
            )
        return results


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coco_gt = COCO(str(args.coco_gt))
    img_ids = coco_gt.getImgIds()
    cat_ids = coco_gt.getCatIds()

    show_config(
        coco_gt=str(args.coco_gt),
        model_path=str(args.model_path),
        classes_path=str(args.classes_path),
        input_shape=args.input_shape,
        phi=args.phi,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        cuda=args.cuda,
        num_frame=args.num_frame,
    )

    yolo = MAPVid(
        model_path=args.model_path,
        classes_path=args.classes_path,
        input_shape=args.input_shape,
        phi=args.phi,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        cuda=args.cuda,
        num_frame=args.num_frame,
    )

    results = []
    results_json = args.output_dir / "eval_results.json"

    with results_json.open("w", encoding="utf-8") as handle:
        for image_id in tqdm(img_ids):
            image_path = coco_gt.loadImgs(image_id)[0]["file_name"]
            images = [Image.open(item) for item in get_history_imgs(image_path, args.num_frame)]
            results = yolo.detect_image(image_id, images, results, cat_ids)
        json.dump(results, handle)

    coco_dt = coco_gt.loadRes(str(results_json))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    precisions = coco_eval.eval["precision"]
    precision_50 = precisions[0, :, 0, 0, -1]
    precision_50 = np.maximum(precision_50, 0)
    recalls = coco_eval.eval["recall"]
    recall_50 = float(recalls[0, 0, 0, -1])

    valid_precision = precision_50[: max(int(round(recall_50 * 100)) + 1, 1)]
    mean_precision = float(np.mean(valid_precision)) if valid_precision.size else 0.0
    f1 = 0.0 if mean_precision + recall_50 == 0 else 2 * recall_50 * mean_precision / (recall_50 + mean_precision)
    print(f"Precision: {mean_precision:.4f}, Recall: {recall_50:.4f}, F1: {f1:.4f}")

    recall_axis = np.linspace(0.0, 1.0, len(precision_50))
    plt.figure(figsize=(7, 6))
    plt.title("PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.plot(recall_axis, precision_50, linewidth=2)
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(args.output_dir / args.save_name, dpi=200)
    plt.close()
    print("Get map done.")


if __name__ == "__main__":
    main()