# CST_AntiUAV frhybrid 训练说明

本文记录这次为 CST_AntiUAV frhybrid 数据集补充的转换脚本、数据读取修复，以及训练方式。

## 1. 数据格式

CST_AntiUAV 目录结构如下：
```
train/
val/
test/
```

每个 split 下都有 images 和 labels 两类目录，标签为逐帧 YOLO txt。每个标签文件对应一张图片。

## 2. 新增的转换脚本

新增脚本：

[utils_coco/cst_antiuav_to_txt.py](../utils_coco/cst_antiuav_to_txt.py)

作用是把 frhybrid 的 YOLO 标注转换成 SSTNet 需要的 COCO 风格 txt。输出每一行的格式与 [coco_train_IRDST.txt](../coco_train_IRDST.txt) 一致：

图片绝对路径 x1,y1,x2,y2,class x1,y1,x2,y2,class ...

生成命令：
```
python utils_coco/cst_antiuav_to_txt.py --dataset-root datasets/CST_AntiUAV/frhybrid --splits train val --output-dir . --prefix coco --suffix CST
```
如果需要测试集，也可以把 splits 改成 train val test。

## 3. 训练入口

当前训练脚本是 [train_IRDST.py](../train_IRDST.py)。它读取的是两个 txt 文件路径：

```
train_annotation_path
val_annotation_path
```

如果你要训练 CST_AntiUAV，只需要把它们改成：

```
train_annotation_path = 'coco_train_CST.txt'
val_annotation_path = 'coco_val_CST.txt'
```

然后在仓库根目录执行：

```
CUDA_VISIBLE_DEVICES=0 uv run train_IRDST.py
```


## 4. 这次修掉的问题

原来的序列数据读取逻辑硬编码了帧文件后缀和帧号回退方式，导致 CST_AntiUAV 的 jpg 帧无法正确加载。现在 [utils/dataloader_for_IRDST.py](../utils/dataloader_for_IRDST.py) 和 [utils/dataloader_for_DAUB.py](../utils/dataloader_for_DAUB.py) 已经改成：

1. 自动根据 txt 里的路径推断图片后缀。
2. 自动按零填充数字去找同目录下的历史帧。
3. 序列开头不再回退到不存在的 0 帧，而是回退到该序列里实际存在的最早帧。

## 5. 已验证结果

这次已经验证过：

1. 转换脚本可正常生成 coco_train_CST.txt 和 coco_val_CST.txt。
2. 数据加载器可通过语法检查。
3. 训练时不再卡在寻找 .bmp 帧或不存在的 0 帧上。

## 6. PR 曲线计算

先把 val 的 txt 转成 COCO json：

```
python utils_coco/cst_txt_to_coco_json.py --txt-path coco_val_CST.txt --output-json json/CST_instances_val2017.json
```

再运行 PR 曲线评估脚本：

```
python vid_map_cst.py --coco-gt json/CST_instances_val2017.json --model-path logs/your_best_checkpoint.pth
```

结果会保存到 `map_out/cst_eval/`，其中包含 `eval_results.json` 和 `pr_curve.png`。
