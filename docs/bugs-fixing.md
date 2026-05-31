# Bugs 修复记录

## CUDA / autograd 相关错误修复记录

摘要
- 本文档记录在训练中遇到的两个会导致 CUDA 抛断言或反向传播失败的常见错误，以及仓库中已做的修复和验证方法。

触发现象
- 训练中出现 `Assertion input_val >= zero && input_val <= one`（BCE 输入断言）或 `CUDA error: device-side assert triggered`。
- 或者出现 `one of the variables needed for gradient computation has been modified by an inplace operation` 的 RuntimeError。

根因与修改
- 溢出导致非有限值：模型在解码 box 宽高时使用 `exp()`，单个批次若 logits 很大会造成 overflow/Inf，进而使 IoU / BCE 出现断言。
  - 修复：在 `nets/yolo_training.py::get_output_and_grid` 中对宽高 logits 使用 `clamp(max=11.0)` 再 `exp()`，避免指数溢出。

- 原地修改导致 autograd 版本冲突：代码中对 `output[..., :2]` 和 `output[..., 2:4]` 做了切片赋值（原地写），在后续计算中会修改需要用于反向传播的中间变量版本号。
  - 修复：改为先计算 `xy`、`wh`（非原地），再用 `torch.cat((xy, wh, output[...,4:]), dim=-1)` 生成新的张量，避免原地写入。

- 非有限训练目标未提前报错：当目标或中间值变为 NaN/Inf 时，CUDA 断言通常会在后续核里异步触发，堆栈不利于定位。
  - 修复：在 `nets/yolo_training.py::get_losses` 中加入 `torch.isfinite` 检查，若发现非有限目标则立刻抛出 `FloatingPointError`，提示更明确的根因方向（标注 / 增强 / 学习率）。

受影响文件
- `nets/yolo_training.py` — 解码（get_output_and_grid）、损失构造（get_losses）处有改动。

如何验证
1. 重新运行训练（尽量先在小数据集+batch 上跑）。
2. 若复现原问题，建议先：
   - 关闭 `mosaic` / `mixup`；
   - 将学习率调低一档；
   - 设置 `CUDA_LAUNCH_BLOCKING=1` 以同步 CUDA 错误并定位发生位置；
   - 开启 `torch.autograd.set_detect_anomaly(True)` 可帮助定位具体触发的原地操作（仅用于调试）。
3. 若出现 `FloatingPointError: Non-finite training targets detected.`，说明问题更可能出在标注或数据增强，检查相应样本的标注/增强逻辑。

回滚建议
- 若需要回退这些改动，可以使用 git 回滚对应提交，例如：

```bash
git checkout -- nets/yolo_training.py
```

联系方式
- 若仍无法定位，欢迎打开 issue 并附上训练时的 stderr 与少量能复现的样本路径，方便 debug。
