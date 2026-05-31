#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import argparse
import time

import torch
from thop import clever_format, profile
from torchsummary import summary
from torch.utils.data import DataLoader

from nets.Network import Network
from utils.dataloader_for_DAUB import dataset_collate, seqDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Model summary / FLOPs / FPS benchmark")
    parser.add_argument("--input-shape", type=int, nargs=2, default=[512, 512], help="Input image shape: H W")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of classes")
    parser.add_argument("--num-frame", type=int, default=5, help="Number of frames used by the model")
    parser.add_argument("--val-annotation-path", type=str, default="coco_val_DAUB.txt", help="Validation txt path")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max iterations for FPS benchmark")
    parser.add_argument("--log-interval", type=int, default=50, help="Log FPS every N iterations")
    parser.add_argument("--num-warmup", type=int, default=5, help="Warmup iterations ignored in FPS")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for FPS benchmark")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    input_shape = args.input_shape
    num_classes = args.num_classes
    num_frame = args.num_frame
    
    # 需要使用device来指定网络在GPU还是CPU运行
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    m = Network(num_classes, num_frame=num_frame).to(device)
    summary(m, (3, num_frame, input_shape[0], input_shape[1]))
    
    dummy_input = torch.randn(1, 3, num_frame, input_shape[0], input_shape[1]).to(device)
    flops, params = profile(m.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
    # 计算FPS

    max_iter = args.max_iter
    log_interval = args.log_interval
    num_warmup = args.num_warmup
    pure_inf_time = 0
    fps = 0
    val_annotation_path = args.val_annotation_path
    val_dataset = seqDataset(val_annotation_path, input_shape[0], num_frame, 'val')
    gen_val = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=use_cuda,
                                    drop_last=True, collate_fn=dataset_collate)
    # benchmark with 2000 image and take the average
    for i, data in enumerate(gen_val):
        if use_cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            m(data[0].to(device))

        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    print("FPS:" ,fps)
