import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset.cifar import DATASET_GETTERS 
from utils import AverageMeter, accuracy
from torchvision import transforms
from dataset.shezhen_json import get_shezhen9
from sklearn.metrics import f1_score, accuracy_score
from torch.amp import autocast, GradScaler
scaler = GradScaler()
torch.multiprocessing.set_start_method('spawn', force=True)
import torch.distributed as dist
import multiprocessing as mp
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


logger = logging.getLogger(__name__)
best_acc = 0


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--root', default='/git/datasets/fixmatch_dataset',
                        type=str, help='path to labeled dataset')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100','shezhen'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='model architecture')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    args = parser.parse_args()
    return args



def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def create_model(args):
    if args.arch == 'wideresnet':
        import models.wideresnet as models
        model = models.build_wideresnet(depth=args.model_depth,
                                          widen_factor=args.model_width,
                                          dropout=0,
                                          num_classes=args.num_classes)
    elif args.arch == 'resnext':
        import models.resnext as models
        model = models.build_resnext(cardinality=args.model_cardinality,
                                       depth=args.model_depth,
                                       width=args.model_width,
                                       num_classes=args.num_classes)
    logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6))
    return model

def main(args):
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    args.world_size = dist.get_world_size() if dist.is_initialized() else 1
    print(f"local_rank: {args.local_rank}, world_size: {args.world_size}")

    args.multi_label = (args.dataset == 'shezhen')

    # ------------------- 初始化分布式环境 -------------------
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1
    else:
        args.n_gpu = torch.cuda.device_count()
        if args.n_gpu > 0:
            device = torch.device('cuda')
            args.world_size = args.n_gpu
        else:
            device = torch.device('cpu')
            args.world_size = 1
    args.device = device

    # ------------------- 日志设置 -------------------
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}")
    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    # ------------------- 模型结构设置 -------------------
    if args.dataset == 'cifar10':
        args.num_classes = 10
        args.model_depth, args.model_width = (28, 2) if args.arch == 'wideresnet' else (28, 4)
        if args.arch == 'resnext':
            args.model_cardinality = 4
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        args.model_depth, args.model_width = (28, 8) if args.arch == 'wideresnet' else (29, 64)
        if args.arch == 'resnext':
            args.model_cardinality = 8
    elif args.dataset == 'shezhen':
        args.num_classes = 9
        args.model_depth, args.model_width = (28, 2) if args.arch == 'wideresnet' else (28, 4)
        if args.arch == 'resnext':
            args.model_cardinality = 4
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # ------------------- 数据加载 -------------------
    if args.dataset == 'shezhen':
        labeled_dataset, unlabeled_dataset, test_dataset = get_shezhen9(args)
    else:
        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, args.data_dir)

    if args.local_rank == 0:
        torch.distributed.barrier()

    num_workers = 0
 

    # 判断是否使用 DDP（DistributedSampler）
    use_ddp = args.local_rank != -1

    # 创建 Sampler
    if use_ddp:
        labeled_sampler = DistributedSampler(labeled_dataset, shuffle=True)
        unlabeled_sampler = DistributedSampler(unlabeled_dataset, shuffle=True)
    else:
        labeled_sampler = RandomSampler(labeled_dataset)
        unlabeled_sampler = RandomSampler(unlabeled_dataset)

    # 构建 DataLoader
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=labeled_sampler,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=None,
    )

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=unlabeled_sampler,
        batch_size=args.batch_size * args.mu,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=None,
    )

    # 测试集保持顺序采样
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size * 2,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=None,
        )

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # ------------------- 模型、EMA、优化器 -------------------
    model = create_model(args)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov)
    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)

    # ------------------- EMA 模型 -------------------
    ema_model = None
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    # ------------------- Checkpoint 恢复 -------------------
    args.start_epoch = 0
    best_acc = 0
    if args.resume:
        if args.local_rank in [-1, 0]:
            logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(args.resume), "Checkpoint not found!"
        checkpoint = torch.load(args.resume, map_location='cpu')
        args.out = os.path.dirname(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint.get('best_acc', 0)
        model.load_state_dict(checkpoint['state_dict']) if not hasattr(model, 'module') else model.module.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    # ------------------- AMP 混合精度 -------------------
    scaler = GradScaler() if args.amp else None

    # ------------------- 训练日志和 SummaryWriter -------------------
    writer = None
    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(log_dir=f'runs/fixmatch_{args.dataset}_{args.num_labeled}_experiment')
        logger.info("***** Running training *****")
        logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
        logger.info(f"  Num Epochs = {args.epochs}")
        logger.info(f"  Batch size per GPU = {args.batch_size}")
        logger.info(f"  Total train batch size = {args.batch_size * args.world_size}")
        logger.info(f"  Total optimization steps = {args.total_steps}")

    optimizer.zero_grad(set_to_none=True)

    # ✅ 正确位置：执行训练函数
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, scaler, writer)

    return args, model, optimizer, scheduler, scaler, ema_model, best_acc, \
           labeled_trainloader, unlabeled_trainloader, test_loader, writer


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler, scaler, writer):
    global best_acc
    test_accs = []
    end = time.time()

    labeled_epoch = 0
    unlabeled_epoch = 0

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()

    for epoch in range(args.start_epoch, args.epochs):
        if isinstance(labeled_trainloader.sampler, DistributedSampler):
            labeled_trainloader.sampler.set_epoch(epoch)

        if isinstance(unlabeled_trainloader.sampler, DistributedSampler):
            unlabeled_trainloader.sampler.set_epoch(epoch)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        if not args.no_progress and args.local_rank in [-1, 0]:
            p_bar = tqdm(range(args.eval_step))

        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)

            batch_size = inputs_x.size(0)
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)

            with autocast(enabled=args.amp):
                logits = model(inputs)
                logits = de_interleave(logits, 2 * args.mu + 1)
                logits_x = logits[:batch_size]
                logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

                if args.multi_label:
                    Lx = F.binary_cross_entropy_with_logits(logits_x, targets_x.float(), reduction='mean')
                    pseudo_label = torch.sigmoid(logits_u_w.detach() / args.T)
                    mask = (pseudo_label >= args.threshold).float()
                    targets_u = mask.clone()
                    Lu = F.binary_cross_entropy_with_logits(logits_u_s, targets_u, reduction='none')
                    Lu = (Lu * mask).mean()
                else:
                    Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
                    pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
                    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(args.threshold).float()
                    Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

                loss = Lx + args.lambda_u * Lu

            optimizer.zero_grad(set_to_none=True)

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()

            if args.use_ema:
                ema_model.update(model)

            losses.update(loss.detach())
            losses_x.update(Lx.detach())
            losses_u.update(Lu.detach())
            mask_probs.update(mask.mean().detach())
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress and args.local_rank in [-1, 0] and batch_idx % 10 == 0:
                p_bar.set_description(
                    "Epoch {}/{}. Batch: {}/{}. Data: {:.3f}s. Batch: {:.3f}s. Loss: {:.4f}. Lx: {:.4f}. Lu: {:.4f}. Mask: {:.2f}".format(
                        epoch + 1, args.epochs, batch_idx + 1, args.eval_step,
                        data_time.avg, batch_time.avg, losses.avg, losses_x.avg, losses_u.avg, mask_probs.avg
                    ))
                p_bar.update(10)


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()  # 只设置一次
    if hasattr(torch, 'compile'):
        model = torch.compile(model)  # PyTorch 2.x 加速（可选）

    end = time.time()
    test_iter = tqdm(test_loader, disable=args.no_progress, mininterval=5.0)

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)

            with autocast():  # AMP 推理
                outputs = model(inputs)

                if args.multi_label:
                    loss = F.binary_cross_entropy_with_logits(outputs, targets)
                    all_targets.append(targets)  # 暂不 .cpu()
                    all_outputs.append(torch.sigmoid(outputs))  # 暂不 .cpu()
                else:
                    loss = F.cross_entropy(outputs, targets)
                    prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                    top1.update(prec1.item(), inputs.size(0))
                    top5.update(prec5.item(), inputs.size(0))

                losses.update(loss.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                if args.multi_label:
                    test_iter.set_description(
                        "Test {}/{}. Data: {:.3f}s. Batch: {:.3f}s. Loss: {:.4f}".format(
                            batch_idx + 1, len(test_loader), data_time.avg, batch_time.avg, losses.avg
                        )
                    )
                else:
                    test_iter.set_description(
                        "Test {}/{}. Data: {:.3f}s. Batch: {:.3f}s. Loss: {:.4f}. Top1: {:.2f}. Top5: {:.2f}".format(
                            batch_idx + 1, len(test_loader), data_time.avg, batch_time.avg, losses.avg, top1.avg, top5.avg
                        )
                    )

        if not args.no_progress:
            test_iter.close()

    if args.multi_label:
        # 统一收集后再迁移到 CPU，减少同步等待
        all_outputs = torch.cat(all_outputs).cpu()
        all_targets = torch.cat(all_targets).cpu()
        preds = (all_outputs > 0.5).int()
        acc = (preds == all_targets.int()).float().mean().item()
        f1 = f1_score(all_targets.numpy(), preds.numpy(), average='macro')

        logger.info("Multi-label accuracy: {:.2f}".format(acc * 100))
        logger.info("Multi-label F1 (macro): {:.2f}".format(f1 * 100))
        return losses.avg, acc * 100
    else:
        logger.info("Top-1 acc: {:.2f}".format(top1.avg))
        logger.info("Top-5 acc: {:.2f}".format(top5.avg))
        return losses.avg, top1.avg

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    args = get_args()
    main(args)
