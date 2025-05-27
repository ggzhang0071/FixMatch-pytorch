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

logger = logging.getLogger(__name__)
best_acc = 0


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


def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--data_dir', default='/git/datasets/fixmatch_dataset',
                        type=str, help='path to dataset')
    parser.add_argument('--root', default='/git/datasets/fixmatch_dataset',
                        type=str, help='path to labeled dataset')
    parser.add_argument('--gpu-id', default='0', type=str,  # 修改为 str 支持多卡传入 e.g. "0,1,2"
                        help='id(s) for CUDA_VISIBLE_DEVICES')
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
                        help='train batchsize per GPU')
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
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', 'O3']")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    args = parser.parse_args()
    args.world_size = dist.get_world_size() if dist.is_initialized() else 1
    print(f"local_rank: {args.local_rank}, world_size: {args.world_size}")


    # multi-label flag
    args.multi_label = (args.dataset == 'shezhen')

    # -------------------
    # 初始化分布式环境
    # -------------------
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1
    else:
        if torch.cuda.is_available():
            if ',' in args.gpu_id:
                device_ids = [int(x) for x in args.gpu_id.split(',')]
                device = torch.device('cuda', device_ids[0])
                args.world_size = len(device_ids)
                args.n_gpu = len(device_ids)
            else:
                device = torch.device('cuda', int(args.gpu_id))
                args.world_size = 1
                args.n_gpu = 1
        else:
            device = torch.device('cpu')
            args.world_size = 1
            args.n_gpu = 0

    args.device = device

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

    # --------------------
    # 设置模型结构参数
    # --------------------
    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
    elif args.dataset == 'shezhen':
        args.num_classes = 9
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
    else:
        raise ValueError(f"Unknown dataset {args.dataset}, please check the dataset name.")

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # --------------------
    # 加载数据集（你这里根据自己项目改）
    # --------------------
    if args.dataset == 'shezhen':
        labeled_dataset, unlabeled_dataset, test_dataset = get_shezhen9(args)
    else:
        # 这里需要定义你的 DATASET_GETTERS 字典
        labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args, args.data_dir)

    if args.local_rank == 0:
        torch.distributed.barrier()

    # --------------------
    # DataLoader 及 Sampler
    # --------------------
    num_workers = min(4, (os.cpu_count() or 4) // args.world_size)

    LabeledSampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    UnlabeledSampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=LabeledSampler(labeled_dataset),
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=UnlabeledSampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size * 2,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # --------------------
    # 创建模型
    # --------------------
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

    model = create_model(args)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)

    # --------------------
    # 优化器和调度器
    # --------------------
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.total_steps)

    # --------------------
    # EMA
    # --------------------
    ema_model = None
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    # --------------------
    # 恢复训练
    # --------------------
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
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    # --------------------
    # AMP 支持
    # --------------------
    scaler = GradScaler() if args.amp else None

    # --------------------
    # 训练信息打印（主进程）
    # --------------------
    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(log_dir=f'runs/fixmatch_{args.dataset}_{args.num_labeled}_experiment')
        logger.info("***** Running training *****")
        logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
        logger.info(f"  Num Epochs = {args.epochs}")
        logger.info(f"  Batch size per GPU = {args.batch_size}")
        logger.info(f"  Total train batch size = {args.batch_size * args.world_size}")
        logger.info(f"  Total optimization steps = {args.total_steps}")

    optimizer.zero_grad(set_to_none=True)

    return args, model, optimizer, scheduler, scaler, ema_model, best_acc, \
           labeled_trainloader, unlabeled_trainloader, test_loader, writer if args.local_rank in [-1, 0] else None
    
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()

    for epoch in range(args.start_epoch, args.epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])

        for batch_idx in range(args.eval_step):
            try:
                inputs_x, targets_x = next(labeled_iter)
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_iter)

            try:
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)

            batch_size = inputs_x.size(0)
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2 * args.mu + 1).to(args.device)
            targets_x = targets_x.to(args.device)

            with autocast():
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
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if args.use_ema:
                ema_model.update(model)

            losses.update(loss.detach())
            losses_x.update(Lx.detach())
            losses_u.update(Lu.detach())
            mask_probs.update(mask.mean().detach())
            batch_time.update(time.time() - end)
            end = time.time()

            # tqdm 仅每10步更新一次
            if not args.no_progress and batch_idx % 10 == 0:
                p_bar.set_description(
                    f"Epoch {epoch+1}/{args.epochs} | Step {batch_idx}/{args.eval_step} | "
                    f"Loss: {losses.avg.item():.4f} | Lx: {losses_x.avg.item():.4f} | Lu: {losses_u.avg.item():.4f} | "
                    f"Mask: {mask_probs.avg.item():.2f} | Data: {data_time.avg:.3f}s | Batch: {batch_time.avg:.3f}s"
                )
                p_bar.update(10)

        if not args.no_progress:
            p_bar.close()

        # Evaluation
        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            args.writer.add_scalar('train/1.train_loss', losses.avg.item(), epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg.item(), epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg.item(), epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg.item(), epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            ema_to_save = ema_model.ema.module if hasattr(ema_model.ema, "module") else ema_model.ema

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()



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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)  # 非常关键
    args, unknown = parser.parse_known_args()
    
    # 这一步非常关键，argparse 需要将 local_rank 传给主函数
    import sys
    sys.argv += [f"--local_rank={args.local_rank}"]
    main()
