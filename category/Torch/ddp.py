"""
GPU多卡并行训练总结（以pytorch为例）
https://mp.weixin.qq.com/s/QhNHAjLt7WVFAArLqmAh4A
"""
import argparse
import os
import sys
import torch
import torch.distributed as dist
import torchvision


def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    # 如果是多机多卡的机器, WORLD_SIZE代表使用的机器数, RANK对应第几台机器
    # 如果是单机多卡的机器, WORLD_SIZE代表有几块GPU, RANK和LOCAL_RANK代表第几块GPU
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])  # LOCAL_RANK代表某个机器上第几块GPU
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    args.dist_backend = 'nccl'  # 通信后端, nvidia GPU推荐使用NCCL

    dist.init_process_group(backend=args.dist_backend, init_method='env://')

    torch.cuda.set_device(args.gpu)  # 对当前进程指定使用的GPU

    dist.barrier()  # 等待每个GPU都运行完这个地方以后再继续


def reduce_value(value, average=True):
    """ All reduce among GPUs """
    world_size = get_world_size()

    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)  # 对不同设备之间的value求和
        if average:  # 如果需要求平均, 获得多块GPU计算loss的均值
            value /= world_size

    return value


def train_one_epoch(model, optimizer, loss_function, data_loader, device, epoch):
    model.train()

    mean_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        loss = loss_function(pred, labels.to(device))

        loss.backward()

        loss = reduce_value(loss, average=True)  # 在单GPU中不起作用, 多GPU时, 获得所有GPU的loss的均值.
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # if is_main_process():
        #     print("[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3)))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 用于存储预测正确的样本个数, 每块GPU都会计算自己正确样本的数量
    sum_num = torch.zeros(1).to(device)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    # print("sum_num {} in rank {}".format(sum_num.item(), get_rank()))
    sum_num = reduce_value(sum_num, average=False)  # 预测正确样本个数
    # print("sum_num {} after all-reduce".format(sum_num.item()))

    return sum_num.item()


def ddp(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    device = torch.device(args.gpu)

    batch_size = args.batch_size
    num_classes = args.num_classes
    load_path = args.load_path
    ckp_path = args.ckp_path
    args.lr *= args.world_size  # 学习率要根据并行GPU的数倍增, 这里使用简单的倍增方法

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    train_data_set = torchvision.datasets.CIFAR10(
        root='./datasets/cifar10/', train=True, transform=train_transform,  download=True)
    test_data_set = torchvision.datasets.CIFAR10(
        root='./datasets/cifar10/', train=False, transform=test_transform, download=True)

    # DistributedSampler给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(test_data_set)
    # BatchSampler将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,  # 直接加载到显存中，达到加速效果
                                               num_workers=args.n_workers)
    val_loader = torch.utils.data.DataLoader(test_data_set,
                                             batch_size=batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=args.n_workers)

    # 实例化模型
    model = torchvision.models.resnet34(num_classes=num_classes).to(device)

    # 需要保证每块GPU加载的权重是一模一样的
    if load_path is not None and os.path.exists(load_path):
        # 如果存在预训练权重则载入
        weights_dict = torch.load(load_path, map_location=device)
        # 简单对比每层的权重参数个数是否一致
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        print("Load the initial weights in rank {} from {}".format(get_rank(), load_path))
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        # 如果不存在预训练权重, 需要将第一个进程中的权重保存, 然后其他进程载入, 保持初始化权重一致
        if not os.path.exists(ckp_path):
            os.makedirs(ckp_path)
        checkpoint_path = "{}/initial_weights.pth".format(ckp_path)
        if is_main_process():
            print("Save the initial weights in rank {}".format(get_rank()))
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()  # 等待每个GPU都运行完这个地方以后再继续
        print("Load the initial weights in rank {}".format(get_rank()))
        # 这里注意，一定要指定map_location参数, 否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:
        for name, param in model.named_parameters():
            # 除全连接层外, 其他权重全部冻结
            if "fc" not in name:
                param.requires_grad_(False)

    # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
    #     不使用同步BN, i.e., 每个设备计算自己的批次数据的均值方差: 效果与单GPU一致, 仅仅能提升训练速度;
    #     使用同步BN: 效果会有一定提升, 但是会损失一部分并行速度, i.e., 更耗时
    if args.syncBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer使用SGD+余弦淬火策略
    loss_function = torch.nn.CrossEntropyLoss()
    params_group = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params_group, lr=args.lr, momentum=0.9, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=10, eta_min=0.1*args.lr)

    for epoch in range(args.epochs):
        # 在每次迭代的时候获得一个不同的生成器,每一轮开始迭代获取数据之前设置随机种子,
        # 通过改变传进的epoch参数改变打乱数据顺序. 通过设置不同的随机种子,
        # 可以让不同GPU每轮拿到的数据不同. 后面的部分和单GPU相同.
        train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    loss_function=loss_function,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)
        if is_main_process():
            print("[Training] epoch {} mean loss {}".format(epoch, mean_loss))

        scheduler.step()

        sum_num = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)
        acc = sum_num / val_sampler.total_size
        if is_main_process():
            print("[Testing] epoch {} Acc {}".format(epoch, acc))

        # 保存模型的权重需要在主进程中进行保存.
        if is_main_process() and epoch % 10 == 0:
            torch.save(model.module.state_dict(), "{}/ckp-{}.pth".format(ckp_path, epoch))

    dist.destroy_process_group()  # 撤销进程组, 释放资源


def main():
    parser = argparse.ArgumentParser(description="PyTorch DDP")
    parser.add_argument('--local_rank', type=int, default=-1, help="local process rank.")  # 接受DDP传递的参数，必不可少
    parser.add_argument('--n_workers', type=int, default=1, help="number of workers.")

    parser.add_argument('--epochs', type=int, default=100, help="training epochs.")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size.")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes.")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate.")
    parser.add_argument('--load_path', type=str, help="path of initialized weights.")
    parser.add_argument('--ckp_path', type=str, help="path of saved models.")
    parser.add_argument('--freeze_layers', action='store_true', help="freeze some layers.")
    parser.add_argument('--syncBN', action='store_true', help="synchronization of Batch Normalization in DDP.")

    args = parser.parse_args()

    ddp(args)


if __name__ == '__main__':
    main()
