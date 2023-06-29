import argparse

from training import FedAVG, FedSPF, FedSGD, FedAPF, FedSPF2, FedSPF3
from utils import public_utils


def unfold_args(args, fun):
    fun(
        seed=args.seed,
        client_num=args.client_num,
        batch_size=args.batch_size,
        epoch=args.epoch,
        local_sgd=args.local_sgd,
        lr=args.lr,
        dataset=args.dataset,
        distribution=args.distribution,
        class_num=args.class_num,
        mod_name=args.mod_name,
        check=args.check,
        cuda=args.cuda
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = public_utils.get_args()

    if arg.type is None:
        print("type 为空，请重新输入！！！")
        exit()
    elif arg.type == "FedAVG":
        unfold_args(arg, FedAVG.train)
    elif arg.type == "FedSPF":
        unfold_args(arg, FedSPF.train)
    elif arg.type == "FedSPF2":
        unfold_args(arg, FedSPF2.train)
    elif arg.type == "FedSPF3":
        unfold_args(arg, FedSPF3.train)
    elif arg.type == "FedSGD":
        unfold_args(arg, FedSGD.train)
    elif arg.type == "FedAPF":
        unfold_args(arg, FedAPF.train)
