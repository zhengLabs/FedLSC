import argparse

from training import LSTMFedAVG, LSTMFedSPF, LSTMFedSGD, LSTMFedAPF
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
    arg = public_utils.get_args_lstm()
    print(arg)

    if arg.type is None:
        print("type 为空，请重新输入！！！")
        exit()
    elif arg.type == "LSTMAVG":
        unfold_args(arg, LSTMFedAVG.train)
    elif arg.type == "LSTMSPF":
        unfold_args(arg, LSTMFedSPF.train)
    elif arg.type == "LSTMSGD":
        unfold_args(arg, LSTMFedSGD.train)
    elif arg.type == "LSTMAPF":
        unfold_args(arg, LSTMFedAPF.train)
