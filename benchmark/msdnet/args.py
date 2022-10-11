import os
import glob
import time
import argparse

model_names = ["msdnet"]
from brt.runtime.benchmark import (
    BenchmarkArgumentManager,
    Benchmarker,
    CUDATimer,
    MemoryStats,
    profile,
)
arg_parser = argparse.ArgumentParser(description="Image classification PK main script")



exp_group = arg_parser.add_argument_group("exp", "experiment setting")
exp_group.add_argument(
    "--save",
    default="save/default-{}".format(time.time()),
    type=str,
    metavar="SAVE",
    help="path to the experiment logging directory" "(default: save/debug)",
)
exp_group.add_argument(
    "--resume", action="store_true", help="path to latest checkpoint (default: none)"
)
exp_group.add_argument('--parallel', action='store_true',
                       help='if parallel mode')
exp_group.add_argument(
    "--evalmode",
    default=None,
    choices=["anytime", "dynamic", "threshold"],
    help="which mode to evaluate",
)

exp_group.add_argument(
    "--init_routers", action="store_true", help="whether to initialize routers"
)

exp_group.add_argument(
    "--thresholds", default=
    ## block 3 gather path
    # [100000000.00000000,
    #  1000000.0000,
    #  0.69275671,
    #  -1]
    
    
    
    # block two block(args.grFactor
    # [100000000.00000000,
    #  0.85972512,
    #  0.69275671,
    #  -1]
    
    # # p=torch.tensor([0,0.3,0.2,0.2,0.3]) with this probablity block 1 block
    # [100000000.00000000,
    # 0.85972512,
    # 0.69275671,
    # 0.47197723]
    
    ## block the first block
    # [1,0.03818264,0.01631335,0.01181476]
    
    
    # threshold that cater to the probability of the p=torch.tensor([0.1,0.2,0.2,0.2,0.3])
    # [ 9.6454e-01,  8.6269e-01,  6.9252e-01,  4.7205e-01]
    [0.96454340,0.86269057, 0.69251990, 0.47205138]
    
    #non parallel allign(maybe oen error due to the precision but it is acceptable, we just need to elevate the precision)
    # [ 8.4275e-02,  3.8183e-02,  1.6313e-02,  1.1815e-02]
    # [0.08427517,0.03818264,0.01631335,0.01181476]
    
    
    ##parallel allign(maybe oen error due to the precision but it is acceptable, we just need to elevate the precision)
    # [ 8.4491e-02,  3.6570e-02,  1.7349e-02,  1.2810e-02]
    
    , type=float, nargs="+", help="threshold"
)


exp_group.add_argument(
    "--evaluate-from",
    default=None,
    type=str,
    metavar="PATH",
    help="path to saved checkpoint (default: none)",
)
exp_group.add_argument(
    "--print-freq",
    "-p",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 100)",
)
exp_group.add_argument("--seed", default=0, type=int, help="random seed")
exp_group.add_argument("--gpu", default=None, type=str, help="GPU available.")

# dataset related
data_group = arg_parser.add_argument_group("data", "dataset setting")
data_group.add_argument(
    "--data",
    metavar="D",
    default="cifar10",
    choices=["cifar10", "cifar100", "ImageNet"],
    help="data to work on",
)
data_group.add_argument(
    "--data-root", metavar="DIR", default="data", help="path to dataset (default: data)"
)
data_group.add_argument(
    "--use-valid", action="store_true", help="use validation set or not"
)
data_group.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)

# model arch related
arch_group = arg_parser.add_argument_group("arch", "model architecture setting")
arch_group.add_argument(
    "--arch",
    "-a",
    metavar="ARCH",
    default="resnet",
    type=str,
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: msdnet)",
)
arch_group.add_argument(
    "--reduction",
    default=0.5,
    type=float,
    metavar="C",
    help="compression ratio of DenseNet"
    " (1 means dot't use compression) (default: 0.5)",
)


# msdnet config
arch_group.add_argument("--nBlocks", type=int, default=1)
arch_group.add_argument("--nChannels", type=int, default=32)
arch_group.add_argument("--base", type=int, default=4)
arch_group.add_argument("--stepmode", type=str, choices=["even", "lin_grow"])
arch_group.add_argument("--step", type=int, default=1)
arch_group.add_argument("--growthRate", type=int, default=6)
arch_group.add_argument("--grFactor", default="1-2-4", type=str)
arch_group.add_argument("--prune", default="max", choices=["min", "max"])
arch_group.add_argument("--bnFactor", default="1-2-4")
arch_group.add_argument("--bottleneck", default=True, type=bool)


# training related
optim_group = arg_parser.add_argument_group("optimization", "optimization setting")

optim_group.add_argument(
    "--epochs",
    default=300,
    type=int,
    metavar="N",
    help="number of total epochs to run (default: 164)",
)
optim_group.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
optim_group.add_argument(
    "-b",
    "--batch-size",
    default=64,
    type=int,
    metavar="N",
    help="mini-batch size (default: 64)",
)
optim_group.add_argument(
    "--optimizer",
    default="sgd",
    choices=["sgd", "rmsprop", "adam"],
    metavar="N",
    help="optimizer (default=sgd)",
)
optim_group.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate (default: 0.1)",
)
optim_group.add_argument(
    "--lr-type",
    default="multistep",
    type=str,
    metavar="T",
    help="learning rate strategy (default: multistep)",
    choices=["cosine", "multistep"],
)
optim_group.add_argument(
    "--decay-rate",
    default=0.1,
    type=float,
    metavar="N",
    help="decay rate of learning rate (default: 0.1)",
)
optim_group.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum (default=0.9)"
)
optim_group.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
bench_arg_manager = BenchmarkArgumentManager(arg_parser)
bench_arg_manager.add_item("liveness")
bench_arg_manager.add_item("memory_plan")
bench_arg_manager.add_item("transform")

