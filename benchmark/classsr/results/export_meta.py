import json

from argparse import ArgumentParser
from typing import Dict, List, Union
from brt.runtime import BRT_KERNEL_TEMPLATE_PATH, BRT_LOG_PATH

from net_conv_meta import NET_CONV_META

if __name__ == "__main__":
    argp = ArgumentParser()
    argp.add_argument("json_filename", type=str)

    with open(argp.parse_args().json_filename) as f:
        jsonf = json.load(f)

    if "ClassSR_FSRCNN" in jsonf["name"]:
        meta = NET_CONV_META["ClassSR_FSRCNN"]
        subnet_name = "fsrcnn"
    elif "ClassSR_RCAN" in jsonf["name"]:
        meta = NET_CONV_META["ClassSR_RCAN"]
        subnet_name = "rcan"
    else:
        assert False, "Unsupported net type"

    for dataset in jsonf["dataset"]:
        print(dataset["name"])
        maxnums = [
            max(img["subimgs"][i] for img in dataset["imgs"]) for i in range(3)
        ]
        print(f"\t{maxnums=}")
        minnums = [
            min(img["subimgs"][i] for img in dataset["imgs"]) for i in range(3)
        ]
        print(f"\t{minnums=}")

        with open(
            BRT_LOG_PATH
            / f"benchmark/classsr/{subnet_name}/conv_params_{dataset['name']}.json",
            mode="w",
        ) as f:
            for subnet_meta, maxnum in zip(
                meta, maxnums
            ):
                for conv_meta in subnet_meta:
                    conv_meta["input_shape"][0] = maxnum
                    conv_meta["output_shape"][0] = maxnum
                    f.write(json.dumps(conv_meta))
                    f.write("\n")
