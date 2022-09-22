# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.

import json
from PIL import Image
import pathlib
import argparse


def _load_image(path):
    im = Image.open(path)
    return im


def _save_image(im: Image, base_path: pathlib.Path, name: str):
    im_path = base_path / name
    im_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(base_path / name)
    return im


def extract_valid(ann_path, origin_path, new_path):
    ann_path = pathlib.Path(ann_path)
    origin_path = pathlib.Path(origin_path)
    new_path = pathlib.Path(new_path)
    database = json.load(open(ann_path))
    for i in range(len(database)):
        idb = database[i]
        images = _load_image(origin_path / idb[0])
        _save_image(images, new_path, idb[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', type=str, required=True)
    parser.add_argument('--origin_path', type=str, required=True)
    parser.add_argument('--new_path', type=str, required=True)
    args = parser.parse_args()
    extract_valid(args.ann_path, args.origin_path, args.new_path)