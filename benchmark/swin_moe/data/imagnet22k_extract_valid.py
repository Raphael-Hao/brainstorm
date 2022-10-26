# Copyright (c) 2022 by Microsoft Corporation.
# Licensed under the MIT license.
from typing import List
import json
from PIL import Image
import pathlib
import argparse
import multiprocessing as mp


def _load_image(path):
    im = Image.open(path)
    return im


def _save_image(im: Image, base_path: pathlib.Path, name: str):
    im_path = base_path / name
    im_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(base_path / name)
    return im


class Extractor(mp.Process):
    def __init__(self, ann_path, origin_path, new_path, start_id, end_id):
        super().__init__()
        self.ann_path = pathlib.Path(ann_path)
        self.origin_path = pathlib.Path(origin_path)
        self.new_path = pathlib.Path(new_path)
        self.start_id = start_id
        self.end_id = end_id

    def run(self):
        self.database = json.load(open(self.ann_path))
        for i in range(self.start_id, self.end_id):
            idb = self.database[i]
            try:
                images = _load_image(self.origin_path / idb[0])
                _save_image(images, self.new_path, idb[0])
            except Exception as e:
                print(f"failing to load {idb[0]} due to {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_path", type=str, required=True)
    parser.add_argument("--origin_path", type=str, required=True)
    parser.add_argument("--new_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    database = json.load(open(args.ann_path))
    all_images = len(database)
    print(f"Total images: {all_images}")
    num_workers = args.num_workers
    start_id = 0
    image_per_worker = int(all_images // num_workers)
    extractors: List[mp.Process] = []
    for i in range(num_workers):
        end_id = start_id + image_per_worker if i < num_workers - 1 else all_images
        extractors.append(
            Extractor(args.ann_path, args.origin_path, args.new_path, start_id, end_id)
        )
        extractors[-1].start()
        start_id += image_per_worker
    for extractor in extractors:
        extractor.join()

