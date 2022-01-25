#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /horizontal_expert.sh
# \brief:
# Author: raphael hao
# batchsizes=(1 2 4 8 16 32 64 128 256 512 1024)
# tokens=(10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100)
cd ../fusion_expert || exit
# pyenv shell miniconda3-4.7.12
python tvm_expert.py --task search --type all --E 2
python tvm_expert.py --task search --type fusion --E 4
python tvm_expert.py --task search --type fusion --E 8
python tvm_expert.py --task search --type fusion --E 16
python tvm_expert.py --task search --type fusion --E 32
python tvm_expert.py --task search --type all --E 2 --M 40
python tvm_expert.py --task search --type fusion --E 4 --M 40
python tvm_expert.py --task search --type fusion --E 8 --M 40
python tvm_expert.py --task search --type fusion --E 16 --M 40
python tvm_expert.py --task search --type fusion --E 32 --M 40