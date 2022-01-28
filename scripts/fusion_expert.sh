#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /horizontal_expert.sh
# \brief:
# Author: raphael hao
# batchsizes=(1 2 4 8 16 32 64 128 256 512 1024)
# tokens=(10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100)

usage() {
  echo "Usage: $0 [-t <search|report_best|export>] [-e <expert number>]" 1>&2
  exit 1
}

while getopts ":t:e:r:h" o; do
  case "${o}" in
  t)
    task=${OPTARG}
    ((task == "search" || task == "report_best" || task == "export_best")) || usage
    ;;
  e)
    expert_num=${OPTARG}
    ((expert_num >= 1 && expert_num <= 6)) || usage
    ;;
  r)
    resume=true
    ;;
  *)
    usage
    ;;
  esac
done
shift $((OPTIND - 1))

if [ -z "${t}" ] || [ -z "${e}" ]; then
  usage
fi

cd microbench/fusion_expert || exit

echo "task: ${task}, candidate_expert: ${expert_num}"
for ((i = 1; i <= expert_num; i*=2)); do
  python tvm_expert.py --task "${task}" --type all --M 40 --K 512 --N 1024 --E 2 --batch 2
done

python tvm_expert.py --task search --type all --M 40 --K 512 --N 1024 --E 2 --batch 2
python tvm_expert.py --task search --type fusion --M 40 --K 512 --N 1024 --E 4 --batch 4
python tvm_expert.py --task search --type fusion --M 40 --K 512 --N 1024 --E 8 --batch 8
python tvm_expert.py --task search --type fusion --M 40 --K 512 --N 1024 --E 16 --batch 16
python tvm_expert.py --task search --type fusion --M 40 --K 512 --N 1024 --E 32 --batch 32

python tvm_expert.py --task search --type all --M 128 --K 512 --N 1024 --E 2 --batch 2
python tvm_expert.py --task search --type fusion --M 128 --K 512 --N 1024 --E 4 --batch 4
python tvm_expert.py --task search --type fusion --M 128 --K 512 --N 1024 --E 8 --batch 8
python tvm_expert.py --task search --type fusion --M 128 --K 512 --N 1024 --E 16 --batch 16
python tvm_expert.py --task search --type fusion --M 128 --K 512 --N 1024 --E 32 --batch 32

python tvm_expert.py --task search --type all --M 128 --K 1024 --N 1024 --E 2 --batch 2
python tvm_expert.py --task search --type fusion --M 128 --K 1024 --N 1024 --E 4 --batch 4
python tvm_expert.py --task search --type fusion --M 128 --K 1024 --N 1024 --E 8 --batch 8
python tvm_expert.py --task search --type fusion --M 128 --K 1024 --N 1024 --E 16 --batch 16
python tvm_expert.py --task search --type fusion --M 128 --K 1024 --N 1024 --E 32 --batch 32

python tvm_expert.py --task search --type all --M 128 --K 3072 --N 768 --E 2 --batch 2
python tvm_expert.py --task search --type fusion --M 128 --K 3072 --N 768 --E 4 --batch 4
python tvm_expert.py --task search --type fusion --M 128 --K 3072 --N 768 --E 8 --batch 8
python tvm_expert.py --task search --type fusion --M 128 --K 3072 --N 768 --E 16 --batch 16
python tvm_expert.py --task search --type fusion --M 128 --K 3072 --N 768 --E 32 --batch 32
