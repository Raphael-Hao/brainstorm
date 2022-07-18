## Docker

```bash
sudo docker run -it --name=swin_moe --privileged --net=host --ipc=host --gpus=all -v /mnt:/mnt -v /mntdata:/mntdata -v
/opt:/opt2 -w=/mnt/data0/workdir zeliu98/pytorch:superbench-nvcr21.05-fixfusedlamb-itp-mmcv bash
```

## Tutel

```bash
git pull
cd 3rd_packages_20220218_fc1fp32
/opt/conda/bin/python3 -m pip uninstall tutel -y
/opt/conda/bin/python3 ./tutel/setup.py install
cd ..
```

## Data

```bash
cd ./data
chmod 777 ./azcopy
./azcopy copy "https://zeliuus.blob.core.windows.net/data/imagenet1000/?sv=2019-02-02&ss=btqf&srt=sco&st=2021-01-01T16%3A20%3A22Z&se=2022-12-31T23%3A23%3A00Z&sp=rwdlacup&sig=zBPVBV%2F3%2BLi%2FKz6jalmIX%2BtOWWCeTHrAKyTOwweGwNY%3D" ./ --recursive
cd ..
```

## Run

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
  --nnode=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=6500 \
  main.py --cfg configs/swinv2_moe_22182_64_patch4_window7_224_norm8log_cosine_s2it2_s3b1_gradS8_vitmoeloss_GwoN_bpr_cap2.yaml --batch-size 128 --data-path ./data/imagenet1000 --output ./results/MoE/ --tag Apr05
```

## Eval

### 22K  data

```bash
cd ./data
chmod 777 ./azcopy
./azcopy copy "https://zeliuus.blob.core.windows.net/largedata/fall11_whole/?sv=2020-10-02&st=2022-04-24T14%3A19%3A45Z&se=2022-05-25T14%3A19%3A00Z&sr=c&sp=rl&sig=uPyBaj9zL8bYQnDJyDlNfitaE%2FblBjBPA3J3fMHgbD0%3D" ./ --recursive
./azcopy copy "https://zeliuus.blob.core.windows.net/largedata/ILSVRC2011fall_tarball_map_train.txt?sv=2020-10-02&st=2022-04-24T14%3A19%3A45Z&se=2022-05-25T14%3A19%3A00Z&sr=c&sp=rl&sig=uPyBaj9zL8bYQnDJyDlNfitaE%2FblBjBPA3J3fMHgbD0%3D" ./
./azcopy copy "https://zeliuus.blob.core.windows.net/largedata/ILSVRC2011fall_tarball_map_val.txt?sv=2020-10-02&st=2022-04-24T14%3A19%3A45Z&se=2022-05-25T14%3A19%3A00Z&sr=c&sp=rl&sig=uPyBaj9zL8bYQnDJyDlNfitaE%2FblBjBPA3J3fMHgbD0%3D" ./ 
cd ..
```

### checkpoint

```bash
cd ./data
chmod 777 ./azcopy
./azcopy copy "https://zeliuus.blob.core.windows.net/largedata/moe_results/swinv2_moe_small_pre_nattn_cpb_patch4_window12_192_s2it2_s3b1_top1_vitmoeloss_GwoN_bpr_cap125_moedrop01_nobias_22k_32gpu_16exp/?sv=2020-10-02&st=2022-04-24T14%3A19%3A45Z&se=2022-05-25T14%3A19%3A00Z&sr=c&sp=rl&sig=uPyBaj9zL8bYQnDJyDlNfitaE%2FblBjBPA3J3fMHgbD0%3D" ./ --recursive

cd ..
```

### eval

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
  --nnode=4 --node_rank=<rank> \
  --master_addr=<master-ip> --master_port=6500 \
  main.py --cfg configs/swinv2_moe_small_pre_nattn_cpb_patch4_window12_192_s2it2_s3b1_top1_vitmoeloss_GwoN_bpr_cap125_moedrop01_nobias_22k_32gpu_16exp.yaml --batch-size 128 --data-path ./data/ --output ./results/MoE/ --eval --resume ./data/swinv2_moe_small_pre_nattn_cpb_patch4_window12_192_s2it2_s3b1_top1_vitmoeloss_GwoN_bpr_cap125_moedrop01_nobias_22k_32gpu_16exp/model.pth
  
```

