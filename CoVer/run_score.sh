dataset=roc

CUDA_VISIBLE_DEVICES=0 nohup python train.py \
    --lr 1e-6 \
    --dataset ${dataset} \
    --epochs 1 \
    --hdim 200 \
    --batch-size 10 \
    --margin 0.1 \
    > results/${dataset}/score.train 2>&1 &
