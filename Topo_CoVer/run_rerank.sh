dataset=sis
threshold=1

CUDA_VISIBLE_DEVICES=0 nohup python rerank.py --lr 1e-6 --dataset ${dataset} \
    --hdim 200 --batch-size 10 --pair-runID 1669174612 --score-runID 1673265520 \
    --has-doc --diff-ss --threshold ${threshold} \
    > results/${dataset}/${dataset}_whole.train 2>&1 &
