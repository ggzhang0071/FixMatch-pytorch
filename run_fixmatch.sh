
mkdir -p Logs


#python   dataset/shezhen_data_processing.py

CUDA_VISIBLE_DEVICES=0,1 

python  train.py --dataset shezhen --label_root /git/datasets/fixmatch_dataset \
    --unlabeled_root    /git/datasets/shezhen_original_data/shezhen_unlabeled_data --num-labeled 4000 --arch wideresnet \
     --image_size 32 --batch-size 128  --lr 0.03 --expand-labels --seed 5 --out results/shezhen@4000.5 \
     2>&1 | tee Logs/training_$(date +%Y%m%d_%H%M%S).log

