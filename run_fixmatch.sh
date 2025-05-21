#python -m pdb train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/cifar10@4000.5  


#python   dataset/shezhen_data_processing.py
python -m pdb  train.py --dataset shezhen --label_root /git/datasets/fixmatch_dataset \
    --unlabeled_root    /git/datasets/shezhen_original_data/shezhen_unlabel_data --num-labeled 4000 --arch wideresnet \
     --image_size 32 --batch-size 1 --eval-step 2  --lr 0.03 --expand-labels --seed 5 --out results/shezhen@4000.5



