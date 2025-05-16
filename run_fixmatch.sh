#python train.py --dataset cifar10 --num-labeled 4000 --arch wideresnet --batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/cifar10@4000.5  --resume "./result/cifar10@4000/checkpoint.pth.tar"

python train.py --dataset shezhen --num-labeled 4000 --arch wideresnet \
--batch-size 64 --lr 0.03 --expand-labels --seed 5 --out results/shezhen@4000.5

