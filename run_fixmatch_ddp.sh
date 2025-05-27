
mkdir -p Logs
rm Logs/*


#python   dataset/shezhen_data_processing.py
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_NO_SHM_WRITER=1



TORCHELASTIC_ERROR_FILE=/tmp/torch_ddp_errors \
torchrun --nproc_per_node=2   --nnodes=1 --node_rank=0 train_ddp.py \
    --dataset shezhen --root /git/datasets/fixmatch_dataset \
  --num-labeled 4000 --arch wideresnet --image_size 32 \
  --batch-size 8 --lr 0.03 --amp --expand-labels \
  --seed 5 --out results/shezhen@4000.5
