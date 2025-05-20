img="nvcr.io/nvidia/pytorch:21.02-py3" 
# img="padim:0.1"

docker run --rm  --gpus all --privileged=true  --workdir /git --name "fixmatch"  -e DISPLAY --ipc=host -d --rm  -p 5233:8889  \
-v /mnt/newdisk/she_zhen_code/FixMatch-pytorch:/git/fixmatch/ \
-v /mnt/newdisk/shezhen_original_data:/git/datasets/shezhen_original_data \
$img sleep infinity


docker exec -it fixmatch /bin/bash


