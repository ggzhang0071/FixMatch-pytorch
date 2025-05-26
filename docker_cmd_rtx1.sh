img="nvcr.io/nvidia/pytorch:25.02-py3" 
# img="padim:0.1"

docker run --rm  --gpus all --privileged=true  --workdir /git --name "fixmatch"  -e DISPLAY --ipc=host -d --rm  -p 5233:8889  \
-v /home/fengyun/shezhen_code/FixMatch-pytorch:/git/fixmatch/ \
-v /home/fengyun/datasets/shezhen_original_data:/git/datasets/shezhen_original_data \
$img sleep infinity


docker exec -it fixmatch /bin/bash

