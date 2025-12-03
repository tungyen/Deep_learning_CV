export PYTHONPATH=$(pwd)
NUM_GPU=$1
EXP=$2
CONFIG=$3

torchrun --nproc_per_node=${NUM_GPU} Segmentation_3d/PointNet/pipelines/train.py \
--exp ${EXP} --config_path ${CONFIG}

# torchrun --nproc_per_node=${NUM_GPU} Object_detection_2d/CenterNet/pipelines/eval.py \
# --exp ${EXP} --config_path ${CONFIG}

# torchrun --nproc_per_node=${NUM_GPU} Object_detection_2d/CenterNet/pipelines/test.py \
# --exp ${EXP} --config_path ${CONFIG}