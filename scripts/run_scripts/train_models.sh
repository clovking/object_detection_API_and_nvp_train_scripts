# =======================================================================================
# Running the Training Job
# need to modify parameter in ssd_mobilenet_v1_coco.config, 
#     total five places,both with fine_tune_checkpoint,input_path,label_map_path
# output: model.ckpt, a checkpoint file, will be generated in ssd_mobilenet_train_logs
# =======================================================================================
if [ ! ${1} ]; then
    echo "Error: train_models.sh, please enter root path"
    exit
fi

root_path=${1}

input_path="${root_path}scripts/input_config/"
output_path="${root_path}scripts/output/"

# remove original train logs
# rm -rf ${output_path}train_logs/*
# python3, need GPU to train
source activate tensorflow-gpu
# train
# use only one GPU to train
CUDA_VISIBLE_DEVICES="1" \
python ${root_path}models/research/object_detection/train.py \
    --logtostderr \
    --train_dir=${output_path}train_logs \
    --pipeline_config_path=${input_path}ssd_mobilenet_v1_nvp.config
