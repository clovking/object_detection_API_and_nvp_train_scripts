# =======================================================================================
# evaluate models
# input: validation_eval_config.pbtxt, which sets metrics and
#        validation_input_config.pbtxt, which sets input_path(validation_detections.record)
# output: metrics.csv will be generated in validation_eval_metrics folder
# =======================================================================================
# python2, need python2 environment
if [ ! ${1} ]; then
    echo "Error: evaluate_models.sh, please enter root path"
    exit
fi
root_path=${1}

input_path="${root_path}scripts/input_config/"
output_path="${root_path}scripts/output/"

source activate tensorflow-gpu-python2
python ${root_path}models/research/object_detection/metrics/offline_eval_map_corloc.py \
    --eval_dir=${output_path} \
    --eval_config_path=${input_path}validation_eval_config.pbtxt \
    --input_config_path=${input_path}validation_input_config.pbtxt
