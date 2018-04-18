# =======================================================================================
# change ourselves set to tfrecord format which is accepted by tensorflow object detection API
# input: PASCAL format
# output: nvp_train.record and nvp_val.record
# =======================================================================================

if [ ! ${4} ]; then
    echo "Error: nvp_to_tfrecord.sh command line parameter missing"
    exit
fi

which_model=${1}
dataset_path=${2}
root_path=${3}
record_path=${4}

input_path="${root_path}scripts/input_config/"
output_path="${root_path}scripts/output/"

source activate tensorflow-gpu-python2
FILTER_TXT="${output_path}trainval_filter.txt"
LABEL_MAP_PATH="${input_path}nvp_label_map_${which_model}.pbtxt"

python "${root_path}models/research/object_detection/dataset_tools/create_nvp_tf_record.py" \
    --data_dir=${dataset_path} \
    --label_map_path=${LABEL_MAP_PATH} \
    --list_txt=${FILTER_TXT} \
    --output_dir=${record_path}
