# =======================================================================================
# inferring detections
# input: a trained object detection model frozen_inference_graph.pb and pascal_val.record
# output: validation_detections.record
# =======================================================================================
if [ ! ${3} ]; then
    echo "Error: infer_detections.sh, please enter steps, root path and record path"
    exit
fi
num_steps=${1}
root_path=${2}
record_path=${3}

input_path="${root_path}scripts/input_config/"
output_path="${root_path}scripts/output/"

# python2, need python2 environment
source activate tensorflow-gpu-python2
python ${root_path}models/research/object_detection/inference/infer_detections.py \
    --input_tfrecord_paths=${record_path}nvp_val.record \
    --output_tfrecord_path=${record_path}nvp_val_detections.record \
    --inference_graph=${output_path}frozen_inference_graph/frozen_inference_graph_${num_steps}/frozen_inference_graph.pb \
    --discard_image_pixels


