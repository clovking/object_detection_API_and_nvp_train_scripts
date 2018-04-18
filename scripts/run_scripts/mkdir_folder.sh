# =======================================================================================
# mdir output folder 
# =======================================================================================
if [ ! ${1} ]; then
    echo "Error: mkdir_folder.sh, please enter root path"
    exit
fi
root_path=${1}
output="${root_path}scripts/output/"
frozen_inference_graph="${output}frozen_inference_graph"
train_logs="${output}train_logs"

if [ ! -d "$output" ]; then
	mkdir -p "$output" "$frozen_inference_graph" "$train_logs"
fi
if [ ! -d "$frozen_inference_graph" ]; then
	mkdir -p "$frozen_inference_graph"
fi
if [ ! -d "$train_logs" ]; then
	mkdir -p "$train_logs"
fi
