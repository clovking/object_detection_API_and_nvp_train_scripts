import os
import sys

num_steps = int(sys.argv[1])
which_model = sys.argv[2]
root_path = sys.argv[3]
fine_tune_model = sys.argv[4]
record_path = sys.argv[5]

input_config_path = "%s%s" % (root_path, "scripts\/input_config\/")
nvp_config = "%s%s" % (input_config_path, "ssd_mobilenet_v1_nvp.config")

# 修改ssd_mobilenet_v1_nvp.config中迭代步数
modify_steps_cmd = "sed -i '/num_steps/s/[0-9]\+/%d/g' %s" % (num_steps, nvp_config)
os.system(modify_steps_cmd)

# 修改ssd_mobilenet_v1_nvp.config中fine_tune_checkpoint
fine_tune_checkpoint = "%s%s%s\/%s" % (root_path, "scripts\/fine_tune_checkpoint\/", fine_tune_model, "model.ckpt")
modify_checkpoint_cmd = "sed -i '/fine_tune_checkpoint:/s/\/\(.*\)model.ckpt/%s/g' %s" % (fine_tune_checkpoint, nvp_config)   
os.system(modify_checkpoint_cmd)

# 修改ssd_mobilenet_v1_nvp.config中input_path
input_path = "%s" % (record_path)
modify_input_path_cmd = "sed -i '/input_path:/s/\/\(.*\)\//%s/g' %s" % (input_path, nvp_config)
os.system(modify_input_path_cmd)

# 修改ssd_mobilenet_v1_nvp.config中label_map_path
label_map_path = "%s%s%s%s" % (input_config_path, "nvp_label_map_", which_model, ".pbtxt")
modify_label_map_path = "sed -i '/label_map_path:/s/\/\(.*\)\.pbtxt/%s/g' %s" % (label_map_path, nvp_config)
os.system(modify_label_map_path)

# 修改validation_input_config.pbtxt中label_map_path
validation_pbtxt = "%s%s" % (input_config_path, "validation_input_config.pbtxt")
modify_validation_input_pbtxt = "sed -i '/label_map_path:/s/\/\(.*\)\.pbtxt/%s/g' %s" % (label_map_path, validation_pbtxt)
os.system(modify_validation_input_pbtxt)

# 修改validation_input_config.pbtxt中input_path
modify_validation_input_path_cmd = "sed -i '/input_path:/s/\/\(.*\)\//%s/g' %s" % (input_path, validation_pbtxt)
os.system(modify_validation_input_path_cmd)
