python  object_detection/inference/infer_detections.py \
  --input_tfrecord_paths=/mnt/extdisk1/datasets/tfrecord/generic/nvp_val.record \
  --output_tfrecord_path=/mnt/extdisk1/datasets/tfrecord/generic/nvp_val_det2.record \
  --inference_graph=/home/hugo/work/models/nvp/generic/output/frozen_inference_graph.pb \
  --discard_image_pixels
