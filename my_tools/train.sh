CONFIG_FILE='configs/adabins/custom_ada.py'
GPU_NUM=4


tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}