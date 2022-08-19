CONFIG_FILE='configs/depthformer/depthformer_scared.py'
#dpt/dpt_scared.py'
#'configs/depthformer/depthformer_scared.py'
#'configs/adabins/custom_scared.py'

GPU_NUM=4


tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}