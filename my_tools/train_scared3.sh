CONFIG_FILE='configs/simipu/sim_scared2.py'
#'configs/depthformer/depthformer_scared.py'
#'configs/adabins/custom_scared.py'

GPU_NUM=4


tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM}