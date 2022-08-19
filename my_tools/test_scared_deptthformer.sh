CONFIG_FILE='configs/depthformer/depthformer_scared.py'   #adabins/custom_scared.py'
CHECKPOINT_FILE='work_dirs/depthformer_scared/latest.pth'     #'../adabins_efnetb5_kitti.pth'
RESULT_FILE='./results_scared_df'
GPU_NUM=2
#EVAL_METRICS='miou'

#./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --show-dir=${RESULT_FILE} #--eval ${EVAL_METRICS}


python tools/test.py $CONFIG_FILE $CHECKPOINT_FILE --show-dir $RESULT_FILE
