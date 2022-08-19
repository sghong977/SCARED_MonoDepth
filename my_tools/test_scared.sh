CONFIG_FILE='configs/simipu/sim_scared.py'   #adabins/custom_scared.py'
CHECKPOINT_FILE='work_dirs/sim_scared/latest.pth'     #'../adabins_efnetb5_kitti.pth'
RESULT_FILE='./inference_scared_sim_20220810'
GPU_NUM=2
EVAL_METRICS='miou'

#./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --show-dir=${RESULT_FILE} #--eval ${EVAL_METRICS}


python tools/test.py $CONFIG_FILE $CHECKPOINT_FILE --eval ${EVAL_METRICS}  # --show-dir $RESULT_FILE