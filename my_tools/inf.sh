CONFIG_FILE='configs/adabins/custom_ada.py'
CHECKPOINT_FILE='work_dirs/custom_ada/latest.pth'     #'../adabins_efnetb5_kitti.pth'
RESULT_FILE='../results'
GPU_NUM=2
#EVAL_METRICS='miou'

#./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --show-dir=${RESULT_FILE} #--eval ${EVAL_METRICS}


python tools/test.py $CONFIG_FILE $CHECKPOINT_FILE --show-dir $RESULT_FILE --eval miou