_base_ = [
    '../_base_/models/densedepth.py', '../_base_/datasets/scared.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_24x.py'
]

model = dict(
    decode_head=dict(
        scale_up=True,
        min_depth=1e-3,
        max_depth=200,
        loss_decode=dict(
            type='SigLoss', valid_mask=True, loss_weight=1.0)),
    )
