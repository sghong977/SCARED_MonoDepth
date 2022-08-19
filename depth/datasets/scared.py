from logging import raiseExceptions
import os.path as osp
import warnings
from collections import OrderedDict
from functools import reduce

import tifffile as tiff

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from depth.core import pre_eval_to_metrics, metrics, eval_metrics
from depth.utils import get_root_logger
from depth.datasets.builder import DATASETS
from depth.datasets.pipelines import Compose

from depth.ops import resize

from PIL import Image

import torch
import os

import cv2


@DATASETS.register_module()
class SCAREDDataset(Dataset):
    """Custom dataset for supervised monocular depth esitmation. 
    An example of file structure. is as followed.
    .. code-block:: none
        ├── data
        │   ├── custom
        │   │   ├── train
        │   │   │   ├── rgb
        │   │   │   │   ├── 0.xxx
        │   │   │   │   ├── 1.xxx
        │   │   │   │   ├── 2.xxx
        │   │   │   ├── depth
        │   │   │   │   ├── 0.xxx
        │   │   │   │   ├── 1.xxx
        │   │   │   │   ├── 2.xxx
        │   │   ├── val
        │   │   │   ...
        │   │   │   ...

    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        data_root (str, optional): Data root for img_dir.
        test_mode (bool): test_mode=True
        min_depth=1e-3: Default min depth value.
        max_depth=10: Default max depth value.
    """


    def __init__(self,
                 pipeline,
                 data_root,
                 split=None,
                depth_scale=256,
                 test_mode=True,
                 min_depth=1e-3,
                 max_depth=200,
                ):

        self.split = split
        self.pipeline = Compose(pipeline)
        self.img_path = os.path.join(data_root)
        self.depth_path = os.path.join(data_root)
        self.test_mode = test_mode
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale

        # load annotations
        self.img_infos = self.load_annotations(self.img_path, self.depth_path)
        

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, depth_dir):
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory. Load all the images under the root.
        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []

        #imgs = os.listdir(img_dir)
        imgs = list()
        depths = list()
        ds = os.listdir(img_dir)
        if self.split == 'train':
            ds = ['dataset_1', 'dataset_2', 'dataset_3',
                'dataset_6', 'dataset_7',]
        else:
            ds = ['dataset_8', 'dataset_9',]

        for d in ds:
            ks = os.listdir(os.path.join(img_dir, d))
            for k in ks:
                case_path = os.path.join(img_dir, d, k, "data")
                img = os.listdir(os.path.join(case_path, "left_finalpass"))
                img = [os.path.join(case_path, "left_finalpass", i) for i in img]
                depth = os.listdir(os.path.join(case_path, "disparity"))
                depth = [os.path.join(case_path, "disparity", d) for d in depth]
                imgs += img
                depths += depth
        imgs.sort()
        depths.sort()

        if self.test_mode is not True:
            #depths = os.listdir(depth_dir)
            #depths.sort()

            for img, depth in zip(imgs, depths):
                img_info = dict()
                img_info['filename'] = img
                img_info['depth_path'] = depth
                img_infos.append(img_info)        
        else:
            # change including depth
            for img, depth in zip(imgs, depths):
                img_info = dict()
                img_info['filename'] = img
                img_info['depth_path'] = depth
                img_infos.append(img_info) 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images.', logger=get_root_logger())

        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['depth_fields'] = []
        results['img_prefix'] = self.img_path
        results['depth_prefix'] = self.depth_path
        results['depth_scale'] = self.depth_scale

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)

        depth_gt = tiff.imread(img_info['depth_path'])
        depth_gt = np.ascontiguousarray(depth_gt, dtype=np.float32) #/ self.depth_scale      # 2022.08.10 div updated
        results['depth_gt'] = depth_gt
        results['depth_ori_shape'] = depth_gt.shape
        results['depth_fields'].append('depth_gt')

        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)

        depth_gt = tiff.imread(img_info['depth_path'])
        depth_gt = np.ascontiguousarray(depth_gt, dtype=np.float32) #/ self.depth_scale      # 2022.08.10 div updated
        results['depth_gt'] = depth_gt
        results['depth_ori_shape'] = depth_gt.shape
        results['depth_fields'].append('depth_gt')
        return self.pipeline(results)
    
    def get_depth(self, idx):
        img_info = self.img_infos[idx]

        depth_img = tiff.imread(img_info['depth_path']) #/ self.depth_scale      # 2022.08.10 div updated
        depth_img = np.ascontiguousarray(depth_img, dtype=np.float32)
        depth_img = cv2.resize(depth_img, (600, 480))
        return depth_img

    def get_ann_info(self, idx):
        """Get annotation by index.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']
    
    # waiting to be done
    def format_results(self, results, imgfile_prefix=None, indices=None, **kwargs):
        """Place holder to format result to dataset specific output."""
        #results[0] = (results[0] * self.depth_scale) # Do not convert to np.uint16 for ensembling. # .astype(np.uint16)
        return results

    # i dont apply crop
    # get original GT! without scaling
    def eval_mask(self, depth_gt):
        depth_gt = np.squeeze(depth_gt)
        valid_mask = np.logical_and(depth_gt > self.min_depth, depth_gt < self.max_depth)
        #print(valid_mask, valid_mask.shape)
        #valid_mask = np.logical_and(valid_mask, eval_mask)
        valid_mask = np.expand_dims(valid_mask, axis=0)
        return valid_mask

    # design your own evaluation pipeline
    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.
        Args:
            preds (list[torch.Tensor] | torch.Tensor): the depth estimation.
            indices (list[int] | int): the prediction related ground truth
                indices.
        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []
        pre_eval_preds = []

        for i, (pred, index) in enumerate(zip(preds, indices)):
            depth_map_gt = self.get_depth(index)[np.newaxis, :, :]
            # remove kb_crop
            valid_mask = self.eval_mask(depth_map_gt)
            
            #defined in depth.core
            #print(depth_map_gt[0,100:150, 100:150], pred[0,100:150, 100:150], valid_mask[0,100:150, 100:150])
            eval = metrics(depth_map_gt[valid_mask], 
                           pred[valid_mask], 
                           min_depth=self.min_depth,
                           max_depth=self.max_depth)
            #print(eval)
            pre_eval_results.append(eval)

            # save prediction results
            pre_eval_preds.append(pred)

        return pre_eval_results, pre_eval_preds

    def evaluate(self, results, metric='eigen', logger=None, **kwargs):
        """Evaluate the dataset.
        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict depth map for computing evaluation
                 metric.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str, float]: Default metrics.
        """
        metric = ["a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel"]
        
        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            gt_depth_maps = self.get_gt_depth_maps()
            ret_metrics = eval_metrics(
                gt_depth_maps,
                results)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results)
        
        ret_metric_names = []
        ret_metric_values = []
        for ret_metric, ret_metric_value in ret_metrics.items():
            ret_metric_names.append(ret_metric)
            ret_metric_values.append(ret_metric_value)

        num_table = len(ret_metrics) // 9
        for i in range(num_table):
            names = ret_metric_names[i*9: i*9 + 9]
            values = ret_metric_values[i*9: i*9 + 9]

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value), 4)
                for ret_metric, ret_metric_value in zip(names, values)
            })

            # for logger
            summary_table_data = PrettyTable()
            for key, val in ret_metrics_summary.items():
                summary_table_data.add_column(key, [val])

            print_log('Summary:', logger)
            print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics.items():
            eval_results[key] = value

        return eval_results
