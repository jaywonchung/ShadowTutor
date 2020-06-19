"""
Running this file will create the COCO semantic segmentation dataset from
the original COCO instance segmentation dataset.
Importing this file will register the COCO semantic segmentation dataset.
"""
import os
import random

from tqdm import tqdm

import numpy as np

from PIL import Image
from pycocotool import mask as maskUtils

import detectron2
from detectron2.data.datasets import load_sem_seg
from detectron2.evaluation import DatasetEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode


# Register train dataset
get_dicts_train = lambda : load_sem_seg(
    image_root='datasets/coco/train2017',
    image_ext='jpg',
    gt_root='datasets/coco/gt_train2017',
    gt_ext='png'
)
DatasetCatalog.register('coco_2017_sem_train', get_dicts_train)
meta = MetadataCatalog.get('coco_2017_sem_train')
meta.name = 'coco_2017_sem_train'
meta.image_root = 'datasets/coco/train2017'
meta.evaluator_type = 'sem_seg'
meta.stuff_classes = MetadataCatalog.get('coco_2017_train').thing_classes
meta.stuff_colors = MetadataCatalog.get('coco_2017_train').thing_colors

# Register val dataset
get_dicts_val = lambda : load_sem_seg(
    image_root='datasets/coco/val2017',
    image_ext='jpg',
    gt_root='datasets/coco/gt_val2017',
    gt_ext='png'
)
DatasetCatalog.register('coco_2017_sem_val', get_dicts_val)
meta = MetadataCatalog.get('coco_2017_sem_val')
meta.name = 'coco_2017_sem_val'
meta.image_root = 'datasets/coco/val2017'
meta.evaluator_type = 'sem_seg'
meta.stuff_classes = MetadataCatalog.get('coco_2017_val').thing_classes
meta.stuff_colors = MetadataCatalog.get('coco_2017_val').thing_colors


class SemSegVisualizer(DatasetEvaluator):
    def __init__(self, dataset_name, num_save, output_dir):
        self._dataset_name = dataset_name
        self._saved = 0
        self._num_reg_save = num_save
        self._num_rand_save = num_save
        self._output_dir = output_dir

        self.input_file_to_gt_file = {
            dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
            for dataset_record in DatasetCatalog.get(dataset_name)
        }

        for filename in os.listdir(output_dir):
            if filename.endswith('png'):
                os.remove(os.path.join(output_dir, filename))

    def reset(self):
        pass

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            if self._saved == self._num_reg_save + self._num_rand_save:
                return
            if self._saved >= self._num_reg_save and random.random() > 0.2:
                continue
            output = output["sem_seg"].argmax(dim=0).cpu()
            with open(input["file_name"], 'rb') as f:
                im = np.array(Image.open(f), dtype=np.int)
            
            # Draw prediction
            vis = Visualizer(im, MetadataCatalog.get(self._dataset_name))
            vis = vis.draw_sem_seg(output, alpha=0.5)
            vis.save(os.path.join(
                self._output_dir,
                os.path.basename(self.input_file_to_gt_file[input["file_name"]])
            ))

            # Draw ground truth
            vis = Visualizer(im, MetadataCatalog.get(self._dataset_name))
            vis = vis.draw_dataset_dict(
                dict(sem_seg_file_name=self.input_file_to_gt_file[input["file_name"]])
            )
            vis.save(os.path.join(
                self._output_dir,
                'gt_' + os.path.basename(self.input_file_to_gt_file[input["file_name"]])
            ))
            self._saved += 1
    
    def evaluate(self):
        pass            


if __name__ == "__main__":
    # Generate semantic segmentation labels from COCO
    for split in ['val', 'train']:
        print(f'Processing {split} split...')

        # Folder to save semantic segmentation label files
        gt_root = os.path.join('datasets', 'coco', f'gt_{split}2017')
        if not os.path.exists(gt_root):
            os.mkdir(gt_root)

        # Load dataset
        coco = DatasetCatalog.get(f'coco_2017_{split}')

        # Coalesce instances of the same category
        # sem_seg: [H, W] where each pixel denotes its category 0~79
        for img in tqdm(coco):
            # Fill with background label 80
            sem_seg = np.ones((img['height'], img['width'])) * 80

            # Sort annotations with category id
            annos = img['annotations']
            annos.sort(key=lambda anno: anno['category_id'])

            # Coalesce instances from the same category in the list rles
            rles = []
            cat_list = [anno['category_id'] for anno in annos]
            for i, anno in enumerate(annos):
                # Polygon -> compressed RLE -> merge
                rle_maybe_list = maskUtils.frPyObjects(
                    anno['segmentation'], img['height'], img['width']
                )
                if not isinstance(rle_maybe_list, list):
                    rle_maybe_list = [rle_maybe_list]
                rles.append(maskUtils.merge(rle_maybe_list))

                # At the end of the current category, apply to sem_seg
                if i+1 == len(annos) or cat_list[i] != cat_list[i+1]:
                    rle = maskUtils.merge(rles)
                    cat_mask = maskUtils.decode(rle)
                    sem_seg[cat_mask.astype(np.bool)] = cat_list[i]
                    rles = []

            # Label file name is exactly the same as the input image
            sem_seg_file_name = os.path.join(gt_root, os.path.basename(img["file_name"]).replace('jpg','png'))
            Image.fromarray(sem_seg.astype('uint8')).save(sem_seg_file_name)

