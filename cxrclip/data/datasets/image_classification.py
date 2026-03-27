import ast
from typing import Dict, List

import pandas as pd
import torch
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from cxrclip.data.data_utils import load_transform, transform_image
from cxrclip.evaluator_utils import disease_name_mapping
from cxrclip.prompt import constants
from cxrclip.util.utils import curate_dqn_input_labels
import logging
log = logging.getLogger(__name__)


class ImageClassificationDataset(Dataset):
    def __init__(
        self,
        name: str,
        data_path: str,
        split: str,
        normalize: str,
        data_frac: float = 1.0,
        sample_shots_not_percentage: bool = False,
        transform_config: Dict = None,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.split = split
        self.data_frac = data_frac
        self.normalize = normalize

        self.image_transforms = load_transform(split=split, transform_config=transform_config)
        self.df = pd.read_csv(data_path)

        # for few shot
        self.tokenizer = kwargs.get('tokenizer', None)
        self.dqn_label_max_length = kwargs.get('dqn_label_max_length', 48)
        self.prompt_template = "There is {}." if kwargs.get('add_alignment_prompt_prefix', False) else "{}."

        if 'class' in self.df:
            self.df['class'] = self.df['class'].apply(lambda x: ast.literal_eval(x) if x.startswith('[') else [x])
        if 'label' in self.df:
            self.df['label'] = self.df['label'].apply(lambda x: ast.literal_eval(x) if type(x) is str and x.startswith('[') else [float(x)])

    def _create_multi_hot_label_vector(self, dataframe):
        """
        mainly to override the label vector
        """
        unique_classes = self.disease_of_interest_list

        # 2. Map each class name to a specific index (0, 1, 2...)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        num_classes = len(unique_classes)

        # 3. Define a helper to convert a list of names into a list of 0s and 1s
        def create_multi_hot(class_list):
            label = [0] * num_classes
            for c in class_list:
                # Set the index to 1 if the class exists in the row
                label[class_to_idx[c]] = 1
            return label

        # 4. Replace dataframe['label'] with the new multi-hot lists
        dataframe['label'] = dataframe['class'].apply(create_multi_hot)
        return dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df["image"][index]
        if image_path.startswith("["):
            image_path = ast.literal_eval(image_path)[0]
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = np.stack([original_image] * 3, axis=-1)
        image = transform_image(self.image_transforms, image, normalize=self.normalize)

        label = None
        if "label" in self.df:
            label = self.df["label"][index]
            if type(label) is str:
                label = ast.literal_eval(label)
            label = torch.Tensor(label)

        label_name = None
        if "class" in self.df:
            label_name = self.df["class"][index]
            if isinstance(label_name, str):
                label_name = [label_name]

        mask = None

        # for pointing game.
        boxes = []
        if "boxes" in self.df:
            boxes = self.df["boxes"][index]
            boxes = ast.literal_eval(boxes)

        return {
            "image": image, 
            "label": label, 
            "label_name": label_name, 
            "boxes": boxes,
            "mask": mask,
            "image_path": self.df['image'][index]
        }

    def collate_fn(self, instances: List):

        labels, masks, label_names = None, [], []
        if len(instances) > 0 and instances[0]["label"] is not None:
            labels = torch.stack([ins["label"] for ins in instances], dim=0)
        if len(instances) > 0 and instances[0]["mask"] is not None:
            masks = [ins["mask"] for ins in instances]

        if len(instances) > 0 and instances[0]["label_name"] is not None:
            label_names = list([ins["label_name"] for ins in instances])

        images = torch.stack([ins["image"] for ins in instances], dim=0)
        boxes = list([ins["boxes"] for ins in instances])
        image_paths = list([ins["image_path"] for ins in instances])

        label_tokens = None
        return {
            "images": images, 
            "labels": labels, 
            "label_names": label_names,
            "label_tokens": label_tokens,
            "boxes": boxes,
            "masks": masks,
            "image_paths": image_paths,
            "multihot_label": labels.numpy() if labels is not None else labels
        }
