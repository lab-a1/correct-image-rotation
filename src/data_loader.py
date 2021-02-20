import torch
import os
from torch.utils.data import Dataset
import cv2


class RotatedImagesDataset(Dataset):
    def __init__(self, images_path, ground_truth, transform=None):
        self.ground_truth = ground_truth
        self.images_path = images_path
        self.transform = transform
        self.label_to_index = {
            "upright": 0,
            "rotated_right": 1,
            "upside_down": 2,
            "rotated_left": 3,
        }
        self.index_to_label = {
            self.label_to_index[x]: x for x in self.label_to_index
        }

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(index.start, index.stop)]
        elif torch.is_tensor(index):
            index = index.tolist()

        item = self.ground_truth.loc[index]
        image_path = os.path.join(self.images_path, item[0])
        image = cv2.imread(image_path)
        # OpenCV reads by default as BGR.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        return image, self.label_to_index[item[1]]
