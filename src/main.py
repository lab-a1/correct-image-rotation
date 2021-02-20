import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data_loader import RotatedImagesDataset
from network import CNNNetwork


ground_truth = pd.read_csv("../dataset/ground-truth.csv")
ground_truth = ground_truth.sample(frac=1)
ground_truth.reset_index(inplace=True, drop=True)

validation_dataset_size = int(len(ground_truth) * 0.2)
ground_truth_train = ground_truth[:validation_dataset_size]
ground_truth_validation = ground_truth[validation_dataset_size:-10]
ground_truth_test = ground_truth[-10:]

dataset_transform = A.Compose(
    [
        A.ShiftScaleRotate(p=0.6, shift_limit=0.1, rotate_limit=15),
        A.RandomBrightnessContrast(p=0.7),
        A.ToGray(p=0.3),
        A.ToSepia(p=0.2),
        ToTensorV2(),
    ]
)
train_dataset_loader = RotatedImagesDataset(
    "../dataset/images", ground_truth_train, dataset_transform
)
validation_dataset_loader = RotatedImagesDataset(
    "../dataset/images", ground_truth_validation, dataset_transform
)
test_dataset_loader = RotatedImagesDataset(
    "../dataset/images", ground_truth_test, dataset_transform
)

params = {
    "device": "cuda",
    "learning_rate": 1e-4,
    "batch_size": 64,
    "epochs": 20,
}

model = CNNNetwork()
model = model.to(params["device"])

