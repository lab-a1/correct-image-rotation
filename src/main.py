import os
import torch
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data_loader import RotatedImagesDataset
from network import CNNNetwork
from train import train
from validation import validation
from test import test
from lib.utils import split_dataset


# Enables the inbuilt cuDNN auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True

ground_truth = pd.read_csv("../dataset/ground-truth.csv")
ground_truth = ground_truth.sample(frac=1)
ground_truth.reset_index(inplace=True, drop=True)

train_dataset, validation_dataset, test_dataset = split_dataset(
    ground_truth, 0.1, 0.1
)

dataset_transform = A.Compose(
    [
        A.ShiftScaleRotate(p=0.6, shift_limit=0.1, rotate_limit=15),
        A.RandomBrightnessContrast(p=0.7),
        A.ToGray(p=0.3),
        A.ToSepia(p=0.2),
        ToTensorV2(),
    ]
)
test_dataset_transform = A.Compose([ToTensorV2()])
train_dataset = RotatedImagesDataset(
    "../dataset/images", train_dataset, dataset_transform
)
validation_dataset = RotatedImagesDataset(
    "../dataset/images", validation_dataset, dataset_transform
)
test_dataset = RotatedImagesDataset(
    "../dataset/images", test_dataset, test_dataset_transform
)

params = {
    "device": "cuda",
    "learning_rate": 1e-4,
    "batch_size": 64,
    "epochs": 10,
    "num_workers": 4,
}

train_dataset_loader = DataLoader(
    train_dataset,
    batch_size=params["batch_size"],
    shuffle=True,
    num_workers=params["num_workers"],
    pin_memory=True,
)
validation_dataset_loader = DataLoader(
    validation_dataset,
    batch_size=params["batch_size"],
    shuffle=True,
    num_workers=params["num_workers"],
    pin_memory=True,
)
test_dataset_loader = DataLoader(
    test_dataset,
    batch_size=params["batch_size"],
    shuffle=False,
    num_workers=params["num_workers"],
    pin_memory=True,
)

model = CNNNetwork()
model = model.to(params["device"])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

for epoch in range(1, params["epochs"] + 1):
    train(model, params, train_dataset_loader, criterion, optimizer, epoch)
    validation(model, params, validation_dataset_loader, criterion, epoch)
test(model, params, test_dataset_loader, criterion)

model_path = "../saved_model"
if not os.path.exists(model_path):
    os.mkdir(model_path)
torch.save(model.state_dict(), os.path.join(model_path, "trained_model.pt"))

print("Model saved.")
