from data_loader import RotatedImagesDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

dataset_transform = A.Compose(
    [
        A.ShiftScaleRotate(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2(),
    ]
)
rotated_dataset_loader = RotatedImagesDataset(
    "../dataset/images", "../dataset/ground-truth.csv", dataset_transform
)

print(rotated_dataset_loader[0][0].shape)
