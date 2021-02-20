## Task

The task classifies the images based on its rotation. There are four possible rotations: upright, rotated_right, upside_down and rotated_left. The dataset used to train the model is available [here](https://github.com/lab-a1/correct-image-rotation/releases/tag/dataset).

[notebook.ipynb](notebook.ipynb) shows how the dataset is structured, the augmentation techniques applied and predictions made using the trained model. The trained model is available [here](https://github.com/lab-a1/correct-image-rotation/releases/tag/v0.0.1).

## Requirements

Make sure you have the following required libraries installed.

- [Albumentations](https://github.com/albumentations-team/albumentations)
- [Matplotlib](https://github.com/matplotlib/matplotlib)
- [OpenCV](https://pypi.org/project/opencv-python)
- [Pandas](https://github.com/pandas-dev/pandas)
- [PyTorch](https://github.com/pytorch/pytorch)
- [tqdm](https://github.com/tqdm/tqdm)

## How to use

[`src/main.py`](src/main.py) is the starting point of the application. There you can see where the dataset loader and the network topology are defined.

After changing the hyperparameters according to your needs, you can train your own model by running the script below:

```bash
git clone git@github.com:lab-a1/correct-image-rotation.git
cd correct-image-rotation
wget https://github.com/lab-a1/correct-image-rotation/releases/download/dataset/dataset.tar.gz -O dataset.tar.gz
tar -xzvf dataset.tar.gz -C dataset
cd src
python main.py
```
