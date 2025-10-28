import torch
from torch.utils.data import Dataset
from cv2 import imread, cvtColor, COLOR_BGR2HSV, resize
from einops import rearrange
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class DroneDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(os.path.join(path, 'images'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, 'images', self.files[idx])
        label_path = os.path.join(self.path, 'labels', os.path.splitext(self.files[idx])[0] + '.txt')

        bgr_img = resize(imread(img_path), (640, 640))
        cv_img = cvtColor(bgr_img, COLOR_BGR2HSV) / 255.  # 转为hsv并且标准化
        img: torch.Tensor = torch.from_numpy(cv_img).float()
        img.to(device)


        label: torch.Tensor = torch.from_numpy(np.fromfile(label_path, sep=' ', dtype=np.float32))
        label.to(device)
        if len(label) < 5:
            label = torch.zeros(5).float()
            label.to(device)

        return img, label[1:5]


if __name__ == '__main__':
    dataset = DroneDataset('./test')
    print(dataset[0])
