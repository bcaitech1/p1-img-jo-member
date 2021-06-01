from data import MaskBaseDataset, cfg, get_transforms
import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from time import time
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import copy
from adamp import AdamP
print ("PyTorch version:[%s]."%(torch.__version__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print ("device:[%s]."%(device))

mean, std = (0.5, 0.5, 0.5), (0.2, 0.2, 0.2)
transform = get_transforms(mean=mean, std=std)
dataset = MaskBaseDataset(
    img_dir=cfg.img_dir
)

n_val = int(len(dataset) * 0.2)
n_train = len(dataset) - n_val
train_dataset, val_dataset = data.random_split(dataset, [n_train, n_val])

train_dataset.dataset.set_transform(transform['train'])
val_dataset.dataset.set_transform(transform['val'])

train_loader = data.DataLoader(
    train_dataset,
    batch_size=12,
    num_workers=4,
    shuffle=True
)

val_loader = data.DataLoader(
    val_dataset,
    batch_size=12,
    num_workers=4,
    shuffle=False
)

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b4')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 18) 
model_ft = model_ft.to(device)
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = AdamP(model_ft.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-2)
EPOCHS = 3
for epoch in range(EPOCHS):
    model_ft.train()
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.long().to(device)
        output = model_ft(inputs)
        optimizer.zero_grad()
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()


test_dir = 'input/data/eval'

class TestDataset(data.Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

# meta 데이터와 이미지 경로를 불러옵니다.
submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
image_dir = os.path.join(test_dir, 'images')

# Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
transform = transforms.Compose([
    Resize((512, 384), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])
dataset = TestDataset(image_paths, transform)

loader = data.DataLoader(
    dataset,
    shuffle=False
)

# 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)

model_ft.eval()

# 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
all_predictions = []
for images in loader:
    with torch.no_grad():
        images = images.to(device)
        pred = model_ft(images)
        pred = pred.argmax(dim=-1)
        all_predictions.extend(pred.cpu().numpy())
submission['ans'] = all_predictions

# 제출할 파일을 저장합니다.
submission.to_csv(os.path.join(test_dir, 'submission.csv'), index=False)
print('test inference is done!')
