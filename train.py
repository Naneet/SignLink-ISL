import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import subprocess
from PIL import Image
import shutil

from data_loader.VideoDatasetLoader import VideoDatasetLoader
from models.Conv3D import SignLanguageClassifier
from trainers.trainer import Trainer
from utils.save_load import save_checkpoint,load_checkpoint
from utils.data_words import data_and_words
from utils.show_sequence import show_sequence

SEED = 42
torch.manual_seed(SEED)
resolution = '1366:768'
NUM_FRAMES = 9
video_fps = 12
batch_size = 16
train_data_path = '/data/train'
test_data_path = '/data/test'
temp_data_path = '/temp'
epochs = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"{device=}")

# Preprocessing transform
preprocess_frame = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop((700,1244)),
    transforms.CenterCrop((640,640)),

    #transforms.RandomApply(torch.nn.ModuleList([
    #transforms.ColorJitter(),
    #]), p=0.3)
])

preprocess_video = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
])

train_data, idx_to_word, word_to_idx = data_and_words(train_data_path)
test_data, idx_to_word, word_to_idx = data_and_words(test_data_path)

train_dataset = VideoDatasetLoader(data=train_data, temp_data_folder=temp_data_path, transform_frame=preprocess_frame, transform_video=preprocess_video,
                             NUM_FRAMES=NUM_FRAMES, video_fps=video_fps, resolution=resolution)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


test_dataset = VideoDatasetLoader(data=test_data, temp_data_folder=temp_data_path, transform_frame=preprocess_frame, transform_video=preprocess_video,
                            NUM_FRAMES=NUM_FRAMES, video_fps=video_fps, resolution=resolution)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

var = input("Test Dataloader?:(y or n):")
if var == "y" or var == "Y":
    vid, label = next(iter(train_dataloader))
    show_sequence(vid[2], NUM_FRAMES)
    print("label:",idx_to_word[label[2]])
    print(f"{vid.shape=}")

model = SignLanguageClassifier(len(word_to_idx)).to(device)
# model = torch.compile(model) # Remove!!!!
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

trainer = Trainer(device=device,optimizer=optimizer,loss_fn=loss_fn,train_dataloader=train_dataloader,test_dataloader=test_dataloader,save=True,model=model)

for epoch in range(epochs):
    trainer.train_step(epoch=epoch)
    trainer.test_step(epoch=epoch)