import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb

class CramedDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        # self.visual_feature_path = args.visual_path
        # self.audio_feature_path = args.audio_path
        #
        # self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
        # self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')

        if mode == 'train':
            with open(self.args.train_path, 'rb') as f:
                data = pickle.load(f)
            self.label = data['label']
            self.audio = data['spectrogram']
            self.image = data['image']
        else:
            with open(self.args.test_path, 'rb') as f:
                data = pickle.load(f)
            self.label=data['label']
            self.audio=data['spectrogram']
            self.image=data['image']



    # def __len__(self):
    #     return len(self.ids)
    #
    # def __getitem__(self, idx):
    #
    #     if self.mode == 'train':
    #         transform = transforms.Compose([
    #             transforms.ToPILImage(),
    #             transforms.RandomResizedCrop(224),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ])
    #     else:
    #         transform = transforms.Compose([
    #             transforms.ToPILImage(),
    #             # transforms.Resize(size=(224, 224)),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ])
    #
    #     # Visual
    #     images = torch.zeros((self.args.fps, 3, 224, 224))
    #     images[0] = transform(self.image[self.ids[idx]].astype('uint8'))
    #     images=torch.permute(images, (1,0,2,3))
    #
    #
    #     return self.audio[self.ids[idx]], images, self.label[self.ids[idx]]

class DatasetSplit(Dataset):
    def __init__(self, dataset, ids):
        self.image = dataset.image
        self.audio = dataset.audio
        self.label = dataset.label
        self.mode=dataset.mode
        self.args=dataset.args
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        images = torch.zeros((self.args.fps, 3, 224, 224))
        images[0] = transform(self.image[self.ids[idx]].astype('uint8'))
        images=torch.permute(images, (1,0,2,3))


        return self.audio[self.ids[idx]], images, self.label[self.ids[idx]], idx