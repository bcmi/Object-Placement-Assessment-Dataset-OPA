import csv
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from config import opt

torch.random.manual_seed(1)


class ImageDataset(Dataset):
    def __init__(self, istrain=True):
        self.istrain = istrain
        with open(opt.train_data_path if istrain else opt.test_data_path, "r") as f:
            reader = csv.reader(f)
            reader = list(reader)
            reader = reader[1:]

        self.labels = []
        self.images_path = []
        self.mask_path = []
        self.target_box = []  # foreground box
        self.dic_name = []
        for row in reader:
            label = int(row[-3])
            image_path = row[-2]
            mask_path = row[-1]
            target_box = eval(row[2])
            self.labels.append(label)
            self.images_path.append(os.path.join(opt.img_path, image_path))
            self.mask_path.append(os.path.join(opt.mask_path, mask_path))
            self.target_box.append(target_box)
            self.dic_name.append(image_path)

        self.img_transform = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor()
        ])

        self.transforms_flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1)
        ])

    def __getitem__(self, index):
        ## img & mask
        img = Image.open(self.images_path[index]).convert('RGB')
        w = img.width
        h = img.height
        img = self.img_transform(img)

        mask = Image.open(self.mask_path[index]).convert('L')  # gray
        mask = self.img_transform(mask)

        # flip
        is_flip = False
        if self.istrain and np.random.uniform() < 0.5:
            img = self.transforms_flip(img)
            mask = self.transforms_flip(mask)
            is_flip = True
        img_mask = torch.cat([img, mask], dim=0)

        label = self.labels[index]
        target_box = self.target_box[index]
        x1, y1, bw, bh = target_box
        x2, y2 = x1 + bw, y1 + bh
        if is_flip:
            x1 = w - x1
            x2 = w - x2
            x1, x2 = x2, x1
        target_box = torch.tensor([x1, y1, x2, y2])

        return img_mask, label, target_box

    def __len__(self):
        return len(self.labels)


def get_train_dataloader():
    trainset = ImageDataset(istrain=True)
    print('Training images', len(trainset))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                               shuffle=True, num_workers=opt.num_workers,
                                               drop_last=True, pin_memory=True)
    return train_loader


def get_test_dataloader():
    testset = ImageDataset(istrain=False)
    print('Testing images', len(testset))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size * 2,
                                              shuffle=False, num_workers=opt.num_workers,
                                              drop_last=False, pin_memory=True)
    return test_loader


if __name__ == "__main__":
    train_loader = get_train_dataloader()
    for batch_index, (img, label, target_box) in enumerate(train_loader):
        print(img.shape, label.shape, target_box.shape)
        if batch_index > 10:
            break

    test_loader = get_test_dataloader()
    for batch_index, (img, label, target_box) in enumerate(test_loader):
        print(img.shape, label.shape, target_box.shape)
        if batch_index > 10:
            break
