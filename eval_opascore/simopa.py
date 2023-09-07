import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
import torchvision
import numpy as np
import os,sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(cur_dir))
from object_place_net import ObjectPlaceNet
from PIL import Image
import torchvision.transforms as transforms
import cv2
import warnings
warnings.filterwarnings('ignore')
import argparse 

def get_args():
    parser = argparse.ArgumentParser(description='Object Placement Assessment Score Predictor')
    parser.add_argument('--weight', default=os.path.join(cur_dir, 'checkpoints/simopa.pth'), type=str, help='path to the weight file of simopa')
    parser.add_argument('--image', default=os.path.join(cur_dir, 'examples/composite_1.jpg'), type=str, help='composite image')
    parser.add_argument('--mask', default=os.path.join(cur_dir, 'examples/mask_1.jpg'), type=str, help='foreground mask')
    parser.add_argument('--gpu', default=0, type=int, help='device ID')
    args = parser.parse_args()
    return args

class ObjectPlacementAssessmentModel:
    def __init__(self, device, opt):
        self.device = device
        self.model = self.build_pretrained_model(opt.weight)
        self.image_size = 256
        self.transformer = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    def build_pretrained_model(self, weight):
        model  = ObjectPlaceNet(False)
        assert os.path.exists(weight), weight
        print('Build ObjectPlaceAssessment Model')
        model = model.eval().to(self.device)
        model.load_state_dict(torch.load(weight, map_location='cpu'))
        return model

    def data_preprocess(self, image, mask):
        img = Image.open(image).convert('RGB')
        img = self.transformer(img)

        mask = Image.open(mask).convert('L')  # gray
        mask = self.transformer(mask)

        cat_img = torch.cat([img, mask], dim=0)
        cat_img = cat_img.unsqueeze(0).to(self.device)
        return cat_img

    def __call__(self, image, mask):
        cat_img = self.data_preprocess(image, mask)
        logits  = self.model(cat_img)
        score   = torch.softmax(logits, dim=-1)[0, 1].cpu().item()
        score  = round(score, 2)
        return score

if __name__ == '__main__':
    opt = get_args()
    device = torch.device('cuda:{}'.format(1))
    model = ObjectPlacementAssessmentModel(device, opt)
    score = model(opt.image, opt.mask)
    print(os.path.basename(opt.image), score)