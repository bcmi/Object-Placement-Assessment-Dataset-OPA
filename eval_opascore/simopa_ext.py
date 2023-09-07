import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
import torchvision
import numpy as np
import os,sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(cur_dir))
from simopa_ext_net import ObjectPlaceNet
from simopa_ext_config import opt as net_opt
sys.path.insert(0, os.path.join(cur_dir, '../faster-rcnn'))
from generate_tsv import load_model, parse_args, get_detections_from_im
from PIL import Image
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Object Placement Assessment Score Predictor')
    parser.add_argument('--weight', default=os.path.join(cur_dir, 'checkpoints/simopa_ext.pth'), type=str, help='path to the weight file of simopa-ext')
    parser.add_argument('--image', default=os.path.join(cur_dir, 'examples/composite_0.jpg'), type=str, help='composite image')
    parser.add_argument('--mask', default=os.path.join(cur_dir, 'examples/mask_0.jpg'), type=str, help='foreground mask')
    parser.add_argument('--gpu', default=0, type=int, help='device ID')
    args = parser.parse_args()
    return args

def mask2bbox(mask):
    bin = np.where(mask >= 127)
    x1 = int(np.min(bin[1]))
    x2 = int(np.max(bin[1]))
    y1 = int(np.min(bin[0]))
    y2 = int(np.max(bin[0]))
    return [x1, y1, x2, y2] 

def generate_binary_mask(target_box, refer_box, w, h, bm_size, dtype):
    # produce binary mask for target/reference boxes
    scale_x1 = (target_box[0] / w * bm_size).int()
    scale_y1 = (target_box[1] / h * bm_size).int()
    scale_x2 = (target_box[2] / w * bm_size).int()
    scale_y2 = (target_box[3] / h * bm_size).int()

    target_mask = torch.zeros(1, bm_size, bm_size, dtype=dtype)
    target_mask[0, scale_y1: scale_y2, scale_x1: scale_x2] = 1
    
    refer_num  = refer_box.shape[0]
    refer_mask = torch.zeros(refer_num, bm_size, bm_size, dtype=dtype)
    scale_x1 = (refer_box[:, 0] / w * bm_size).int()
    scale_y1 = (refer_box[:, 1] / h * bm_size).int()
    scale_x2 = (refer_box[:, 2] / w * bm_size).int()
    scale_y2 = (refer_box[:, 3] / h * bm_size).int()

    for i in range(refer_num):
        refer_mask[i, scale_y1[i]: scale_y2[i], scale_x1[i]: scale_x2[i]] = 1
    return target_mask.unsqueeze(0), refer_mask.unsqueeze(0)

class ObjectPlacementAssessmentModel:
    def __init__(self, device, opt):
        self.device = device
        self.build_faster_rcnn()
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
    
    def build_faster_rcnn(self):
        cwd = os.getcwd()
        os.chdir(os.path.join(cur_dir, '../faster-rcnn'))
        self.frn_args = parse_args()
        self.frn_classes, self.frn_remained_ids, self.fasterRCNN = load_model(self.frn_args)
        os.chdir(cwd)
        
    def data_preprocess(self, image, mask):
        # concat composite image and mask
        img = Image.open(image).convert('RGB')
        w,h = img.width, img.height
        img_t   = self.transformer(img)
        mask    = Image.open(mask).convert('L')  # gray
        mask_t  = self.transformer(mask)
        img_cat = torch.cat([img_t, mask_t], dim=0)
        img_cat = img_cat.unsqueeze(0).to(self.device)
        # extract target feature of foreground object and the features of reference objects
        target_box = mask2bbox(np.asarray(mask))
        feat_info   = get_detections_from_im(self.fasterRCNN, self.frn_classes, self.frn_remained_ids, image, 0, target_box, self.frn_args, base64=False)
        target_box  = torch.tensor(target_box).float().to(self.device)
        refer_box   = feat_info['boxes']
        refer_score = refer_box[:, -1]
        refer_keep  = np.argsort(refer_score)[::-1][:net_opt.refer_num]
        refer_box   = torch.tensor(refer_box[refer_keep]).float().to(self.device)
        refer_feat  = feat_info['features'][refer_keep.tolist()]
        target_feat = feat_info['fg_feature'].unsqueeze(0).to(self.device)
        # generate rectangular mask for foreground object and reference objects
        target_mask, refer_mask = generate_binary_mask(target_box, refer_box, w, h, net_opt.binary_mask_size, img_t.dtype)
        return img_cat, target_box.unsqueeze(0), refer_box.unsqueeze(0), target_feat, refer_feat, target_mask, refer_mask, w, h

    def __call__(self, image, mask):
        img_cat, target_box, refer_box, target_feat, refer_feat, target_mask, refer_mask, w, h = self.data_preprocess(image, mask)
        logits = self.model(img_cat, target_box, refer_box, target_feat, refer_feat, target_mask, refer_mask, w, h)
        score  = torch.softmax(logits, dim=-1)[0, 1].cpu().item()
        score  = round(score, 2)
        return score

if __name__ == '__main__':
    opt = get_args()
    device = torch.device('cuda:{}'.format(1))
    model = ObjectPlacementAssessmentModel(device, opt)
    score = model(opt.image, opt.mask)
    print(os.path.basename(opt.image), score)