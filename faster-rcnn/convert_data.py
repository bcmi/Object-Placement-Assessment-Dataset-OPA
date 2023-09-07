"""Convert image features from bottom up attention to numpy array"""

# Example
# python convert_data.py --expid ${expid}$ --epoch ${epoch}$

import os
import base64
import csv
import sys
import argparse
import numpy as np
from tqdm import tqdm
csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_name', 'image_w', 'image_h', 'num_boxes', 'boxes', 'pred_scores', 'features', 'fg_feature']

dataset_dir = '../cache/faster_rcnn_features'
os.makedirs(dataset_dir, exist_ok=True)
csv_file = '../cache/roiinfos.csv'
assert (os.path.exists(csv_file)), csv_file
csv_data = csv.DictReader(open(csv_file, 'r'))

with open(csv_file, "r+") as tsv_in_file:
    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
    for item in tqdm(reader):
        img_path = item['image_name']
        # item['image_id'] = int(item['image_id'])
        item['image_h'] = int(item['image_h'])
        item['image_w'] = int(item['image_w'])
        item['num_boxes'] = int(item['num_boxes'])
        for field in ['boxes', 'pred_scores', 'features']:
            data = item[field]
            buf = base64.b64decode(data[1:])
            temp = np.frombuffer(buf, dtype=np.float32)
            item[field] = temp.reshape((item['num_boxes'], -1))
        for field in ['fg_feature']:
            data = item[field]
            buf = base64.b64decode(data[1:])
            temp = np.frombuffer(buf, dtype=np.float32)
            item[field] = temp.reshape((1, -1))
        idx = np.argsort(-item['boxes'][:, 5])
        item['boxes'] = item['boxes'][idx, :]
        item['pred_scores'] = item['pred_scores'][idx, :]
        item['features'] = item['features'][idx, :]

        # if item['image_id'] in bboxes:
        #     bboxes[item['image_id']] = item['boxes']
        #     scores[item['image_id']] = item['pred_scores']
        #     features[item['image_id']] = item['features']
        #     fg_feature[item['image_id']] = item['fg_feature']
        res_file = os.path.join(dataset_dir, os.path.basename(img_path).split('.')[0] + '.npz')
        np.savez(res_file, item)

# output_dict = {
#     "bboxes": bboxes,
#     "scores": scores,
#     "feats": features,
#     "fgfeats": fg_feature
# }
# for k, v in output_dict.items():
#     output_file = os.path.join(dataset_dir, "{}.npy".format(k))
#     data_out = np.stack([v[sid] for sid in meta], axis=0)
#     np.save(output_file, data_out)
