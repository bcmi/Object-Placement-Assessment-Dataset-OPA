# Object-Placement-Assessment-Dataset-OPA

**Object-Placement-Assessment** (OPA) is to verify whether a composite image is plausible in terms of the object placement. The foreground object should be placed at a reasonable location on the background considering location, size, occlusion, semantics, and etc.


## Dataset

Our dataset **OPA** is a synthesized dataset for Object Placement Assessment based on [COCO](http://cocodataset.org) dataset.  We select unoccluded objects from multiple categories as our candidate foreground objects. The foreground objects are pasted on their compatible background images with random sizes and locations to form composite images, which are sent to human annotators for rationality labeling. Finally, we split the collected dataset into training set and test set, in which the background images and foreground objects have no overlap between training set and test set. We show some example positive and negative images in our dataset in the figure below.

<img src='Examples/dataset_sample.png' align="center" width=1024>

Illustration of OPA dataset samples: Some positive and negative samples in our OPA dataset and the inserted foreground objects are marked with red outlines. Top row: positive samples; Bottom rows: negative samples, including objects with inappropriate size (e.g., f, g, h), without supporting force (e.g., i, j, k), appearing in the semantically unreasonable place (e.g., l, m, n), with unreasonable occlusion (e.g., o, p, q), and with inconsistent perspectives (e.g., r, s, t).

Our OPA dataset contains 62,074 training images and 11,396 test images, in which the foregrounds/backgrounds in training set and test set have no overlap. The training (resp., test) set contains 21,376 (resp.,3,588) positive samples and 40,698 (resp., 7,808) negative samples. Besides, the training (resp., test) set contains 2,701 (resp., 1,436) unrepeated foreground objects and1,236 (resp., 153) unrepeated background images. The OPA dataset is provided in [**Baidu Cloud**](https://pan.baidu.com/s/1IzVLcXWLFgFR4GAbxZUPkw) (access code: a982) or [**Google Drive**](https://drive.google.com/file/d/133Wic_nSqfrIajDnnxwvGzjVti-7Y6PF/view?usp=sharing).

## Dataset Extension

Based on the foregrounds and backgrounds from OPA dataset, we additionally synthesize 80263 composite images and annotate their binary rationality labels. We refer to the extended set as OPA-ext, which includes 28455 positive composite images and 51808 negative composite images. The labels in OPA-ext are relatively more noisy than OPA dataset. Note that the foregrounds/backgrounds in OPA-ext have overlap with those in OPA test set, so using OPA-ext to augment OPA training set could lead to unreasonably high performance on OPA test set due to foregrounds/backgrounds leakage. With the same data format as OPA, the OPA-ext dataset is provided in [**Baidu Cloud**](https://pan.baidu.com/s/1GTGwISKJIIp1HZ5AYJSH4w?pwd=fogy) (access code: fogy). 

## Prerequisites

- Python

- Pytorch

- PIL



## Getting Started

### Installation

- Clone this repo:

  ```bash
  git clone https://github.com/bcmi/Object-Placement-Assessment-Dataset-OPA.git
  cd Object-Placement-Assessment-Dataset-OPA
  ```

- Download the OPA dataset. We show the file structure below:

  ```
  ├── background:
       ├── category:
                ├── imgID.jpg
                ├── ……
       ├── ……
  ├── foreground:
       ├── category:
                ├── imgID.jpg
                ├── mask_imgID.jpg
                ├── ……
       ├── ……
  ├── composite:
       ├── train_set:
                ├── fgimgID_bgimgID_x_y_w_h_scale_label.jpg
                ├── mask_fgimgID_bgimgID_x_y_w_h_scale_label.jpg
                ├── ……
       └── test_set:
  ├── train_set.csv
  └── test_set.csv
  ```

  All backgrounds and foregrounds have their own IDs for identification. Each category of foregrounds and their compatible backgrounds are placed in one folder. The corresponding masks are placed in the same folder with a mask prefix.

  Four values are used to identify the location of a foreground in the background, including `x y` indicating the upper left corner of the foreground and `w h` indicating width and height. Scale is the maximum of `fg_w/bg_w` and `fg_h/bg_h`. The label (0 or 1) means whether the composite is reasonable in terms of the object placement.

  The training set and the test set each has a CSV file to record their information.

- We also provide a script in `/data_processing/` to generate composite images:

  ```
  python generate_composite.py
  ```

  After running the script, input the foreground ID, background ID, position, label, and storage path to generate your composite image.

## Our SimOPA 
- Download pretrained model from [Baidu Cloud](https://pan.baidu.com/s/1xozUrbiBjGrchdcF1007sA)(access code: up1c) or [GoogleDrive](https://drive.google.com/file/d/11h5L_C-RjgRHDdgk5to7ZfgPFWcUbwxv/view?usp=drive_link) and put it in "best-acc.pth"
- Download pretrained resnet18 from https://download.pytorch.org/models/resnet18-5c106cde.pth or [Baidu Cloud](https://pan.baidu.com/s/1RCrfRiKCpY_SY7Ddeo_B1A)(access code: msqg) and put it in "pretrained_models/resnet18.pth"
- To train a model, run:
```
python train.py
```
- To test the pretrained model, run:
```
python test_model.py
```

## OPA Score

To get general assessment model to evaluate the rationality of object placement, we train SimOPA and [extended SimOPA](https://github.com/bcmi/GracoNet-Object-Placement) on the combination of the whole OPA dataset and the whole OPA-ext dataset, and release the trained models as two assessment models. 

- Download pretrained models from [Baidu Cloud](https://pan.baidu.com/s/1011OoeZ8oQuEX43SN-loKQ)(access code: bcmi) or [GoogleDrive](https://drive.google.com/file/d/1jepX6-8FhIfFEOGob-9rSj25BaZwuXIP/view?usp=sharing) and unzip them to ``eval_opascore/checkpoints``.

- Estimate OPA score with SimOPA:
```
python eval_opascore/simopa.py --image <composite-image-path> --mask <foreground-mask-path> --gpu <gpu-id>
```
We aslo provide several examples of paired composite image and mask in ``eval_opascore/examples``.

- Estimate OPA score with SimOPA-ext:
  1. install packages in ``requirements.txt``.
  2. download the faster-rcnn model pretrained on visual genome from [google drive](https://drive.google.com/file/d/18n_3V1rywgeADZ3oONO0DsuuS9eMW6sN/view) (provided by [Faster-RCNN-VG](https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome)) to ```faster-rcnn/models/faster_rcnn_res101_vg.pth```.
  3. build faster-rcnn：
     ```
     cd faster-rcnn/lib
     python setup.py build develop
     cd ../..
     ```
  4. run evaluation code:
     ```
     python eval_opascore/simopa_ext.py --image <composite-image-path> --mask <foreground-mask-path> --gpu <gpu-id>
     ```

## Extension to FOPA

With a composite image and its composite mask as input, SimOPA can only predict a rationality score for one scale and location in one forward pass, which is very inefficient. We have extended SimOPA to Fast Object Placement Assessment ([FOPA](https://github.com/bcmi/FOPA-Fast-Object-Placement-Assessment)), which can predict the rationality scores for all locations with a pair of background and scaled foreground as input in a single forward pass. 


# Other Resources

+ [Awesome-Object-Placement](https://github.com/bcmi/Awesome-Object-Placement)
+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Image-Composition)


## Bibtex

If you find this work useful for your research, please cite our paper using the following BibTeX  [[arxiv](https://arxiv.org/pdf/2107.01889.pdf)]:

```
@article{liu2021OPA,
  title={OPA: Object Placement Assessment Dataset},
  author={Liu,Liu and Liu,Zhenchen and Zhang,Bo and Li,Jiangtong and Niu,Li and Liu,Qingyang and Zhang,Liqing},
  journal={arXiv preprint arXiv:2107.01889},
  year={2021}
}
```
