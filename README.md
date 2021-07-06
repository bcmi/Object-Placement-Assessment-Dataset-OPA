# Object-Shadow-Generation-Dataset-DESOBA

**Object-Placement-Assessment** is to verifies whether a composite image is plausible in terms of the object placement. The foreground object should be placed at a reasonable location on the background considering location, size, occlusion, semantics, and etc.

Our dataset **OPA** is a synthesized dataset for Object Placement Assessment based on [COCO](http://cocodataset.org) dataset.  We select unoccluded objects from multiple categories as our candidate foreground objects. The foreground objects are pasted on their compatible background images with random sizes and locations to form composite images, which are sent to human annotators for rationality labeling. Finally, we split the collected dataset into training set and test set, in which the background images and foreground objects have no overlap between training set and test set. We show some example positive and negative images in our dataset examples in the figure below.

<img src='Examples/dataset_sample.png' align="center" width=1024>

Illustration of OPA dataset samples: Some positive and negative samples in our OPA dataset and the inserted foreground objects are marked with red outlines. Top row: positive samples; Bottom rows: negative samples, including objects with inappropriate size (e.g., f, g, h), without supporting force (e.g., i, j, k), appearing in the semantically unreasonable place (e.g., l, m, n), with unreasonable occlusion (e.g., o, p, q), and with inconsistent perspectives (e.g., r, s, t).

Our OPA dataset contains 62,074 training images and 11,396 test images, in which the foregrounds/backgrounds in training set and test set have no overlap. The training (resp., test) set contains 21,351 (resp.,3,566) positive samples and 40,724 (resp., 7,830) negative samples. Besides, the training (resp., test) set contains 2,701 (resp., 1,436) unrepeated foreground objects and1,236 (resp., 153) unrepeated background images. The OPA dataset is provided in [**Baidu Cloud**](https://pan.baidu.com/s/1r2p8Ll1AjUNLSHru87XsqQ) (access code: qb1r), or [**Google Drive**](https://drive.google.com/file/d/11GiT3J4Vaf6l-WOUN1xjDpzc5SQCUFXj/view?usp=sharing).




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

- Download the OPA dataset. We show the file structure in below:

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

  Any of the backgrounds or foregrounds has its own ID for identification. Different kinds of foreground and background are placed in different folders. The masks corresponding to the images are placed in the same folder with a mask prefix.

  Four values are used to identify the location of a foreground in the background, including `x y` indicating the upper left corner of the foreground and `w h` indicating width and height . Scale is the maximum of `fg_w/bg_w` and `fg_h/bg_h`. The label(0 or 1) means whether the composite is reasonable.

  The training set and the test set each has a CSV file to record their information.

- We also provide a script to generate composite, run:

  ```
  python generate_composite.py
  ```

  `generate_composite.py` is available in `/data_processing/`. After the script runs, input the foreground ID, background ID, position, label and storage path to generate your composite. Illegal input will be prompted and re-input is required.

  

## Bibtex

If you find this work is useful for your research, please cite our paper using the following **BibTeX  [[arxiv](https://arxiv.org/pdf/2107.01889.pdf)]:**

```
@article{liu2021OPA,
  title={OPA: Object Placement Assessment Dataset},
  author={Liu,Liu and Zhang,Bo and Li,Jiangtong and Niu,Li and Liu,Qingyang and Zhang,Liqing},
  journal={arXiv preprint arXiv:2107.01889},
  year={2021}
}
```

