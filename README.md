## OPITrack(TIP 2022): An Object Point Set Inductive Tracker for Multi-Object Tracking and Segmentation

This codebase implements **OPITrack**, a highly effective framework for multi-object tracking and segmentation (MOTS) described in: 

[An Object Point Set Inductive Tracker for Multi-Object Tracking and Segmentation](https://ieeexplore.ieee.org/abstract/document/9881968)
**Yan Gao**, Haojun Xu, Yu Zheng, Jie Li, Xinbo Gao
IEEE Transactions on Image Processing 2022

**OPITrack presents a new learning strategy and new triplet loss (Trip-HSAugLoss) for embedding learning. The Trip-HSAugLoss can be used in any MOT/MOTS method.**

## Getting started

This codebase showcases the proposed framework named OPITrack for MOTS using the KITTI MOTS dataset. 

### Prerequisites
Dependencies: 
- Pytorch 1.3.1 (and others), please set up an virtual env and run:
```
$ pip install -r requirements.txt
```
- Python 3.6 (or higher)
- [KITTI Images](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip) + [Annotations](https://www.vision.rwth-aachen.de/media/resource_files/instances.zip)

Note that the scripts for evaluation is included in this repo. After images and instances (annotations) are downloaded, put them under **kittiRoot** and change the path in **repoRoot**/config.py accordingly. 
The structure under **kittiRoot** should looks like:

```
kittiRoot
│   images -> training/image_02/ 
│   instances
│   │    0000
│   │    0001
│   │    ...
│   training
│   │   image_02
│   │   │    0000
│   │   │    0001
│   │   │    ...  
│   testing
│   │   image_02
│   │   │    0000
│   │   │    0001
│   │   │    ... 
```

## Training of OPITrack
The training procedure of instance association is as follows.

1.To generate the segmentation result on the validation set as the instruction of the first step in Testing.

2.To generate the instance DB from videos:
```
$ python -u datasets/MOTSInstanceMaskPool.py
``` 

3.Afterwards start training:
```
$ python -u train_tracker_with_val.py car_finetune_tracking
``` 
The best tracker on the validation set will be saved under the folder specified in **repoRoot**/config_mots/car_finetune_tracking.py.


## Training of SpatialEmbedding

Note that the training of SpatialEmbedding needs KITTI object detection left color images as well as the KINS annotations.
Please download images from [KITTI dataset](http://www.cvlibs.net/download.php?file=data_object_image_2.zip), and unzip the zip file under **kittiRoot**.
Please download two KINS annotation json files from [instances_train.json,instances_val.json](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset), and put files under **kittiRoot**.

For the training of SpatialEmbedding, we follow the original training setting of [SpatialEmbedding](https://github.com/davyneven/SpatialEmbeddings). 
Different foreground weights are adopted for different classes (200 for cars and 50 for pedestrians). In this following, we take cars for example to explain training procedures. 

0.As there are many in-valid frames in MOTS that contain no cars, we only select these valid frames for training SpatialEmbedding.
 ```
$ python -u datasets/MOTSImageSelect.py
``` 

1.To parse KINS annotations, run:
```
$ python -u datasets/ParseKINSInstance.py
``` 
After this step, KINS annotations are saved under **kittiRoot**/training/KINS/ and **kittiRoot**/testing/KINS/.

2.To generate these crops do the following:
```
$ python -u utils/generate_crops.py
``` 
After this step, crops are saved under **kittiRoot**/crop_KINS. (roughly 92909 crops)

3.Afterwards start training on crops: 
```
$ python -u train_SE.py car_finetune_SE_crop
```

4.Afterwards start finetuning on KITTI MOTS with BN fixed:
```
$ python -u train_SE.py car_finetune_SE_mots
```


## Cite us
We borrow some code from [SpatialEmebdding](https://github.com/davyneven/SpatialEmbeddings) and [PointTrack](https://github.com/detectRecog/PointTrack).
```
@ARTICLE{Gao_OPITrack,
  author={Gao, Yan and Xu, Haojun and Zheng, Yu and Li, Jie and Gao, Xinbo},
  journal={IEEE Transactions on Image Processing}, 
  title={An Object Point Set Inductive Tracker for Multi-Object Tracking and Segmentation}, 
  year={2022},
  volume={31},
  number={},
  pages={6083-6096},
  doi={10.1109/TIP.2022.3203607}}
```

## Contact
If you find problems in the code, please open an issue.

For general questions, please contact the author Yan Gao (gyy1101@outlook.com).


## License

This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](http://creativecommons.org/licenses/by-nc/4.0/).






