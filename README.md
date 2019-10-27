# LargeScaleIncrementalLearning
This is the implementation of CVPR 2019 paper "Large Scale Incremental Learning". If the paper and code helps you, we would appreciate your kindly citations of our paper.
```
@inproceedings{wu2019large,
  title={Large Scale Incremental Learning},
  author={Wu, Yue and Chen, Yinpeng and Wang, Lijuan and Ye, Yuancheng and Liu, Zicheng and Guo, Yandong and Fu, Yun},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={374--382},
  year={2019}
}
```


## Abstract
In this paper, we proposed a new method to address the imbalance issue in incremental learning, which is critical when the number of classes becomes large. Firstly, we validated our hypothesis that the classifier layer (the last fully connected layer) has a strong bias towards the new classes, which has substantially more training data than the old classes. Secondly, we found that this bias can be effectively corrected by applying a linear model with a small validation set. Our method has excellent results on two large datasets with 1,000+ classes (ImageNet ILSVRC 2012 and MS-Celeb-1M), outperforming the state-of-the-art by a large margin (11.1% on ImageNet ILSVRC 2012 and 13.2% on MS-Celeb-1M).

## Environment Setup
Words before the code: most codes and experiments are finished in late 2017 and earlier 2018. It is hard to retrieve exact the same environment for experiments, which I remember that the system was in Ubuntu 14. CUDA and tensorflow are all with earlier versions. I re-installed the system several times last year (2018) because some conficts in setting up environment for pytorch, which was original fit for caffe and tensorflow. And also, I upgraded the system form Ubuntu 14 to Ubuntu 16.

The resnet implementation is the official tensorflow official models at:
```
https://github.com/tensorflow/models
```
In the latest repo, the most similar implementaion is:
```
https://github.com/tensorflow/models/blob/master/official/r1/resnet/imagenet_main.py
```
Unfortunately, we were not able to run our code with the latest tensorflow-2.0 or tensorflow-1.14. 

We understand that how important it is to reproduce the results of published papers. I find a compatible tensorflow-1.5 version that is able to run the code with my current CUDA settings. To not mess up with my current software enviroment for pytorch mostly, I use the virtualenv to set up a seperate environment for experiments. The dependency of the code is lite and should be able work in most cases if you get the tensorflow version set up correctly. I summarize my current environment for reference.  


System Information<br/>
Distributor ID:	Ubuntu<br/>
Description:	Ubuntu 16.04.3 LTS<br/>
Release:	16.04<br/>
Codename:	xenial<br/>


CUDA version: 9.0<br/>
CUDNN version: 7.0.5<br/>
GPU: TITAN X (Pascal) 12GB<br/>

Python version: 3.5.2<br/>
Environemnt is setup using virtualenv.<br/>
```
virtualenv venv
pip install scipy
pip install tensorflow-gpu==1.5
```

To activate the environment:
```
source venv/bin/activate
```

## To run the code
Dataset:<br/>
We mainly clean the code for ImageNet dataset and leave the other two datasets (CIFAR and MS-Celeb-1M) in future work.

Data preparation:<br/>
We have put essential files in the repo so that what you need to do is to download the ImageNet-1000 images. 
For the images, we used the ILSVRC2016_CLS-LOC.tar.gz file, which should be kept unchanged since 2012. <br/> 
Some links to handle the 50K validation images to split validation images in class folders.
```
https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset
```

After downloading and processing, the structure in the data forlder (dataImageNet100 and dataImageNet1000) should be like:
```
|--train
|  |--n01440764
|  |--n01443537
|  |.......
|--train.txt
|--val
|  |--n01440764
|  |--n01443537
|  |.......
|--val.txt
```

Experiments run command:
```
CUDA_VISIBLE_DEVICES=0 python imagenet_main.py 1>log 2>&1
```

## Results
We run the code once and report the result here. Top-5 accuracy is reported on ImageNet dataset. We observe similar results on ImageNet-100 form this repo with new environment and what we reported in paper. On ImageNet-1000, we notice some performance drop in several first increments but the final results are even better than what we reported in our paper, which might be caused by different operation systems and software environments. 

Training on ImageNet-100 takes around 15 hours. Results are:

|           | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|----------|---:|---:|---:|---:|---:|---:|---:|---:|---:|----:|
| In paper (%)    | 98.40 | 96.20 | 94.00 | 92.90 | 91.10 | 89.40 | 88.10 | 86.50 | 85.40 | 84.40|
| This Repo (%) | 98.79 | 96.10 | 95.06 | 93.50 | 90.96 | 89.73 | 89.02 | 87.22 | 85.97 | 84.24|
| Before Bias Correction (%) | - | 95.70 | 93.06 | 90.75 | 86.60 | 86.06 | 83.60 | 81.25 | 78.28 | 76.32|
| \beta  | 1.0 | 0.5900 | 0.5140 | 0.4742 | 0.4839 | 0.4648 | 0.4389 | 0.4297 | 0.4335 | 0.3941|
| \gamma | 0.0 | -0.4416 | -0.5401 | -0.4701 | -0.5323 | -0.4672 | -0.4830 | -0.5349 | -0.5804 | -0.4609|
| Training Samples | 12800 | 14417 | 14600 | 14600 | 14600 | 14441 | 14600 | 14620 | 14331 |14566|
| Val Samples | 200 | 400 | 300 | 240 | 250 | 240 | 210 | 160 | 180 | 200 |
| Test Samples | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000|


Training on ImageNet-100 takes around 100 hours. Results are:

|           | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 1000 |
|----------|---:|---:|---:|---:|---:|---:|---:|---:|---:|----:|
| In paper (%)    | 94.10 | 92.50 | 89.60 | 89.10 | 85.70 | 83.20 | 80.20 | 77.50 | 75.00 | 73.20|
| This Repo (%) | 93.72 | 91.46 | 88.70 | 86.63 | 84.64 | 83.08 | 81.37 | 79.82| 78.22 |76.76|
| Before Bias Correction (%) | - | 89.64 | 84.50 | 80.84 |78.07 |74.89 | 72.66 | 70.38| 67.93 | 63.34|
| \beta  | 1.0 | 0.7873 | 0.7382 | 0.7053 |  0.6884 |  0.6704 | 0.6609 | 0.6515 | 0.6239 | 0.6334 |
| \gamma | 0.0 | -0.5586 | -0.5759 | -0.5015 | -0.5505 | -0.5137 | -0.4677 | -0.4471 | -0.4118 | -0.4064 |
| Training Samples | 126856 | 144159 | 144505 | 144301 | 143776 | 145238 | 143391| 144418|143277| 143046 |
| Val Samples | 2000 | 4000 | 3000 | 2400 | 2500 | 2400 | 2100 | 1600 | 1800 | 2000 |
| Test Samples | 5000 | 10000 | 15000 | 20000 | 25000 | 30000 | 35000 | 40000 | 45000 | 50000|

Results are from one run of the model on ImageNet-100 and ImageNet-1000. Log files are located at 
```
./logs/log-ImageNet100
./logs/log-ImageNet1000
```

## Implementation notes
Class Order:<br/>
To keep the same order with iCaRL (https://github.com/srebuffi/iCaRL), we use the same random seed (1993) from numpy to generate the order. 

Distilling Loss:<br/>
We store the previous network for distilling loss. 

Bias Correction:<br/>
After learning the Bias Correction parameters (\beta and \gamma), classifier after correction is used for the distilling loss in the next incremental training. 

Validation Samples from exemplars:<br/>
10\% selection is limited on exemplars (old classes). Samples from new classes will match the same number of validation samples. 

## Useful links
Awesome-Incremental-Learning: https://github.com/xialeiliu/Awesome-Incremental-Learning

## Contact
If you found any issue of the code, please contact Yue Wu ([@wuyuebupt](http://github.com/wuyuebupt), Email: yuewu@ece.neu.edu or wuyuebupt@gmail.com)
