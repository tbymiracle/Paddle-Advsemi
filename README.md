# Paddle-Advsemi

## 1.Introduction
This project is based on the paddlepaddle_V2.2.0-rc0 framework to reproduce Brain-Tumor-Segmentation. 

We put the torch version in AdvSemiSeg-torch/ (the [official code](https://github.com/hfslyc/AdvSemiSeg)

We put our project in AdvSemiSeg-paddle/. Our project can achieve almost the same results. 
## 2.Result

The model is trained on the train set of VOC, and we test it on test set of VOC.

<!-- Average result in all 240 slices:

 Version | Dice Complete | Dice Core | Dice Enhancing
 ---- | ----- | -----  | -----
 keras version(official)  | 0.907  | 0.961 | 1.0
 paddle version(ours) | 0.907|  0.961 | 1.0
 
 Result in slice 113:
 
  Version | Dice Complete | Dice Core | Dice Enhancing
 ---- | ----- | -----  | -----
 keras version(official)  | 0.828  | 0.935 | 1.0
 paddle version(ours) | 0.828|  0.935 | 1.0 -->
 


## 3.Requirements

 * Hardware：GPU（Tesla V100-32G is recommended）
 * Framework:  PaddlePaddle >= 2.2.0rc0


## 4.Quick Start

### Step1: Clone

``` 
git clone https://github.com/tbymiracle/Paddle-Advsemi.git
cd AdvSemiSeg-paddle
``` 

### Step2: Training

```  
CUDA_VISIBLE_DEVICES=0 python train.py
```  
### Step3: Evaluating

```  
CUDA_VISIBLE_DEVICES=0 python evaluate_voc.py
```  

## 5.Align

We use the [`repord_log`](https://github.com/WenmuZhou/reprod_log) tool to align.


```  
cd AdvSemiSeg-paddle
python bp_align.py # paddle backward
cd AdvSemiSeg-torch
python bp_align.py # torch backward
python AdvSemiSeg-paddle/check_diff.py # check diff of backward step.
```  

        
* Network structure transfer.
* Weight transfer:
  * models of torch version to do the aligh : 链接: https://pan.baidu.com/s/1rAl0rdQAxnM_RLWPw8c7zg 提取码: jknr 复制这段内容后打开百度网盘手机App，操作更方便哦
  * models of paddle version transfered from torch: 链接: https://pan.baidu.com/s/1sv-69Uv1tqRpnIuMNN0quQ 提取码: zk5i 复制这段内容后打开百度网盘手机App，操作更方便哦
* Verify the network.
* Forward align
  * AdvSemiSeg-paddle/
* Backward align
  * AdvSemiSeg-paddle/
* Train align
