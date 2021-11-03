# Paddle-Advsemi

## 1.Introduction
This project is based on the paddlepaddle_V2.2 framework to reproduce Brain-Tumor-Segmentation. 

We put the keras version in keras-bts/ (the [official code](https://github.com/jadevaibhav/Brain-Tumor-Segmentation-using-Deep-Neural-networks) based on keras is implemented in jupyter notebook so we change it a little). 

We put our project in paddle-bts/. Our project can achieve almost the same results. 
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
 * Framework:  PaddlePaddle >= 2.1.2


## 4.Quick Start

### Step1: Clone

``` 
git clone https://github.com/tbymiracle/Paddle-Brain-Tumor-Segmentation.git
cd paddle-bts
``` 

### Step2: Training

```  
CUDA_VISIBLE_DEVICES=0 python train.py
```  
### Step3: Evaluating

```  
CUDA_VISIBLE_DEVICES=0 python test.py # test in one slice
CUDA_VISIBLE_DEVICES=0 python test_all.py # test in all 240 slices
```  

## 5.Align

We use the [`repord_log`](https://github.com/WenmuZhou/reprod_log) tool to align.

The author use keras framework in the official version, so it is hard to align results in all the steps.

Thus, we can only do the forward align.

```  
python keras-bts/forward.py # keras forward
python paddle-bts/forward.py # paddle forward
python keras-bts/check_diff.py # check diff of forward step.
```  

        
* Network structure transfer.
* Weight transfer:
  * model of keras version to do the aligh : [keras-bts/trial_input_cascasde_acc.h5](https://github.com/tbymiracle/Paddle-Brain-Tumor-Segmentation/blob/main/keras-bts/trial_input_cascasde_acc.h5)
  * model of paddle version transfered from keras: [paddle-bts/bts_paddle_ub.pdparams](https://github.com/tbymiracle/Paddle-Brain-Tumor-Segmentation/blob/main/paddle-bts/bts_paddle_ub.pdparams) 
* Verify the network.
* Forward align
  * keras-bts/forward_keras.npy
  * paddle-bts/forward_paddle.npy
  * keras-bts/forward_diff.log
* Train align
  * As shown in part 2.
