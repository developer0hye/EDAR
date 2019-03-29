# EDAR
PyTorch implementation of Deep Convolution Network based on [EDSR](https://arxiv.org/abs/1707.02921) for Compression Artifacts Reduction 

## Requirements
- PyTorch
- tqdm
- Pillow

## Network Architecture

![fig1_EDAR_EXAMPLE](https://user-images.githubusercontent.com/35001605/55232723-b7310e80-5269-11e9-8d28-9d3268f6170f.png)

![fig2_EDAR](https://user-images.githubusercontent.com/35001605/55075467-65a44a80-50d6-11e9-9d4c-3a40944d79b3.png)

<img src="https://user-images.githubusercontent.com/35001605/55075829-49ed7400-50d7-11e9-8179-ebabded17437.png" width="400" height="200" />

## Visual Results

![fig4_bettertomorrow2_better](https://user-images.githubusercontent.com/35001605/55233222-3ffc7a00-526b-11e9-89e2-7d06af04dc54.png)

![fig4_bettertomorrow_better](https://user-images.githubusercontent.com/35001605/55233220-3d9a2000-526b-11e9-8220-e2801fe43257.png)

![fig4_goorinimage](https://user-images.githubusercontent.com/35001605/55236478-5e667380-5273-11e9-93f2-553399efdefb.png)

![fig4_bridge](https://user-images.githubusercontent.com/35001605/55236523-79d17e80-5273-11e9-8ab3-4292460e2d5b.png)

![fig4_iu](https://user-images.githubusercontent.com/35001605/55232466-032f8380-5269-11e9-904d-af1dafa6075e.png)

![fig4_ronaldo](https://user-images.githubusercontent.com/35001605/55232475-088cce00-5269-11e9-8c52-c0184140c764.png)

![fig4_mpeg](https://user-images.githubusercontent.com/35001605/55234087-a5ea0100-526d-11e9-9519-3b766ad9dde5.png)

![fig4_navi](https://user-images.githubusercontent.com/35001605/55057501-b69f4900-50ac-11e9-8e5a-f810feb63034.png)



## Training

Dataset: DIV 2K train set + ...(custom dataset...)

Batch size: 16

Patch size: 48x48

Optimizer: Adam

Loss: L1 Loss

Input: Compressed Image by JPEG (jpeg_quality: rand(0 to 10)) / RGB

Output: Original Image / RGB

Epoch: 450

[Pre-trained weight](https://drive.google.com/open?id=1p8yzQIWPPtS6DoVOSCggTvN5_OoK5fVH)

## How to train

```
python train.py --images_dir [Your training image path] --outputs_dir ./ --jpeg_quality [10 to 100] --batch_size [num] --num_epochs [num]
```

Pre-trained model was trained using the below arguments.
```
python train.py --images_dir ../DIV2K_train_HR --outputs_dir ./ --jpeg_quality 10 --batch_size 16 --num_epochs 200
```

## How to test

```
python test.py --weights_path [your trained weight].pth --image_path [your_image] --outputs_dir ./
```
