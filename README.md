# EDAR
PyTorch implementation of Deep Convolution Network based on [EDSR](https://arxiv.org/abs/1707.02921) for Compression Artifacts Reduction 

## Requirements
- PyTorch
- tqdm
- Pillow

## Network Architecture

![fig1_EDAR_EXAMPLE](https://user-images.githubusercontent.com/35001605/55075459-60df9680-50d6-11e9-8ce9-1b56c0dbf6cb.png)

![fig2_EDAR](https://user-images.githubusercontent.com/35001605/55075467-65a44a80-50d6-11e9-9d4c-3a40944d79b3.png)

<img src="https://user-images.githubusercontent.com/35001605/55075829-49ed7400-50d7-11e9-8179-ebabded17437.png" width="400" height="200" />

## Visual Results

![fig4_bettertomorrow2](https://user-images.githubusercontent.com/35001605/55057005-39270900-50ab-11e9-8985-cf74f324af11.png)

![fig4_bettertomorrow](https://user-images.githubusercontent.com/35001605/55057007-3af0cc80-50ab-11e9-872b-525bdd8b7480.png)

![fig4_goorinimage](https://user-images.githubusercontent.com/35001605/55075978-9d5fc200-50d7-11e9-91a3-92d3a0a7dcfa.png)

![fig4_bridge](https://user-images.githubusercontent.com/35001605/55076144-0ba48480-50d8-11e9-8381-f3e4417a4f53.png)

![fig4_navi](https://user-images.githubusercontent.com/35001605/55057501-b69f4900-50ac-11e9-8e5a-f810feb63034.png)



## Training

Dataset: DIV 2K train set

Batch size: 16

Patch size: 48x48

Optimizer: Adam

Loss: L1 Loss

Input: Compressed Image by JPEG (jpeg_quality: 10) / RGB

Output: Original Image / RGB

Epoch: 200

[Pre-trained weight]()

## How to train

```
python train.py --images_dir [Your training image path] --outputs_dir ./ --jpeg_quality [10 to 100] --batch_size [num] --num_epochs [num]
```

Pre-trained model trained using the below arguments.
```
python train.py --images_dir ../DIV2K_train_HR --outputs_dir ./ --jpeg_quality 10 --batch_size 16 --num_epochs 200
```

## How to test

```
python test.py --weights_path [your trained weight].pth --image_path [your_image] --outputs_dir ./
```
