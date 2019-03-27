# AR_0hyeNet
PyTorch implementation of Deep Convolution Networks for Compression Artifacts Reduction

## Network Architecture

![fig1_AR_EXAMPLE](https://user-images.githubusercontent.com/35001605/55053564-f27fe180-509f-11e9-9d26-8fd01b684b56.png)

![fig2_AR_0hyeNet](https://user-images.githubusercontent.com/35001605/55053576-fdd30d00-509f-11e9-8c85-7efcff08f164.png)

<img src="https://user-images.githubusercontent.com/35001605/55056900-dd5c8000-50aa-11e9-9b6e-2d6b88f21467.png" width="400" height="200" />

## Visual Results

![fig4_bettertomorrow2](https://user-images.githubusercontent.com/35001605/55057005-39270900-50ab-11e9-8985-cf74f324af11.png)

![fig4_bettertomorrow](https://user-images.githubusercontent.com/35001605/55057007-3af0cc80-50ab-11e9-872b-525bdd8b7480.png)

![fig4_navi](https://user-images.githubusercontent.com/35001605/55057501-b69f4900-50ac-11e9-8e5a-f810feb63034.png)

## Training

Dataset: DIV 2K train set

Batch size: 16

Patch size: 48x48

Optimizer: Adam

Loss: L1 Loss

Input: Jpeg Compressed RGB Image(Compression Quality:10, compressed by PIL(Python Image Library))

Output: Original RGB Image

Epoch: 2000

## How to train

```
python train.py --images_dir [Your training image path] --outputs_dir ./ --jpeg_quality [10 to 100] --batch_size [num] --num_epochs [num]
```

Pre-trained model trained using the below arguments.
```
python train.py --images_dir ../DIV2K_train_HR --outputs_dir ./ --jpeg_quality 10 --batch_size 16 --num_epochs 2000
```

## How to test

```
python test.py --weights_path [your trained weight].pth --image_path [your_image] --outputs_dir ./
```
