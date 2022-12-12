# Introduction
This is a PyTorch implementation of [Randomized Quantization for Data Agnostic Representation Learning](https://arxiv.org/).
This paper introduces a self-supervised augmentation tool for data agnostic representation learning, by quantizing each input channel through a non-uniform quantizer, with the quantized value
sampled randomly within randomly generated quantization bins.
Applying the randomized quantization in conjunction with sequential augmentations on self-supervised contrastive models achieves on par results with 
modality-specific augmentation on vision tasks, and state-of-the-art results on 3D point clouds as well as on audio.
We also demonstrate this method to be applicable for augmenting intermediate embeddings in a deep neural network on the comprehensive [DABS](https://arxiv.org/abs/2111.12062) benchmark which is
comprised of various data modalities.

# Usage
The code has been tested with PyTorch 1.10.0, CUDA 11.3 and CuDNN 8.2.0. 
You are recommended to work with [this docker image](https://hub.docker.com/layers/wuzhiron/pytorch/pytorch1.10.0-cuda11.3-cudnn8-singularity/images/sha256-3e0feccdb9a72cc93e520c35dcf08b928ca379234e4ed7fe7376f7eb53d1dd7a?context=explore).
Bellow are use cases based on [moco-v3](https://github.com/facebookresearch/moco-v3) with minimal effort that allow one to immediately inject our augmentation into their own project.

1. Call the augmentation as one of torchvision.transforms modules. 
```python
region_num = 2
https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_moco.py#L262-L285
augmentation1 = [
    transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
    RandomizedQuantizationAugModule(region_num, transforms_like=True),
    transforms.ToTensor()
]
augmentation2 = [
    transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
    RandomizedQuantizationAugModule(region_num, transforms_like=True),
    transforms.ToTensor()
]
```
2. Apply randomly our augmentation with a given probability.
```python
region_num = 2
p_random_apply1, p_random_apply2 = 0.5, 0.5
#https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_moco.py#L262
augmentation1 = [
    transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
    RandomizedQuantizationAugModule(region_num, p_random_apply_rand_quant=p_random_apply1),
    transforms.ToTensor()
]
augmentation2 = [
    transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
    RandomizedQuantizationAugModule(region_num, p_random_apply_rand_quant=p_random_apply2),
    transforms.ToTensor()
]
```
3. Call the augmentation in forward(). This is faster than above two usages since the augmentation is deployed on GPUs.
```python
# https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/moco/builder.py#L35
region_num = 2
self.rand_quant_layer = RandomizedQuantizationAugModule(region_num)
# https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/moco/builder.py#L86-L94
q1 = self.predictor(self.base_encoder(self.rand_quant_layer(x1)))
q2 = self.predictor(self.base_encoder(self.rand_quant_layer(x2)))

with torch.no_grad():  # no gradient
    self._update_momentum_encoder(m)  # update the momentum encoder

    # compute momentum features as targets
    k1 = self.momentum_encoder(self.rand_quant_layer(x1))
    k2 = self.momentum_encoder(self.rand_quant_layer(x2))
```

# Citation
```


```
