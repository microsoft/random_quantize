## Introduction
This is a PyTorch implementation of [Randomized Quantization for Data Agnostic Representation Learning](https://arxiv.org/abs/2212.08663).
This paper introduces a self-supervised augmentation tool for data agnostic representation learning, by quantizing each input channel through a non-uniform quantizer, with the quantized value
sampled randomly within randomly generated quantization bins.
Applying the randomized quantization in conjunction with sequential augmentations on self-supervised contrastive models achieves on par results with 
modality-specific augmentation on vision tasks, and state-of-the-art results on 3D point clouds as well as on audio.
We also demonstrate this method to be applicable for augmenting intermediate embeddings in a deep neural network on the comprehensive [DABS](https://arxiv.org/abs/2111.12062) benchmark which is
comprised of various data modalities.

## Pretrained checkpoints on ImageNet under [moco-v3](https://arxiv.org/abs/2104.02057)

| Augmentations |Pre-trained checkpoints|Linear probe
 :-: | :-:| :-:
|Randomized Quantization (100 epochs) |[model](https://frontiers.blob.core.windows.net/pretraining/projects/whm_ckpt/random_quantize/randomized_quantization_100ep.pth.tar) |42.9
|RRC + Randomized Quantization (100 epochs)  |[model](https://frontiers.blob.core.windows.net/pretraining/projects/whm_ckpt/random_quantize/rrc_randomized_quantization_100ep.pth.tar) |67.9
|RRC + Randomized Quantization (300 epochs)  |[model](https://frontiers.blob.core.windows.net/pretraining/projects/whm_ckpt/random_quantize/rrc_randomized_quantization_300ep.pth.tar) |71.6
|RRC + Randomized Quantization (800 epochs)  |[model](https://frontiers.blob.core.windows.net/pretraining/projects/whm_ckpt/random_quantize/rrc_randomized_quantization_800ep.pth.tar) |72.1

## Pretrained checkpoints on [Audioset](https://ieeexplore.ieee.org/document/7952261) under [byol-a](https://arxiv.org/abs/2103.06695)
We largely follow the experimental settings of [BYOL-A](https://arxiv.org/abs/2103.06695) and treat it as our baseline. We replace the Mixup augmentation used in [BYOL-A](https://arxiv.org/abs/2103.06695) with our randomized quantization. The network is trained on [Audioset](https://ieeexplore.ieee.org/document/7952261) for 100 epoches. On six downstream audio classification datasets, including NSynth ([NS](https://arxiv.org/abs/1704.01279)), UrbanSound8K ([US8K](https://dl.acm.org/doi/abs/10.1145/2647868.2655045)), VoxCeleb1 ([VC1](https://arxiv.org/abs/1706.08612)), VoxForge ([VF](Voxforge.org)), Speech Commands V2 ([SPCV2/12](https://arxiv.org/abs/1804.03209)), Speech Commands V2 ([SPCV2](https://arxiv.org/abs/1804.03209)) , linear probing results are reported as below:
| Method |Augmentations|NS|US8K|VC1|VF|SPCV2/12|SPCV2|Average
 :-: | :-:| :-: | :-: | :-: | :-: | :-: | :-: | :-:
|BYOL-A |RRC + [Mixup](https://arxiv.org/abs/1710.09412)|74.1|79.1|40.1|90.2|91.0|92.2|77.8
|[Our model](https://frontiers.blob.core.windows.net/pretraining/projects/whm_ckpt/random_quantize/randomized_quantization_audio.pth) |RRC + Randomized Quantization|74.2|78.0|45.7|92.6|95.1|92.1|79.6


## Usage
The code has been tested with PyTorch 1.10.0, CUDA 11.3 and CuDNN 8.2.0. 
You are recommended to work with [this docker image](https://hub.docker.com/layers/wuzhiron/pytorch/pytorch1.10.0-cuda11.3-cudnn8-singularity/images/sha256-3e0feccdb9a72cc93e520c35dcf08b928ca379234e4ed7fe7376f7eb53d1dd7a?context=explore).
Bellow are use cases based on [moco-v3](https://github.com/facebookresearch/moco-v3) with minimal effort that allow people having an interest to immediately inject our augmentation into their own project.

1. Call the augmentation as one of torchvision.transforms modules. 
```python
region_num = 8
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
region_num = 8
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
region_num = 8
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

## Citation
```
@Article{wu2022randomized,
      author={Huimin Wu and Chenyang Lei and Xiao Sun and Peng-Shuai Wang and Qifeng Chen and Kwang-Ting Cheng and Stephen Lin and Zhirong Wu},
      journal = {arXiv:2212.08663},
      title={Randomized Quantization for Data Agnostic Representation Learning}, 
      year={2022},
}

```
## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
