# Deep Convolutional Neural Networks for Image Classification

Implementation of various AI papers for image classification  


### Implemented:
<details>
  <summary> Model Architectures </summary>
  
- ResNetV2
- ResNetV2 + Stochastic Depth
- ResNeXt
- SeNet
- MobileNetV2
- MobileNetV3
- DenseNet
<!-- - [ ] ResNeSt
- [ ] EfficientNet
- [ ] NAT
- [ ] TResNet
- [ ] PyramidNet
- [ ] Xception
- [ ] IBN-Net -->

</details>

<details>
  <summary> Other Features </summary>
  
- Step Learning Rate (LR) decay schedule
- HTD (Hyperbolic-Tangent LR Decay schedule)
- Cosine LR decay schedule
- Cutout
- Mixup
- Cutmix
- Mish
<!-- - [ ] Hard and Soft PatchUp -->
<!-- - [ ] Swish
- [ ] EvoNorm -->

</details>

## CIFAR10 Results
GPU: **RTX3090** @1800MHz | **FP16** + **XLA** autoclastering  
**Epochs: 150**  
**Batch Size: 1024** (unless <sub>batch=</sub>)  
Augmentation: random l/r flip -> 4px shift in x/y -> **Cutmix**  
Cos lr schedule 0.5 -> 0.001, 10 epoch warmup  
Optmizer: SGD nesterov m=0.9 

<table>
  <tr>
    <th>Model \ Augmentation</th> 
    <th>Basic</th> 
    <th>Stochastic Depth</th>
    <th>Mixup</th>
    <th>Cutout</th>
    <th>Cutmix</th>
  </tr>
  <tr>
    <th>ResNet50</th> 
    <th>93.46%</th> 
    <th>94.08%</th>
    <th>94.64%</th>
    <th>94.70%</th>
    <th>94.77%</th>
  </tr>
</table>


<table>
  <tr>
    <th colspan="3">⠀⠀⠀⠀⠀⠀⠀Model⠀⠀⠀⠀⠀⠀⠀</th>
    <th>Top1 Accuracy</th>
    <th>Param count</th>
    <th>Training</br>(imgs/sec)</th>
    <th>Inference</br>(imgs/sec)</th>
  </tr>
  <!-- TResNet -->
  <tr>
    <th colspan="3">TResNet</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th rowspan="4"></th>
    <th colspan="2">TResNetM-.5-32px<sub>+HTD+Cutmix</sub></th>
    <th>95.05%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">TResNetM-.5-192px<sub>+HTD+Cutmix</sub></th>
    <th>96.10%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">TResNetM-.75-160px<sub>+Cos+Cutmix</sub></th>
    <th>96.92%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">TResNetM-128px<sub>+Cos+Cutmix</sub></th>
    <th>96.51%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <!-- MobileNetV3 -->
  <tr>
    <th colspan="3">MobileNetV3</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th rowspan="5"></th>
    <th colspan="2">MNetV3S 160px 1.5<sub>+Cos+Cutmix</sub></th>
    <th>94.25%</th>
    <th>1 732 152</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MNetV3S 192px<sub>+Cos+Cutmix</sub></th>
    <th>94.49%</th>
    <th>1 533 896</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MNetV3S 192px 2<sub>+Cos+Cutmix</sub></th>
    <th>95.63%</th>
    <th>1 930 408</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MNetV3S 224px<sub>+Cos+Cutmix</sub></th>
    <th>95.50%</th>
    <th>1 533 896</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MNetV3L<sub>+HTD+Cutmix</sub></th>
    <th>96.37%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <!-- MobileNetV2 -->
  <tr>
    <th colspan="3">MobileNetV2</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th rowspan="6"></th>
    <th colspan="2">MobileNetV2 96px</th>
    <th>93.15%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MobileNetV2 128px</th>
    <th>94.31%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th><sub>batch-size=512</sub></th>
    <th>95.22%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MobileNetV2 192px</th>
    <th>94.43%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th><sub>batch-size=512</sub></th>
    <th>95.53%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MobileNetV2 224px <sub>batch=512</sub></th>
    <th>95.67%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <!-- ResNetV2 -->
  <tr>
    <th colspan="3">ResNetV2</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th rowspan="8"></th>
    <th colspan="2">ResNet18 <sub>mish</sub></th>
    <th>92.81% <sub>93.53%</sub></th>
    <th>692 218</th>
    <th>39 127 <sub>-</sub></th>
    <th>99 028 <sub>-</sub></th>
  </tr>
  <tr>
    <th colspan="2">ResNet34 <sub>mish</sub></th>
    <th>93.69% <sub>94.26%</sub></th>
    <th>1 327 226</th>
    <th>25 534 <sub>-</sub></th>
    <th>75 071 <sub>-</sub></th>
  </tr>
  <tr>
    <th colspan="2">ResNet35 <sub>mish</sub></th>
    <th>94.09% <sub>94.42%</sub></th>
    <th>873 722</th>
    <th>17 304 <sub>-</sub></th>
    <th>58 520 <sub>-</sub></th>
  </tr>
  <tr>
    <th colspan="2">ResNet50 <sub>mish</sub></th>
    <th>94.57% <sub>95.05%</sub></th>
    <th>1 320 570</th>
    <th>12 939 <sub>-</sub></th>
    <th>45 775 <sub>-</sub></th>
  </tr>
  <tr>
    <th colspan="2">ResNet101 <sub>mish</sub></th>
    <th>95.15% <sub>95.57%</sub></th>
    <th>2 530 426</th>
    <th>8 469 <sub>-</sub></th>
    <th>31 813 <sub>-</sub></th>
  </tr>
  <tr>
    <th colspan="2">ResNet152 <sub>mish</sub></th>
    <th>95.62% <sub>95.99%</sub></th>
    <th>3 528 314</th>
    <th>5 954 <sub>-</sub></th>
    <th>23 211 <sub>-</sub></th>
  </tr>
  <tr>
    <th colspan="2">ResNet170 <sub>+ mish</sub></th>
    <th>96.66%</th>
    <th>4 414 202</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">WideResNet18+Cutout+HTD</th>
    <th>94.91%</th>
    <th>11 205 578</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <!-- ResNeXt -->
  <tr>
    <th colspan="3">ResNeXt</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th rowspan="4"></th>
    <th colspan="2">ResNetXt50C32</th>
    <th>94.07%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th>+SD</th>
    <th>94.49%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">ResNetXt101C32</th>
    <th>94.25%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th>+SD</th>
    <th>94.88%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  </th>
  </tr>
  <!-- SeNet -->
  <tr>
    <th colspan="3">SeNet</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr> 
    <th rowspan="4"></th>
    <th colspan="2">SeNet50</th>
    <th>93.40%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th>+SD</th>
    <th>94.25%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">SeNet101</th>
    <th>94.30%</br>94.79%*</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th>+SD⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀</th>
    <th>94.65%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
</table>

* \* -> Reported values

* SD = Stochastic Depth. </br>
  From [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)
* HTD = Hyperbolic-Tangent Learning Rate Decay schedule. </br>
  From [Stochastic Gradient Descent with Hyperbolic-Tangent Decay on Classification](https://arxiv.org/abs/1806.01593)
* Cos = Cosine Learning Rate Decay schedule. </br>
  From [Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
* Mish = Self regularized non-monotonic activation function, f(x) = x*tanh(softplus(x)). </br>
  From [Mish: A Self Regularized Non-Monotonic Activation Function](https://arxiv.org/abs/1908.08681)

