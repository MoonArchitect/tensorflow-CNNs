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
**Batch Size: 1024** (or <sub>b=512</sub>)  
Augmentation: random l/r flip -> 4px shift in x/y -> **Cutmix**  
Cos lr schedule 0.5 -> 0.001, 10 epoch warmup  
Optmizer: SGD nesterov m=0.9 

<table>
  <tr>
    <th>Model \ Augmentation</th> 
    <th>Basic</th> 
    <!-- <th>Stochastic Depth</th> -->
    <th>Mixup</th>
    <th>Cutout</th>
    <th>Cutmix</th>
  </tr>
  <tr>
    <th>ResNet50</th> 
    <th>93.46%</th> 
    <!-- <th>94.08%</th> -->
    <th>94.64%</th>
    <th>94.70%</th>
    <th>94.77%</th>
  </tr>
  <tr>
    <th>MobileNetV3S 192px <sub>w=2</sub></th> 
    <th>94.35%</th> 
    <!-- <th>-</th> -->
    <th>95.14%</th>
    <th>95.44%</th>
    <th>95.85%</th>
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
  <!-- MobileNetV3 -->
  <tr> <!-- MobileNetV3S -->
    <th colspan="7">MobileNetV3</th>
  </tr>
  <tr>
    <th rowspan="18"></th>
    <th colspan="6">MobileNetV3S<sub></sub></th>
  </tr>
  <tr>
    <th rowspan="10"></th>
    <th>128px<sub></sub></th>
    <th>93.72%</th>
    <th rowspan="4">944 890</th>
    <th>11 845</th>
    <th>66 137</th>
  </tr>
  <tr>
    <th>160px<sub></sub></th>
    <th>94.41%</th>
    <th>9 177</th>
    <th>55 245</th>
  </tr>
  <tr>
    <th>192px<sub></sub></th>
    <th>94.86%</th>
    <th>10 675</th>
    <th>43 226</th>
  </tr>
  <tr>
    <th>224px<sub></sub></th>
    <th>95.53%</th>
    <th>8 040</th>
    <th>35 209</th>
  </tr>
  <tr>
    <th>128px<sub> <abbr title="width_factor">w=2</abbr></sub></th>
    <th>95.10%</th>
    <th>3 575 242</th>
    <th>7 254</th>
    <th>44 722</th>
  </tr>
  <tr>
    <th>128px<sub> <abbr title="width_factor">w=4</abbr> b=512 </sub></th>
    <th>95.99%</th>
    <th>13 925 762</th>
    <th>2 608</th>
    <th>23 516</th>
  </tr>
  <tr>
    <th>160px<sub> <abbr title="width_factor">w=2</abbr></sub></th>
    <th>95.56%</th>
    <th rowspan="4">3 575 242</th>
    <th>5 512</th>
    <th>31 467</th>
  </tr>
  <tr>
    <th>192px<sub> <abbr title="width_factor">w=2</abbr></sub></th>
    <th>96.02%</th>
    <th>7 652</th>
    <th>26 653</th>
  </tr>
  <tr>
    <th>224px<sub> <abbr title="width_factor">w=2</abbr></sub></th>
    <th>96.22%</th>
    <th>5 711</th>
    <th>20 760</th>
  </tr>
  <tr>
    <th>224px<sub> <abbr title="width_factor">w=2</abbr> b=512</sub></th>
    <th>96.30%</th>
    <th>5 379</th>
    <th>19 635</th>
  </tr>
  <tr> <!-- MobileNetV3L -->
    <th colspan="6">MobileNetV3L<sub></sub></th>
  </tr>
  <tr>
    <th rowspan="6"></th>
    <th>128px<sub> </sub></th>
    <th>95.57%</th>
    <th rowspan="4">3 011 882</th>
    <th>5 765</th>
    <th>34 980</th>
  </tr>
  <tr>
    <th>160px<sub></sub></th>
    <th>96.07%</th>
    <th>4 303</th>
    <th>25 000</th>
  </tr>
  <tr>
    <th>192px<sub> b=512</sub></th>
    <th>96.58%</th>
    <th>4 531</th>
    <th>17 142</th>
  </tr>
  <tr>
    <th>224px<sub> b=512</sub></th>
    <th>96.52%</th>
    <th>3 494</th>
    <th>13 591</th>
  </tr>
  <tr>
    <th>128px<sub> <abbr title="width_factor">w=2</abbr></sub></th>
    <th>96.06%</th>
    <th rowspan="2">11 737 426</th>
    <th>3 286</th>
    <th>20 087</th>
  </tr>
  <tr>
    <th>192px<sub> <abbr title="width_factor">w=2</abbr>  b=512 </sub></th>
    <th>96.95%</th>
    <th>2 509</th>
    <th>9 733</th>
  </tr>
  <!-- MobileNetV2 -->
  <tr>
    <th colspan="7">MobileNetV2</th>
  </tr>
  <tr>
    <th rowspan="6"></th>
    <th colspan="2">96px</th>
    <th>94.45%</th>
    <th rowspan="5">2,270,794</th>
    <th>5 201</th>
    <th>42 184</th>
  </tr>
  <tr>
    <th colspan="2">128px</th>
    <th>95.10%</th>
    <th>7 739</th>
    <th>27 789</th>
  </tr>
  <tr>
    <th colspan="2">160px<sub></sub></th>
    <th>95.52%</th>
    <th>5 377</th>
    <th>19 118</th>
  </tr>
  <tr>
    <th colspan="2">192px</th>
    <th>95.78%</th>
    <th>4 057</th>
    <th>15 478</th>
  </tr>
  <tr>
    <th colspan="2">224px <sub>batch=512</sub></th>
    <th>96.20%</th>
    <th>2 963</th>
    <th>11 179</th>
  </tr>
  <tr>
    <th colspan="2">128px<sub> <abbr title="width_factor">w=2</abbr></sub></th>
    <th>96.27%</th>
    <th>7 953 802</th>
    <th>4 510</th>
    <th>16 414</th>
  </tr>
  <!-- ResNetV2 -->
  <tr>
    <th colspan="7">ResNetV2</th>
  </tr>
  <tr>
    <th rowspan="11"></th>
    <th colspan="2">ResNet18 <sub>mish</sub></th>
    <th>92.81% <sub>93.53%</sub></th>
    <th>692 218</th>
    <th>39 127 <sub>-4%</sub></th>
    <th>99 028 <sub>-4%</sub></th>
  </tr>
  <tr>
    <th colspan="2">ResNet34 <sub>mish</sub></th>
    <th>93.69% <sub>94.26%</sub></th>
    <th>1 327 226</th>
    <th>25 534 <sub>-4%</sub></th>
    <th>75 071 <sub>-4%</sub></th>
  </tr>
  <tr>
    <th colspan="2">ResNet35 <sub>mish</sub></th>
    <th>94.09% <sub>94.42%</sub></th>
    <th>873 722</th>
    <th>17 304 <sub>-5%</sub></th>
    <th>58 520 <sub>-4%</sub></th>
  </tr>
  <tr>
    <th colspan="2">ResNet50 <sub>mish</sub></th>
    <th>94.57% <sub>95.05%</sub></th>
    <th>1 320 570</th>
    <th>12 939 <sub>-5%</sub></th>
    <th>45 775 <sub>-3%</sub></th>
  </tr>
  <tr>
    <th colspan="2">ResNet101 <sub>mish</sub></th>
    <th>95.15% <sub>95.57%</sub></th>
    <th>2 530 426</th>
    <th>8 469 <sub>-6%</sub></th>
    <th>31 813 <sub>-5%</sub></th>
  </tr>
  <tr>
    <th colspan="2">ResNet152 <sub>mish</sub></th>
    <th>95.62% <sub>95.99%</sub></th>
    <th>3 528 314</th>
    <th>5 954 <sub>-7%</sub></th>
    <th>23 211 <sub>-3%</sub></th>
  </tr>
  <tr>
    <th colspan="2">ResNet170 <sub>mish</sub></th>
    <th>95.68% <sub>96.18%</sub></th>
    <th rowspan="2">4 190 330</th>
    <th rowspan="2">5 113 <sub>-8%</sub></th>
    <th rowspan="2">20 246 <sub>-5%</sub></th>
  </tr>
  <tr>
    <th></th>
    <th><sub>+mish +lr=.75</sub></th>
    <th>96.44%</th>
  </tr>
  <tr>
    <th colspan="2">WideResNet34 <sub> <abbr title="width_factor">w=4</abbr> </sub></th>
    <th>96.40%</th>
    <th>21 123 530</th>
    <th>5 605</th>
    <th>20 382</th>
  </tr>
  <tr>
    <th></th>
    <th colspan="1"><sub> <abbr title="width_factor">w=8</abbr> </sub></th>
    <th>96.91%</th>
    <th>84 419 466</th>
    <th>1 773</th>
    <th>6 539</th>
  </tr>
  <tr>
    <th colspan="2">WideResNet170 <sub>+mish <abbr title="width_factor">w=2</abbr> </sub></th>
    <th>97.18%</th>
    <th>16 588 010</th>
    <th>2 511</th>
    <th>9 392</th>
  </tr>
  <!-- SeNet -->
  <tr>
    <th colspan="7">SeNet</th>
  </tr>
  <tr>
    <th rowspan="6"></th>
    <th colspan="2">senet35 <sub>mish</sub></th>
    <th>94.33% <sub>94.7%</sub></th>
    <th>982 946</th>
    <th>15162<sub>-8%</sub></th>
    <th>52390<sub>-9%</sub></th>
  </tr>
  <tr>
    <th colspan="2">SeNet50<sub>mish</sub></th>
    <th>94.76% <sub>95.17%</sub></th>
    <th>1 483 314</th>
    <th>11277<sub>-8%</sub></th>
    <th>39142<sub>-5%</sub></th>
  </tr>
  <tr>
    <th colspan="2">SeNet101 <sub>mish</sub></th>
    <th>95.43% <sub>96.03%</sub></th>
    <th>2 837 058</th>
    <th>7223<sub>-9%</sub></th>
    <th>25303<sub>-3%</sub></th> 
  </tr>
  <tr>
    <th></th>
    <th><sub>+mish <abbr title="width_factor">w=2</abbr></sub></th>
    <th>96.69%</th>
    <th>11 215 970</th>
    <th>3 985</th>
    <th>13 820</th> 
  </tr>
  <tr>
    <th colspan="2">SeNet152 <sub>mish</sub></th>
    <th>95.78% <sub>96.49%</sub></th>
    <th>3 953 714</th>
    <th>4 976<sub>-8%</sub></th>
    <th>18 747<sub>-6%</sub></th> 
  </tr>
  <tr>
    <th colspan="2">SeNet170 <sub>+mish w=2 b=768</sub></th>
    <th>97.07%</th>
    <th>18 568 450</th>
    <th>2 253</th>
    <th>8 258</th> 
  </tr>
  <!-- ResNeXt -->
  <tr>
    <th colspan="7">ResNeXt</th>
  </tr>
  <tr>
    <th rowspan="3"></th>
    <th colspan="2">ResNeXt35_16x4d <sub>mish</sub></th>
    <th>95.87% <sub>96.37%</sub></th>
    <th>3 554 762</th>
    <th>1 893 <sub>-1%</sub></th>
    <th>20 215 <sub>-1%</sub></th>
  </tr>
  <tr>
    <th colspan="2">ResNeXt50_16x4d <sub>mish</sub></th>
    <th>96.26% <sub>96.45%</sub></th>
    <th>5 461 706</th>
    <th>1 436 <sub>-1%</sub></th>
    <th>15 064 <sub>-1%</sub></th>
  </tr>
  <tr>
    <th colspan="2">ResNeXt101_16x4d <sub>mish</sub></th>
    <th>96.39% <sub>96.74%</sub></th>
    <th>10 614 474</th>
    <th>990 <sub>-1%</sub></th>
    <th>11 063 <sub>-2%</sub></th> 
  </tr>
  <!-- DenseNet -->
  <tr>
    <th colspan="7">DenseNet</th>
  </tr>
  <tr> 
    <th rowspan="4"></th>
    <th colspan="2">DenseNet52k12</th>
    <th>93.75%</th>
    <th>272 398</th>
    <th>7 209</th>
    <th>31 956</th>
  </tr>
  <tr>
    <th colspan="2">DenseNet100k12</th>
    <th>95.4%</th>
    <th>793 150</th>
    <th>2 734</th>
    <th>12 119</th>
  </tr>
  <tr>
    <th colspan="2">DenseNet100k16</th>
    <th>95.87%</th>
    <th>1 386 906</th> 
    <th>2 394</th>
    <th>11 114</th>
  </tr>
  <tr>
    <th colspan="2">DenseNet160k12<sub> b=512</sub></th>
    <th>96.43%</th>
    <th>1 795 090</th>
    <th>1 212</th>
    <th>4 860</th>
  </tr>
</table>


* Cos = Cosine Learning Rate Decay schedule. </br>
  From [Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
* Mish = Self regularized non-monotonic activation function, f(x) = x*tanh(softplus(x)). </br>
  From [Mish: A Self Regularized Non-Monotonic Activation Function](https://arxiv.org/abs/1908.08681)

