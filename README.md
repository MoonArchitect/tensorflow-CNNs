# Deep Convolutional Neural Networks for Image Classification

Implementation of various AI papers for image classification  
\---- Section is under construction ----

### Implemented and Planned Features:
<details>
  <summary> Model Architectures </summary>
  
- [x] ResNetV2
- [x] ResNetV2 + Stochastic Depth
- [x] ResNeXt
- [x] ResNeXt + Stochastic Depth
- [x] DenseNet (Currently Reworking)
- [x] MobileNetV2
- [ ] MobileNetV3
- [ ] SeNet
- [ ] ResNeSt
- [ ] EfficientNet
- [ ] NAT
- [ ] TResNet
- [ ] PyramidNet
- [ ] Xception
- [ ] IBN-Net

</details>

<details>
  <summary> Other Features </summary>
  
- [x] Step Learning Rate (LR) decay schedule
- [x] HTD (Hyperbolic-Tangent LR Decay schedule)
- [x] Cosine LR decay schedule
- [x] Cutout
- [x] Mixup
- [x] Cutmix
- [ ] Hard and Soft PatchUp
- [x] Mish
- [ ] Swish
- [ ] EvoNorm

</details>

## CIFAR10 Results
<table>
  <tr>
    <th colspan="3">⠀⠀⠀⠀⠀⠀⠀⠀⠀Model⠀⠀⠀⠀⠀⠀⠀⠀⠀</th>
    <th>Top1</br>⠀⠀Accuracy⠀⠀</th>
    <th>Top1</br>⠀⠀Error⠀⠀</th>
    <th>Param count</th>
    <th>FLOPs/2</th>
    <th>Training speed</br>(imgs/sec)</th>
  </tr>
  <tr>
    <th colspan="3">ResNeSt</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th colspan="3" style="border-top:5px">EfficientNet</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th colspan="3">Xception</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th colspan="3">IBN-Net</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th colspan="3"></br>TResNet</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th rowspan="5"></th>
    <th colspan="2">TResNetM-.5-32px<sub>+HTD+Cutmix</sub></th>
    <th>95.05%</th>
    <th>4.95%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">TResNetM-.5-192px<sub>+HTD+Cutmix</sub></th>
    <th>96.10%</th>
    <th>3.90%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">TResNetM-.75-160px<sub>+Cos+Cutmix</sub></th>
    <th>96.92%</th>
    <th>3.08%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">TResNetM-128px<sub>+Cos+Cutmix</sub></th>
    <th>96.51%</th>
    <th>3.49%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">TResNetL-.5-192px<sub>+HTD+Cutmix</sub></th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="3"></br>MobileNetV3</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th rowspan="9"></th>
    <th colspan="2">MNetV3S 160px 1.5<sub>+Cos+Cutmix</sub></th>
    <th>94.25%</th>
    <th>5.75%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MNetV3S 192px<sub>+Cos+Cutmix</sub></th>
    <th>94.49%</th>
    <th>5.51%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MNetV3S 192px 1.5<sub>+Cos+Cutmix</sub></th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MNetV3S 192px 2<sub>+Cos+Cutmix</sub></th>
    <th>95.63%</th>
    <th>4.37%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MNetV3S 224px<sub>+Cos+Cutmix</sub></th>
    <th>95.50%</th>
    <th>4.50%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MNetV3S 224px 2<sub>+Cos+Cutmix</sub></th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MNetV3L<sub>+HTD+Cutmix</sub></th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MNetV3L 192px 2<sub>+HTD+Cutmix</sub></th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MNetV3L_2<sub>+HTD+Cutmix</sub></th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="3"></br>MobileNetV2</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th rowspan="2"></th>
    <th colspan="2">MobileNetV2 96px<sub>+HTD+Cutmix</sub></th>
    <th>95.17%</th>
    <th>4.83%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">MobileNetV2 192px<sub>+HTD+Cutmix</sub></th>
    <th>96.23%</th>
    <th>3.77%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="3"></br>ResNetV2</th>
    <th> </th>
    <th> </th>
    <th> </th>
    <th> </th>
    <th> </th>
  </tr>
  <tr>
    <th rowspan="25"></th>
    <th colspan="2">ResNet34</th>
    <th>92.64%</th>
    <th>7.36%</th>
    <th rowspan="2">1,327,226</th>
    <th rowspan="2">72.40M</th>
    <th rowspan="2">4160</th>
  </tr>
  <tr>
    <th rowspan="1"></th>
    <th>+HTD +Cutmix</th>
    <th>94.59%</th>
    <th>5.41%</th>
  </tr>
  <tr>
    <th colspan="2">ResNet35b</th>
    <th>92.83%</th>
    <th>7.17%</th>
    <th rowspan="3">873,722</th>
    <th>51.83M</th>
    <th>3012</th>
  </tr>
  <tr>
    <th rowspan="2"></th>
    <th>+Mish</th>
    <th>93.50%</th>
    <th>6.50%</th>
    <th>52.23M</th>
    <th>2065</th>
  </tr>
  <tr>
    <th>+HTD +Cutmix</th>
    <th>95.06%</th>
    <th>4.96%</th>
    <th>51.83M</th>
    <th>3012</th>
  </tr>
  <tr>
    <th colspan="2">ResNet50b</th>
    <th>93.18%</th>
    <th>6.82%</th>
    <th rowspan="2">1,309,210</th>
    <th rowspan="2">74.11M</th>
    <th rowspan="2">2171</th>
  </tr>
  <tr>
    <th rowspan="12"></th>
    <th>+HTD</th>
    <th>93.65%</th>
    <th>6.35%</th>
  </tr>
  <tr>
    <th>+SD</th>
    <th>94.05%</th>
    <th>5.95%</th>
    <th rowspan="5">1,309,210</th>
    <th rowspan="5">74.11M</th>
    <th rowspan="5">2247</th>
  </tr>
  <tr>
    <th>+SD +HTD +Cutout16</th>
    <th>94.96%</th>
    <th>5.04%</th>
  </tr>
  <tr>
    <th>+SD +HTD +Mixup0.2</th>
    <th>95.42%</th>
    <th>4.58%</th>
  </tr>
  <tr>
    <th>+SD +HTD +Mixup1</th>
    <th>95.48%</th>
    <th>4.52%</th>
  </tr>
  <tr>
    <th>+SD +HTD +Cutmix</th>
    <th>95.56%</th>
    <th>4.44%</th>
  </tr>
  <tr>
    <th>+Mish</th>
    <th>93.94%</th>
    <th>6.06%</th>
    <th rowspan="1">1,309,210</th>
    <th rowspan="1">74.77M</th>
    <th>1478</th>
  </tr>
  <tr>
    <th>+SD +Mish</th>
    <th>94.42%</th>
    <th>5.58%</th>
    <th rowspan="4">1,309,210</th>
    <th rowspan="4">74.77M</th>
    <th rowspan="4">1544</th>
  </tr>
  <tr>
    <th>+SD +HTD +Mish</th>
    <th>94.50%</th>
    <th>5.50%</th>
  </tr>
  <tr>
    <th>+SD +Cos +Mish</th>
    <th>94.69%</th>
    <th>5.31%</th>
  </tr>
  <tr>
    <th>+SD +HTD +Cutout16 +Mish</th>
    <th>95.37%</th>
    <th>4.63%</th>
  </tr>
  <tr>
    <th>+SD +HTD +Cutmix +Mish</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">ResNet101b</th>
    <th>93.79%</th>
    <th>6.21%</th>
    <th>2,530,426</th>
    <th>149.87M</th>
    <th>1387</th>
  </tr>
  <tr>
    <th rowspan="2"></th>
    <th>+SD</th>
    <th>94.40%</th>
    <th>5.60%</th>
    <th>2,530,426</th>
    <th>149.87M</th>
    <th>1512</th>
  </tr>
  <tr>
    <th>+SD +Mish +HTD +Cutout16</th>
    <th>95.55%</th>
    <th>4.45%</th>
    <th>2,530,426</th>
    <th>151.21M</th>
    <th>1074</th>
  </tr>
  <tr>
    <th colspan="2">ResNet152b</th>
    <th>-</th>
    <th>-</th>
    <th>3,528,314</th>
    <th>225.63M</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th>+SD</th>
    <th>95.06%</th>
    <th>4.94%</th>
    <th>3,528,314</th>
    <th>225.63M</th>
    <th>1161</th>
  </tr>
  <tr>
    <th colspan="2">ResNet170b +Mish +HTD +Cutmix</th>
    <th>-</th>
    <th>-</th>
    <th>4,414,202</th>
    <th>252.37M</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">WideResNet18+Cutout+HTD</th>
    <th>94.91%</th>
    <th>5.09%</th>
    <th>11,205,578</th>
    <th>609.96M</th>
    <th>-</th>
  </tr>
  <!--- <<<<<<<<<<<<<<<<<<<<< ResNeXt >>>>>>>>>>>>>>>>>>>>> --->
  <tr>
    <th colspan="3"></br>ResNeXt</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th rowspan="6"></th>
    <th colspan="2">ResNetXt50C32</th>
    <th>94.07%</th>
    <th>5.93%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th>+SD</th>
    <th>94.49%</th>
    <th>5.51%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">ResNetXt101C32</th>
    <th>94.25%</th>
    <th>5.75%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th>+SD</th>
    <th>94.88%</th>
    <th>5.12%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">ResNetXt152C32</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th>+SD</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <!--- <<<<<<<<<<<<<<<<<<<<< SeNet >>>>>>>>>>>>>>>>>>>>> --->
  <tr>
    <th colspan="3"></br>SeNet</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr> 
    <th rowspan="6"></th>
    <th colspan="2">SeNet50</th>
    <th>93.40%</th>
    <th>6.60%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th>+SD</th>
    <th>94.25%</th>
    <th>5.75%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">SeNet101</th>
    <th>94.30%</br>94.79%*</th>
    <th>5.70%</br>5.21%*</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th>+SD</th>
    <th>94.65%</th>
    <th>5.35%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">SeNet152</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th>+SD</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <!--- <<<<<<<<<<<<<<<<<<<<< DenseNet >>>>>>>>>>>>>>>>>>>>> --->
  <tr>
    <th colspan="3"></br>DenseNet</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th rowspan="6"></th>
    <th colspan="2">DenseNet100k12</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">DenseNet100k16</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">DenseNet160k12</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">DenseNet250k12</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">DenseNet250k24</th>
    <th>96.38*</th>
    <th>3.62*</th>
    <th>15.3M</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">DenseNet190k40</th>
    <th>96.54*</th>
    <th>3.46*</th>
    <th>25.6M</th>
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

<!--- colspan="2" rowspan="2" ---> 
