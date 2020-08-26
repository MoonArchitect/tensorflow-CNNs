# Models
## CIFAR10 Results
<table>
  <tr>
    <th colspan="3">⠀⠀⠀⠀⠀⠀⠀⠀⠀Model⠀⠀⠀⠀⠀⠀⠀⠀⠀</th>
    <th>Top1 Accuracy</th>
    <th>Top1 Error</th>
    <th>Param count</th>
    <th>MFLOPs</br>(fwd. path)</th>
    <th>Training Latency</br>(ms.)</th>
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
    <th colspan="3">InceptionV4</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
  </tr>
  <tr>
    <th colspan="3">InceptionV3</th>
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
    <th colspan="3">ResNetV2</th>
    <th> </th>
    <th> </th>
    <th> </th>
    <th> </th>
    <th> </th>
  </tr>
  <tr>
    <th rowspan="21"></th>
    <th colspan="2">ResNet34</th>
    <th>92.64%</th>
    <th>7.36%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">ResNet35b</th>
    <th>92.83%</th>
    <th>7.17%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th>⠀⠀+Mish⠀⠀</th>
    <th>93.50%</th>
    <th>6.50%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">ResNet50b</th>
    <th>93.18%</th>
    <th>6.82%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th rowspan="11"></th>
    <th>+Mish</th>
    <th>93.94%</th>
    <th>6.06%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th>+HTD</th>
    <th>93.65%</th>
    <th>6.35%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th>+SD</th>
    <th>94.05%</th>
    <th>5.95%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th>+SD +Mish</th>
    <th>94.42%</th>
    <th>5.58%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th>+SD +HTD +Mish</th>
    <th>94.50%</th>
    <th>5.50%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th>+SD +Cosine +Mish</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th>+SD +HTD +Cutout16</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th>+SD +HTD +Mixup0.2</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th>+SD +HTD +Mixup1</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th>+SD +HTD +Cutmix</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th>+SD +Mish +HTD +Cutout16</th>
    <th>95.37%</th>
    <th>4.63%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">ResNet101b</th>
    <th>93.79%</th>
    <th>6.21%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th rowspan="2"></th>
    <th>+SD</th>
    <th>94.40%</th>
    <th>5.60%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th>+SD +Mish +HTD +Cutout16</th>
    <th>95.55%</th>
    <th>4.45%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">ResNet152b</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th></th>
    <th>+SD</th>
    <th>95.06%</th>
    <th>4.94%</th>
    <th>-</th>
    <th>-</th>
    <th>-</th>
  </tr>
  <tr>
    <th colspan="2">WideResNet18+Cutout+HTD</th>
    <th>94.91%</th>
    <th>5.09%</th>
    <th>-</th>
    <th>-</th>
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
    <th>94.30%|94.79%*</th>
    <th>5.70%|5.21%*</th>
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

* SD = Stochastic Depth. From [this paper](https://arxiv.org/abs/1603.09382)
* HTD = Hyperbolic-Tangent Learning Rate Decay schedule. From [this paper](https://arxiv.org/abs/1806.01593)
* Mish = Self regularized non-monotonic activation function, f(x) = x*tanh(softplus(x)). From [this paper](https://arxiv.org/abs/1908.08681)

<!--- colspan="2" rowspan="2" ---> 
