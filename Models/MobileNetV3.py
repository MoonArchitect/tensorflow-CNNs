import tensorflow as tf
import tensorflow.keras as nn
from .SeNet import SEBlock
from ..Layers import get_activation_layer

""" 
    Implementation of MobileNetV3 for CIFAR/SVHN/32x32

    From: Searching for MobileNetV3, https://arxiv.org/abs/1905.02244 
    By: Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam
"""
