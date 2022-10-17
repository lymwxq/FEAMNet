# FEAMNet

We use the 4D HCI light field dataset, which is synthetic and composed of 28 light field scenes. And we use 16 scenes of "additional" for the training of the proposed FEAMNet. What's more, the above HCI dataset is available at [HCI](https://pan.baidu.com/s/1MS0B99scmYo8KNmh39prtQ), whose password is fr3a.

The FEAMNet is compared with other five advanced methods, including EPINET(CVPR2018), MANet(ICASSP2020), LFattNet(AAAI2020), FastLFNet(ICCV2021) and DistgDisp(TPAMI2022) and the visual comparison results can be checked at [Visual Results](https://github.com/lymwxq/FEAMNet/tree/master/ComparisonResults).

In our implementation, TensorFlow acts as the backend to construct our proposed network on an NVIDIA GTX 1080Ti GPU. Meanwhile, the 32*32 patches are applied, which are randomly extracted from the 4D light field images among all the training samples. Besides, the textureless and specular reflection patches are excluded to avoid the problem of ambiguous disparity estimation. As for the training parameters, the batch size is 12, and the learning rate is 0.001.
