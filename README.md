# stdthread-gpus

C++のstd::threadを用いたイントラノード・複数GPU計算のテストコード

の計画でしたが、NVIDIA的に非推奨であったため、
for文で回してcudaSetDeviceすることになりました。