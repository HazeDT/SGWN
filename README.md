# Spectral Graph Wavelet Networks(SGWN)
This code is about the implementation of [Filter-Informed Spectral Graph Wavelet Networks for Multiscale Feature Extraction and Intelligent Fault Diagnosis](https://ieeexplore.ieee.org/abstract/document/10079151).

![SGWN](https://github.com/HazeDT/SGWN/blob/main/SGWN.jpg)
![SGWConv](https://github.com/HazeDT/SGWN/blob/main/SGWConv.jpg)

# Note
In SGWN, the spectral graph wavelet convolutional (SGWConv) layer is established upon the spectral graph wavelet transform, which can decompose a graph signal into scaling function coefficients and spectral graph wavelet coefficients. With the help of SGWConv, SGWN is able to prevent the over-smoothing problem caused by long-range low-pass filtering, by simultaneously extracting low-pass and band-pass features. Furthermore, to speed up the computation of SGWN, the scaling kernel function and graph wavelet kernel function in SGWConv are approximated by the Chebyshev polynomials. .


# Implementation
python ./SGWM/train_graph.py --model_name SGWN  --checkpoint_dir ./results/   --data_name XJTUSpurgearKnn --data_dir ./data/XJTUSpurgearKnn --per_node 10 --s 2 --n 2 


# Citation

SGWN:
@ARTICLE{10079151,
  author={Li, Tianfu and Sun, Chuang and Fink, Olga and Yang, Yuangui and Chen, Xuefeng and Yan, Ruqiang},
  journal={IEEE Transactions on Cybernetics}, 
  title={Filter-Informed Spectral Graph Wavelet Networks for Multiscale Feature Extraction and Intelligent Fault Diagnosis}, 
  year={2024},
  volume={54},
  number={1},
  pages={506-518},
  keywords={Feature extraction;Fault diagnosis;Wavelet transforms;Convolution;Band-pass filters;Kernel;Mathematical models;Graph neural networks (GNNs);intelligent fault diagnosis;interpretable;multiscale feature extraction},
  doi={10.1109/TCYB.2023.3256080}}


