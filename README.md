Compression of Deep Convolutional Neural Networks Using Effective Channel Pruning
======

Abstract：Pruning is a promising technology for convolutional neural networks (CNNs) to address the problems of high computational complexity and high memory requirement. However, there is a principal challenge in channel pruning. Although the least important feature-map is removed each time based on one pruning criterion, these pruning may produce considerable fluctuation in classification performance, which easily results in failing to restore its capacity. We propose an effective channel pruning criterion to reduce redundant parameters, while significantly reducing such fluctuations. This criterion adopts the loss-approximating Taylor expansion based on not the pruned parameters but the parameters in the subsequent convolutional layer, which differentiates our method from existing methods, to evaluate the importance of each channel. To improve the learning effectivity and efficiency, the importance of these channels is ranked using a small proportion of training dataset. Furthermore, after each least important channel is pruned, a small fraction of training dataset is used to fine-tune the pruned network to partially recover its accuracy. Periodically, more proportion of training dataset is used for the intensive recovery in accuracy. The proposed criterion significantly addresses the aforementioned problems and shows outstanding performance compared to other criteria, such as Random, APoZ and Taylor pruning criteria. The experimental results demonstrate the excellent compactness performances of our approach, using several public image classification datasets, on some popular deep network architectures. Our code is available at: \url{https://github.com/QingbeiGuo/Based-Taylor-Pruning.git}.

## Citation
@inproceedings{GuoWZ19,

  title={Compression of deep convolutional neural networks using effective channel pruning},

  author={Guo, Qingbei and Wu, Xiao-Jun and Zhao, Xiuyang},

  booktitle={Proceedings of the 10th International Conference on Image and Graphics (ICIG 2019)},

  pages={760--772},

  year={2019}
}

Q. Guo, X.-J. Wu, and X. Zhao, “Compression of deep convolutional neural networks using effective channel pruning,” in Proceedings of the 10th International Conference on Image and Graphics (ICIG 2019), 2019, pp. 760–772.
