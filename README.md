# How-to-train-your-draGAN: A task oriented solution to imbalanced classification

[![arXiv](https://img.shields.io/badge/arXiv-2211.10065-b31b1b.svg)](https://arxiv.org/abs/2211.10065)

 The long-standing challenge of building effective classification models for small and imbalanced datasets has seen little improvement since the creation of the Synthetic Minority Over-sampling Technique (SMOTE) over 20 years ago. Though GAN based models seem promising, there has been a lack of purpose built architectures for solving the aforementioned problem, as most previous studies focus on applying already existing models. This paper proposes a unique, performance-oriented, data-generating strategy that utilizes a new architecture, coined draGAN, to generate both minority and majority samples. The samples are generated with the objective of optimizing the classification model's performance, rather than similarity to the real data. We benchmark our approach against state-of-the-art methods from the SMOTE family and competitive GAN based approaches on 94 tabular datasets with varying degrees of imbalance and linearity. Empirically we show the superiority of draGAN, but also highlight some of its shortcomings.


<img src="https://i.imgur.com/HWLOB8e.png" align="right" width="350"/>


### Python environment setup with Conda
```bash
conda create -n draGAN python=3.7 anaconda
conda activate draGAN

conda install pytorch=1.13 torchvision torchaudio -c pytorch -c nvidia
pip install imbalanced_databases

conda clean --all
```

### Example
example.py contains code that showcases how to use draGAN.



