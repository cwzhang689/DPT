# DPT
A Diffusion and Transformer method in Multimodal Emotion Recognition

# Introduction

# Usage
##  Enviroment
Python = 3.7.13
Pytorch = 1.10.0
CUDA = 11.3

## Datasets
We use three vision and language datasets: . Please download the datasets by yourself. We use pyarrow to serialize the datasets, the conversion codes are located in vilt/utils/wirte_*.py. Please see DATA.md to organize the datasets, otherwise you may need to revise the write_*.py files to meet your dataset path and files. Run the following script to create the pyarrow binary file:

## Evaluation

## Train

# Citation
If you find this work useful for your research, please cite:
@inproceedings{,
 title = {},
 author = {},
 booktitle = {},
 year = {2025}
}

# Contact
If you have any questions, please create an issue on this repository or contact us at zhangchuwen2024@163.com or 2072750036@qq.com.

# Acknowledgements
Our code is based on the backbone MPLMM(https://github.com/zrguo/MPLMM) repository. Thanks for releasing their code. If you use our model and code, please consider citing this work as well.
