# AdvQDet: Detecting Query-Based Adversarial Attacks with Adversarial Contrastive Prompt Tuning

Official PyTorch implementation of the following paper:

AdvQDet: Detecting Query-Based Adversarial Attacks with Adversarial Contrastive Prompt Tuning, ACM MM 2024.

--- 

<p align="center">
<img src="./asserts/fig1.png" width=100% height=100% 
class="center">
</p>

The core of detecting query-based attack, such as Boundary, HSJA, and NESS, is training a robust feature extractor that always produces similar feature vectors for any two adversarial queries crafted from the same image, even for adaptive attacks..

<p align="center">
<img src="./asserts/fig2.png" width=100% height=100% 
class="center">
</p>

In light of this, we propose a simple yet effective framework, Adversarial Contrastive Prompt Tuning (ACPT), to train reliable feature extractors for accurate and robust detection of query-based attacks.

# Environment Setup

To set up the required environment, please follow the installation instructions provided in the [ZSRobust4FoundationModel repository](https://github.com/cvlab-columbia/ZSRobust4FoundationModel)

# Data Setup

You will need to configure `utils/datasets.py` to point to your dataset directory. The expected dataset structure is as follows (using CIFAR10 as an example):

```
cifar10/
    - imgs/
        - 0.png
        - 1.png
        - 2.png
        ...
    - cifar10.json
    - cifar10_targeted.json
```

`cifar10.json`:  A JSON file mapping images to their corresponding labels. Example:

```
{
    "imgs/0.png": 3,
    "imgs/1.png": 8,
    "imgs/2.png": 8,
    ...
}
```

`cifar10_targeted.json`: A JSON file used for targeted attacks. It maps target labels to an initialization image for that label.


# Stateful Detection
   
This project provides scripts for performing stateful detection. You can find all relevant scripts in the `main.sh` directory.

    ```
    main.sh
    ```

# Acknowledgement

This repository is built upon [`ZSRobust4FoundationModel`](https://github.com/cvlab-columbia/ZSRobust4FoundationModel) and [`OARS`](https://github.com/nmangaokar/ccs_23_oars_stateful_attacks). Thanks for those well-organized codebases.

# Citation

```
@inproceedings{wang2024advqdet,
  title={AdvQDet: Detecting Query-Based Adversarial Attacks with Adversarial Contrastive Prompt Tuning},
  author={Wang, Xin and Chen, Kai and Ma, Xingjun and Chen, Zhineng and Chen, Jingjing and Jiang, Yu-Gang},
  booktitle={ACM MM},
  year={2024}
}
```
