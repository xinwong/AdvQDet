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
