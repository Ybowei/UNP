# UNP
This repository contains the source code for our paper " *Understanding Negative Proposals in Generic Few-Shot Object Detection* " by Bowei Yan, Chunbo Lang, Gong Cheng and Junwei Han.

**abstract:** *Recently, Few-Shot Object Detection (FSOD) has received considerable research attention as a strategy for reducing reliance on extensively labeled bounding boxes. However, current approaches encounter significant challenges due to the intrinsic issue of incomplete annotation while building the instance-level training benchmark. In such cases, the instances with missing annotations are regarded as background, resulting in erroneous training gradients back-propagated through the detector, thereby compromising the detection performance. To mitigate this challenge, we introduce a simple and highly efficient method that can be plugged into both meta-learning-based and transfer-learning-based methods. Our method incorporates two innovative components: Confusing Proposals Separation (CPS) and Affinity-Driven Gradient Relaxation (ADGR). Specifically, CPS effectively isolates confusing negatives while ensuring the contribution of hard negatives during model fine-tuning; ADGR then adjusts their gradients based on the affinity to different category prototypes. As a result, false-negative samples are assigned lower weights than other negatives, alleviating their harmful impacts on the few-shot detector without the requirement of additional learnable parameters. Extensive experiments conducted on the PASCAL VOC and MS-COCO datasets consistently demonstrate that our method significantly outperforms both the baseline and recent FSOD methods. Furthermore, its versatility and efficiency suggest the potential to become a stronger new baseline in the field of FSOD.*

---
![Image text](https://github.com/Ybowei/UNP/blob/main/picture/method.jpg)
---


## 📑 Table of Contents

* Understanding Negative Proposals in Generic Few-Shot Object Detection
  * Table of Contents
  * Installation
  * Code Structure
  * Data Preparation
  * Model training and evaluation on MSCOCO
    * Base training
    * Model Fine-tuning
    * Evaluation
  * Model training and evaluation on PASCL VOC
    * Base training
    * Model Fine-tuning
    * Evaluation
  * Model Zoo


## 🧩 Installation

Our code is based on [MMFewShot](https://github.com/open-mmlab/mmfewshot/tree/main) and please refer to [install.md](https://github.com/open-mmlab/mmfewshot/blob/main/docs/en/install.md) for installation of MMFewShot framwork. 
Please note that we used detectron 0.1.0 in this project. Higher versions of detectron might report errors.


## 🏰 Code Structure

* **configs:** Configuration files
* **checkpoints:** Checkpoint
* **Weights:** Pretraing models
* **Data:** Datasets for base training and finetuning
* **mmfewshot:** Model framework
* **Tools:** analysis and visualize tools

## 💾 Data Preparation

* Our model is evaluated on two FSOD benchmarks PASCAL VOC and MSCOCO following the previous work [TFA](https://github.com/ucbdrive/few-shot-object-detection).
* Please prepare the original PASCAL VOC and MSCOCO datasets and also the few-shot datasets in the folder ./data/coco and ./data/voc respectively.
* please refer to [PASCAL VOC](https://github.com/Ybowei/UNP/blob/main/data/voc/README.md) and [MSCOCO](https://github.com/Ybowei/UNP/blob/main/data/coco/README.md) for more detail.

## 📖 Model training and evaluation on MSCOCO

* We have two steps for model training, first training the model over base classes, and then fine-tuning the model over novel classes.
* The training script for base training is
  ```Python
  bash 

