# UNP
This repository contains the source code for our paper " *Understanding Negative Proposals in Generic Few-Shot Object Detection* " by Bowei Yan, Chunbo Lang, Gong Cheng and Junwei Han.

**abstract:** *Recently, Few-Shot Object Detection (FSOD) has received considerable research attention as a strategy for reducing reliance on extensively labeled bounding boxes. However, current approaches encounter significant challenges due to the intrinsic issue of incomplete annotation while building the instance-level training benchmark. In such cases, the instances with missing annotations are regarded as background, resulting in erroneous training gradients back-propagated through the detector, thereby compromising the detection performance. To mitigate this challenge, we introduce a simple and highly efficient method that can be plugged into both meta-learning-based and transfer-learning-based methods. Our method incorporates two innovative components: Confusing Proposals Separation (CPS) and Affinity-Driven Gradient Relaxation (ADGR). Specifically, CPS effectively isolates confusing negatives while ensuring the contribution of hard negatives during model fine-tuning; ADGR then adjusts their gradients based on the affinity to different category prototypes. As a result, false-negative samples are assigned lower weights than other negatives, alleviating their harmful impacts on the few-shot detector without the requirement of additional learnable parameters. Extensive experiments conducted on the PASCAL VOC and MS-COCO datasets consistently demonstrate that our method significantly outperforms both the baseline and recent FSOD methods. Furthermore, its versatility and efficiency suggest the potential to become a stronger new baseline in the field of FSOD.*


![Image text](https://github.com/Ybowei/UNP/blob/main/picture/method.jpg)



