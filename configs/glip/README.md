# GLIP: Grounded Language-Image Pre-training

> [GLIP: Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857)

<!-- [ALGORITHM] -->

## Abstract

This paper presents a grounded language-image pre-training (GLIP) model for learning object-level, language-aware, and semantic-rich visual representations. GLIP unifies object detection and phrase grounding for pre-training. The unification brings two benefits: 1) it allows GLIP to learn from both detection and grounding data to improve both tasks and bootstrap a good grounding model; 2) GLIP can leverage massive image-text pairs by generating grounding boxes in a self-training fashion, making the learned representation semantic-rich. In our experiments, we pre-train GLIP on 27M grounding data, including 3M human-annotated and 24M web-crawled image-text pairs. The learned representations demonstrate strong zero-shot and few-shot transferability to various object-level recognition tasks. 1) When directly evaluated on COCO and LVIS (without seeing any images in COCO during pre-training), GLIP achieves 49.8 AP and 26.9 AP, respectively, surpassing many supervised baselines. 2) After fine-tuned on COCO, GLIP achieves 60.8 AP on val and 61.5 AP on test-dev, surpassing prior SoTA. 3) When transferred to 13 downstream object detection tasks, a 1-shot GLIP rivals with a fully-supervised Dynamic Head.

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/b87228d7-f000-4a5d-b103-fe535984417a"/>
</div>

## Installation

```shell
cd $MMDETROOT

# source installation
pip install -r requirements/multimodal.txt

# or mim installation
mim install mmdet[multimodal]
```

```shell
cd $MMDETROOT

wget https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth

python demo/image_demo.py demo/demo.jpg \
configs/glip/glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py \
--weights glip_tiny_a_mmdet-b3654169.pth \
--texts 'bench. car'
```

<div align=center>
<img src="https://github.com/open-mmlab/mmdetection/assets/17425982/7b450d96-81ac-462a-92bc-0d4ae7b8721c" width="40%"/>
</div>

## Results and Models

|   Model    | Zero-shot or Funetune | COCO mAP |       Pre-Train Data       |                                 Config                                  |                                                                                                                                                                                                   Download                                                                                                                                                                                                    |
| :--------: | :-------------------: | :------: | :------------------------: | :---------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| GLIP-T (A) |       Zero-shot       |   43.0   |            O365            |       [config](glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py)        |                                                                                                                                                         [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_a_mmdet-b3654169.pth)                                                                                                                                                          |
| GLIP-T (A) |       Funetune        |   53.1   |            O365            |   [config](glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)   | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230908_091856-39f01d03.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_a_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230908_091856.log.json) |
| GLIP-T (B) |       Zero-shot       |   44.9   |            O365            |       [config](glip_atss_swin-t_b_fpn_dyhead_pretrain_obj365.py)        |                                                                                                                                                         [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_b_mmdet-6dfbd102.pth)                                                                                                                                                          |
| GLIP-T (B) |       Funetune        |   54.1   |            O365            |   [config](glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)   | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230909_175354-e0c0c6d7.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_b_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230909_175354.log.json) |
| GLIP-T (C) |       Zero-shot       |   46.7   |         O365,GoldG         |    [config](glip_atss_swin-t_c_fpn_dyhead_pretrain_obj365-goldg.py)     |                                                                                                                                                         [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_c_mmdet-2fc427dd.pth)                                                                                                                                                          |
| GLIP-T (C) |       Funetune        |   55.2   |         O365,GoldG         |   [config](glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)   | [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230909_175543-5fcb4b97.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_c_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230909_175543.log.json) |
|   GLIP-T   |       Zero-shot       |   46.4   |    O365,GoldG,CC3M,SBU     | [config](glip_atss_swin-t_fpn_dyhead_pretrain_obj365-goldg-cc3m-sub.py) |                                                                                                                                                          [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_tiny_mmdet-c24ce662.pth)                                                                                                                                                           |
|   GLIP-T   |       Funetune        |   55.2   |    O365,GoldG,CC3M,SBU     |    [config](glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)    |     [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230910_125111-ad1025a0.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-t_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230910_125111.log.json)     |
|   GLIP-L   |       Zero-shot       |   51.3   | FourODs,GoldG,CC3M+12M,SBU |       [config](glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py)       |                                                                                                                                                            [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_l_mmdet-abfe026b.pth)                                                                                                                                                            |
|   GLIP-L   |       Funetune        |   59.4   | FourODs,GoldG,CC3M+12M,SBU |    [config](glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco.py)    |     [model](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230910_100800-e9be4274.pth)\| [log](https://download.openmmlab.com/mmdetection/v3.0/glip/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230910_100800.log.json)     |

Note:

1. The weights corresponding to the zero-shot model are adopted from the official weights and converted using the [script](../../tools/model_converters/glip_to_mmdet.py). We have not retrained the model for the time being.
2. Funetune refers to fine-tuning on the COCO 2017 dataset. The L model is trained using 16 A100 GPUs, while the remaining models are trained using 16 NVIDIA GeForce 3090 GPUs.
3. Taking the GLIP-T(A) model as an example, I trained it twice using the official code, and the fine-tuning mAP were 52.5 and 52.6. Therefore, the mAP we achieved in our reproduction is higher than the official results. The main reason is that we modified the `weight_decay` parameter.
4. Our experiments revealed that training for 24 epochs leads to overfitting. Therefore, we chose the best-performing model. If users want to train on a custom dataset, it is advisable to shorten the number of epochs and save the best-performing model.
5. Due to the official absence of fine-tuning hyperparameters for the GLIP-L model, we have not yet reproduced the official accuracy. I have found that overfitting can also occur, so it may be necessary to consider custom modifications to data augmentation and model enhancement. Given the high cost of training, we have not conducted any research on this matter at the moment.
6. We noticed that there is a discrepancy between the performance evaluation of the checkpoint and the evaluation logs during training. This is because the buffers of different ranks are not the same during training, but we only saved the weights of rank 0. If you want to avoid this issue, you can add the parameter `broadcast_buffers=True` in the configuration.