# 3D Spatial Understanding in MLLMs (ICRA 2025)

<a href='https://birdy666.github.io/projects/3d_spatial_understanding_in_mllms/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='[https://arxiv.org/pdf/2309.05519](https://arxiv.org/pdf/2412.06613)'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a>
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/watch?v=Aq9gmHn-op4)

[Chun-Peng Chang](https://chunpeng-chang.github.io/), [Alain Pagani](https://av.dfki.de/members/pagani/), [Didier Stricker](https://av.dfki.de/members/stricker/)

**[DFKI AV](https://av.dfki.de/), German Research Center for Artificial Intelligence, Augmented Vision**

![teaser](assets/teaser.png)
## Evaluation with 3DVG
For the evaluation with 3D Visual Grounding model in the experiments, please refer to the following works:
* [ReferIt3D](https://github.com/referit3d/referit3d)
* [MVT](https://github.com/sega-hsj/MVT-3DVG)
* [MiKASA](https://birdy666.github.io/projects/mikasa/)

### Setup
We split the training scenes of Nr3D and Sr3D into two halves. The first half is used to train the model. From the second half, we generate approximately 65K spatially-grounded instructions—comparable in size to the Sr3D training set—which are then used to train our 3D Visual Grounding (3DVG) models. This split ensures that the VLM and 3DVG models are trained on disjoint scenes to avoid data leakage.

During instruction generation, we randomly assign target objects from each scene, ensuring that each selected target has at least one distractor (i.e., another object of the same category). We also randomly sample three potential anchors (reference objects) per scene, which the model must use to disambiguate the target.

## Data Preparation
### ScanNet Data
To download the ScanNet scans, see [ScanNet](https://github.com/ScanNet/ScanNet#scannet-data) for the instruction.
To preprocess the data required for Referit3D challenge, visit [ReferIt3D](https://github.com/referit3d/referit3d).

### Referit3D Linguistic Data (Nr3D/Sr3D/Sr3D+)
 See [ReferIt3D](https://github.com/referit3d/referit3d) for more details.
* [**Nr3D**](https://drive.google.com/file/d/1qswKclq4BlnHSGMSgzLmUu8iqdUXD8ZC/view?usp=sharing) (10.7MB)
* [**Sr3D**](https://drive.google.com/drive/folders/1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV?usp=sharing) (19MB)
* [**Sr3D+**](https://drive.google.com/drive/folders/1DS4uQq7fCmbJHeE-rEbO8G1-XatGEqNV?usp=sharing) (20MB)

## Environment
All experiments were conducted using a single A100-80GB GPU.
* Ubuntu: 20.04
* CUDA: 12.1
* PyTorch: 2.1.2
* python: 3.8
* torchvision: 0.14.1
* torchaudio: 0.13.1
* pytorch-cuda: 11.6


## Installation
For the dependencies please refer requirements.txt
* To use a PointNet++ visual-encoder you need to compile its CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413):
```Note: To do this compilation also need: gcc5.4 or later.```
```Console
    cd external_tools/pointnet2
    python setup.py install
```

## Training
* Vision Encoder: Pretrained PointNet++ encoder from [MiKASA](https://birdy666.github.io/projects/mikasa/)

* LLM Base Model: [Vicuna-7b](https://huggingface.co/lmsys/vicuna-7b-delta-v0)
![pipeline](assets/training_pipeline.png)
### Stage 1: Vision-Language Alignment
In Stage 1, we use a pretrained PointNet++ encoder from [MiKASA](https://birdy666.github.io/projects/mikasa/) to extract object-level features from 3D point clouds. The goal of this stage is to align these visual features with corresponding category name embeddings produced by a LLM. This mapping establishes a shared embedding space between 3D geometry and language semantics. 

### Stage 2: Instruction Grounding with LoRA
In Stage 2, we fine-tune the model to perform 3D visual grounding based on generated instructions. All parameters are frozen except for the LoRA (Low-Rank Adaptation) modules, which are trained to adapt the pretrained backbone to the grounding task. This enables efficient fine-tuning while preserving the pretrained knowledge. For more details, refer to our paper.

## Citation
```
@article{chang20243d,
      title={3D Spatial Understanding in MLLMs: Disambiguation and Evaluation},
      author={Chang, Chun-Peng and Pagani, Alain and Stricker, Didier},
      journal={arXiv preprint arXiv:2412.06613},
      year={2024}
    }
```

## Credit
The project is built based on the following repository:
* [NExT-GPT: Any-to-Any Multimodal LLM](https://next-gpt.github.io/)
* [MiKASA](https://birdy666.github.io/projects/mikasa/) 
