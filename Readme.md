# Content Selection Network for Document-grounded Retrieval-based Chatbots

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

#### News
- 2021-2-25: We upload all source code and data files!

## Abstract
This repository contains the source code and datasets for the ECIR 2021 paper [Content Selection Network for Document-grounded Retrieval-based Chatbots](https://arxiv.org/pdf/2101.08426.pdf) by Zhu et al. <br>

Grounding human-machine conversation on a document is an effective way to improve the performance of retrieval-based chatbots. However, only a part of the document content may be relevant to help select the appropriate response at a round. It is thus crucial to select the  part of document content relevant to the current conversation context. In this paper, we propose a document content selection network (CSN) to perform explicit selection of relevant document contents, and filter out the irrelevant parts. We show in experiments on two public document-grounded conversation datasets that CSN can effectively help select the relevant document contents to the conversation context, and it produces better results than the state-of-the-art approaches.

Authors: Yutao Zhu, Jian-Yun Nie, Kun Zhou, Pan Du, Zhicheng Dou

## Requirements
We test the code with the following packages. <br>
- Python 3.5 <br>
- PyTorch 1.3.1 (with GPU support)<br>

## Usage
1. Download the data from the [link](https://drive.google.com/drive/folders/1-lBPcEG1NfJa3CBfWgmk4r-W30dmuOoh?usp=sharing)
2. Unzip PersonaChat_data.zip and move all files into /PersonaChat/data/
3. Unzip CMUDoG_data.zip and move all files into /CMUDoG/data/

For PersonaChat: <br>
```
CUDA_VISIBLE_DEVICES=0 python3 run.py
```

For CMUDoG: <br>
```
CUDA_VISIBLE_DEVICES=0 python3 run.py
```

## Citations
If you use the code and datasets, please cite the following paper:  
```
@inproceedings{ZhuNZDD21,
  author    = {Yutao Zhu and
               Jian{-}Yun Nie and
               Kun Zhou and
               Pan Du and
               Zhicheng Dou
               },
  editor    = {Djoerd Hiemstra and
               Maria-Francine Moens and
               Josiane Mothe and
               Raffaele Perego and
               Martin Potthast and
               Fabrizio Sebastiani},
  title     = {Content Selection Network for Document-grounded Retrieval-based Chatbots},
  booktitle = {Advances in Information Retrieval - 43rd European Conference on {IR}
               Research, {ECIR} 2021, Lucca, Italy, March 28-April 1, 2021, Proceedings},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2021},
}
```
