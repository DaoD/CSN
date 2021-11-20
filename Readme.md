# Content Selection Network for Document-grounded Retrieval-based Chatbots

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)

#### News
- 2021-3-18: We update the missing data file.
- 2021-2-25: We upload all source code and data files!
- 2021-11-20: We upload a new implementation of our method. It can achieve better performance!

## Abstract
This repository contains the source code and datasets for the ECIR 2021 paper [Content Selection Network for Document-grounded Retrieval-based Chatbots](https://arxiv.org/pdf/2101.08426.pdf) by Zhu et al. <br>

Grounding human-machine conversation on a document is an effective way to improve the performance of retrieval-based chatbots. However, only a part of the document content may be relevant to help select the appropriate response at a round. It is thus crucial to select the  part of document content relevant to the current conversation context. In this paper, we propose a document content selection network (CSN) to perform explicit selection of relevant document contents, and filter out the irrelevant parts. We show in experiments on two public document-grounded conversation datasets that CSN can effectively help select the relevant document contents to the conversation context, and it produces better results than the state-of-the-art approaches.

Authors: Yutao Zhu, Jian-Yun Nie, Kun Zhou, Pan Du, Zhicheng Dou

## Requirements
We test the code with the following packages. <br>
- Python 3.5 <br>
- PyTorch 1.3.1 (with GPU support)<br>

## Usage - New
1. Download the new data from the [link](https://drive.google.com/file/d/1kKh-1npyjaW6PvrBhILskBVcDuDhE66x/view?usp=sharing)
2. Unzip the data.zip into /Updated/data/

For PersonaChat: <br>
```
cd Updated
python3 runCSN.py --task personachat --file_suffix self_original
python3 runCSN.py --task personachat --file_suffix self_revised
```

For CMUDoG: <br>
```
cd Updated
python3 runCSN.py --task cmudog --file_suffix self_original_fullSection
```

## Results (CSN-word) - New
| Dataset              | R@1  | R@2  | R@5  | MRR  | 
| -------------------- | ---- | ---- | ---- | ---- |
| PersonaChat Original | 78.6 | 89.5 | 97.3 | 86.6 | 
| PersonaChat Revised  | 71.2 | 84.6 | 95.5 | 81.6 |
| CMUDoG               | 78.7 | 89.3 | 97.1 | 86.6 | 

## Usage - Old Version
1. Download the data from the [link](https://drive.google.com/drive/folders/1-lBPcEG1NfJa3CBfWgmk4r-W30dmuOoh?usp=sharing)
2. Unzip PersonaChat_data.zip and move all files into /PersonaChat/data/
3. Unzip CMUDoG_data.zip and move all files into /CMUDoG/data/

For PersonaChat: <br>
```
CUDA_VISIBLE_DEVICES=0 python3 run.py --task both_original
CUDA_VISIBLE_DEVICES=0 python3 run.py --task both_revised
```

For CMUDoG: <br>
```
CUDA_VISIBLE_DEVICES=0 python3 run.py
```

Parameters:
```
--level, "word" (default)/"sentence", the selection level
--is_training, True (default)/False, train or test the model
--batch_size, 15 (default for PersonaChat), 80 (default for CMUDoG)
--gru_hidden, 300 (defult), the hidden size of RNN
--emb_size, 400 (default for PersonaChat), 300 (default for CMUDoG), the embedding size
--learning_rate, 1e-3 (defult), the learning rate
--gamma, 0.3 (default), the filter threshold 
--decay, 0.9 (default), the decay factor
--epochs, 5 (default for PersonaChat), 8 (default for CMUDoG), the number of training epochs
--save_path, "./checkpoint/" (default), the path to save model
--score_file_path, "score_file.txt" (default), the path to save results
--log_path, "./log/" (default), the path to save log 
```


## Citations
If you use the code and datasets, please cite the following paper:  
```
@inproceedings{ZhuNZDD21,
  author    = {Yutao Zhu and
               Jian{-}Yun Nie and
               Kun Zhou and
               Pan Du and
               Zhicheng Dou},
  editor    = {Djoerd Hiemstra and
               Marie{-}Francine Moens and
               Josiane Mothe and
               Raffaele Perego and
               Martin Potthast and
               Fabrizio Sebastiani},
  title     = {Content Selection Network for Document-Grounded Retrieval-Based Chatbots},
  booktitle = {Advances in Information Retrieval - 43rd European Conference on {IR}
               Research, {ECIR} 2021, Virtual Event, March 28 - April 1, 2021, Proceedings,
               Part {I}},
  series    = {Lecture Notes in Computer Science},
  volume    = {12656},
  pages     = {755--769},
  publisher = {Springer},
  year      = {2021},
  url       = {https://doi.org/10.1007/978-3-030-72113-8\_50},
  doi       = {10.1007/978-3-030-72113-8\_50}
}
```
