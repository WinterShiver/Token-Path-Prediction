# Reading Order Matters: Information Extraction from Visually-rich Documents by Token Path Prediction

This is an **unofficial** re-implementation of:
* **Token Path Prediction** (TPP), an unified model for multiple VrD-IE tasks: [[EMNLP 2023 paper]](https://arxiv.org/abs/2310.11016);
* **LayoutMask**, a novel pre-trained text-and-layout model for VrDU: [[ACL 2023 paper]](https://arxiv.org/abs/2305.18721).

This repository contains the code implementation of TPP for three tasks, and the pre-training code of LayoutMask.
The implementation of this repository has referred to the revised datasets FUNSD-r and CORD-r officially available at [Token-Path-Prediction-Datasets](https://github.com/chongzhangFDU/Token-Path-Prediction-Datasets).

![Token Path Prediction](https://ar5iv.labs.arxiv.org/html/2310.11016/assets/x3.png)
![LayoutMask](https://ar5iv.labs.arxiv.org/html/2305.18721/assets/images/model.jpg)

## Citation

If the work is helpful to you, please kindly cite these paper as:

```
@misc{zhang2023reading,
      title={Reading Order Matters: Information Extraction from Visually-rich Documents by Token Path Prediction}, 
      author={Chong Zhang and Ya Guo and Yi Tu and Huan Chen and Jinyang Tang and Huijia Zhu and Qi Zhang and Tao Gui},
      year={2023},
      eprint={2310.11016},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{tu2023layoutmask,
      title={LayoutMask: Enhance Text-Layout Interaction in Multi-modal Pre-training for Document Understanding}, 
      author={Yi Tu and Ya Guo and Huan Chen and Jinyang Tang},
      year={2023},
      eprint={2305.18721},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Environments

```bash
# python 3.8
conda create -n LayoutIE python=3.8
conda activate LayoutIE
# pip uninstall torchtext
# pip install torch==1.11.0+cu115
# pip install torchvision==0.12.0+cu115
pip install torchmetrics==0.11.1
pip install transformers==4.26.1
pip install pytorch-lightning==1.5.9
pip install nltk==3.8.1
pip install jieba==0.42.1
pip install seqeval==1.2.2
pip install ark_nlp==0.0.9
pip install opencv-python==4.7.0.68
pip install opencv-python-headless==4.7.0.68
pip install timm==0.6.12
pip install sentencepiece==0.1.97
pip install six==1.16.0
```

## Scripts for tasks

### Named Entity Recognition (VrD-NER)

Please use the FUNSD-r/CORD-r datasets, or the pre-processed FUNSD/CORD datasets at [Token-Path-Prediction-Datasets](https://github.com/chongzhangFDU/Token-Path-Prediction-Datasets).

* For LayoutLMv3 results: `src/tasks/layoutlm_v3/ner/run.sh`
* For LayoutMask results: `src/tasks/layoutmask/ner/run.sh`

### Entity Linking (VrD-EL)

Please use the pre-processed FUNSD dataset at `data/FUNSD_entity_linking`.

* For LayoutMask results: `src/tasks/layoutmask/re/run.sh`

### Reading Order Prediction (VrD-ROP)

Sample data for ReadingBank: `data/reading_bank`. For full fine-tuning please process the original ReadingBank dataset into the sample format.

* For LayoutMask results: `src/tasks/layoutmask/reading_order/run.sh`

### LayoutMask Pre-training

Due the policies of Ant Group, the pre-trained weights for `layoutmask-english-base` are currently not released.
Yet you can still pre-train a LayoutMask model using the script `src/tasks/layoutmask/pretrain/run.sh`, with constructing pre-training data corresponding to sample data at `data/pretrain`.

## Experiment Results

Experiments are conducted following the proposed implementation details in TPP paper; the optimal learning rates are searched from {3e-5, 5e-5, 8e-5}.
These results have been updated to [PapersWithCode](https://paperswithcode.com/paper/reading-order-matters-information-extraction).

| Model | Task | Dataset | Entity-level F1 | Precision | Recall | Learning Rate |
|-------|:-----:|:-----:|:---------------:|:---------:|:------:|:-------------:|
| TPP (LayoutMask-base) | VrD-NER | FUNSD | 85.16  | 84.05 |  86.29  | 3e-5 | 
| TPP (LayoutMask-base) | VrD-NER | CORD  | 96.92  | 97.03 |  96.80  | 3e-5 | 
