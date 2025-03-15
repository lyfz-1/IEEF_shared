# README

**Paper title:** IEEF: Enhancing Commit Message Generation by Segmenting Key Information in High-quality Data

**Introduction:** **IEEF **(Information Extraction, Enhancement, Fine-tuning): A novel training framework for enhancing commit message generation, which consists of three stages. By introducing the two key information of ‘what’ and ‘why’ during training, IEEF-enhanced models perform better in generating high-quality commit messages.

**Note:** This repository contains our **data, scripts, and experimental results**.



##  Environment

```
conda create -n IEEF python=3.6 -y
conda activate IEEF
pip install torch==1.10 transformers==4.12.5 tqdm==4.64.1 prettytable==2.5.0 gdown==4.5.1 more-itertools==8.14.0 tensorboardX==2.5.1 setuptools==59.5.0  tensorboard== 2.10.1
```



## Dataset

* In the **“data”** directory, we provide the training set in RQ1 used to train the Bi-LSTM classifier as well as the test set used for RQ1.

* Due to the size of the HQ-dataset we collected, we share it on Google Cloud. Please see the following link:





## Scripts

In the **"code"** directory, we provide:

* The scripts of LLMs to identify high-quality commit messages and segment "what" and "why" information from them in RQ1.
* The basic scripts to fine-tune and test the pre-trained models in RQ2 and RQ3. Meanwhile, the script for evaluating different models under different settings：**evaluate.py**
  * How to evaluate： **python evaluate.py  --refs_filename  [The path of the reference file] --preds_filename [The path of the predicted file]**
* The scripts of LLMs to generate high-quality commit messages in Discussion



Checkpoints from Hugging Face:

| Pre-trained Models | Checkpoint used in our study from Hugging Face               |
| ------------------ | ------------------------------------------------------------ |
| PLBART             | [uclanlp/plbart-base · Hugging Face](https://huggingface.co/uclanlp/plbart-base) |
| CodeT5-small       | [Salesforce/codet5-small · Hugging Face](https://huggingface.co/Salesforce/codet5-small) |
| CodeT5-base        | [Salesforce/codet5-base · Hugging Face](https://huggingface.co/Salesforce/codet5-base) |
| CodeT5+            | [Salesforce/codet5p-220m · Hugging Face](https://huggingface.co/Salesforce/codet5p-220m) |
| CodeTrans-small    | [SEBIS/code_trans_t5_small_commit_generation_transfer_learning_finetune · Hugging Face](https://huggingface.co/SEBIS/code_trans_t5_small_commit_generation_transfer_learning_finetune) |
| CodeTrans-base     | [SEBIS/code_trans_t5_base_commit_generation_transfer_learning_finetune · Hugging Face](https://huggingface.co/SEBIS/code_trans_t5_base_commit_generation_transfer_learning_finetune) |
| UniXcoder          | [microsoft/unixcoder-base · Hugging Face](https://huggingface.co/microsoft/unixcoder-base) |
| CodeReviewer       | [microsoft/codereviewer · Hugging Face](https://huggingface.co/microsoft/codereviewer) |

For detailed code, please refer to the official public implementation packages for each model.



## Results

In the **"results"** directory, we provide the experimental results of our four research questions and the discussion section involved in this study. For ease of viewing, we've divided the subdirectories by name.

Please refer to the paper for details of the experiments and the resulting data.