# README

**Paper title:** IEEF: Enhancing Commit Message Generation by Segmenting Key Information in High-quality Data

**Introduction:** **IEEF **(Information Extraction, Enhancement, Fine-tuning): A novel training framework for enhancing commit message generation, which consists of three stages. By introducing the two key information of ‘what’ and ‘why’ during training, IEEF-enhanced models perform better in generating high-quality commit messages.

**Note:** This repository contains our **data, scripts, and experimental results**.



##  1 Environment

```
conda create -n IEEF python=3.6 -y
conda activate IEEF
pip install torch==1.10 transformers==4.12.5 tqdm==4.64.1 prettytable==2.5.0 gdown==4.5.1 more-itertools==8.14.0 tensorboardX==2.5.1 setuptools==59.5.0  tensorboard== 2.10.1
```



## 2 Dataset

* In the **“data”** directory, we provide the training set in RQ1 used to train the Bi-LSTM classifier as well as the test set used for RQ1.

* Due to the size of the HQCMD we collected, we shared it on Google Drive. Please see the following link:

https://drive.google.com/drive/folders/1m64O3r_GEwbUfTnbmQzRXGWYCG46Eo7j?usp=sharing



## 3 Scripts

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



## 4 Results

In the **"results"** directory, we provide the experimental results of our four research questions and the discussion section involved in this study. For ease of viewing, we've divided the subdirectories by name.

Please refer to the paper for details of the experiments and the resulting data.



## 5 Supplement

### 5.1 Significance analysis

In order to prove that the performance improvement of IEEF for the pre-trained models is statistically significant, we conducted on the experimental results of RQ2 as well as RQ3, respectively. Specifically, we used the Mann-Whitney U Test.



**Significance analysis for RQ2（IEEF-enhanced VS HQ-Fine-tuned）**

![rq2_sa.png](https://youke1.picui.cn/s1/2025/09/28/68d89438c6417.png)

Note: The table shows the p-value, where a p-value < 0.05 indicates a significant difference between the two value distributions (highlighted in green in the table).



**Significance analysis for RQ3（IEEF-enhanced VS w/o what & IEEF-enhanced VS w/o why）**

![rq3_sa.png](https://youke1.picui.cn/s1/2025/09/28/68d8958f8ddfb.png)

The results of significance analysis show that the improvement of IEEF on these pre-trained models is statistically significant in most cases.



### 5.2 Designed prompt for LLMs to generate high-quality commit messages

![llms_prompt.png](https://youke1.picui.cn/s1/2025/09/28/68d89957bc304.png)

In this prompt, we first assign a specific role to the LLMs and define what constitutes a high-quality commit message. Then, we provide several examples of code changes along with their corresponding high-quality commit messages to guide the LLMs in learning the task pattern. Finally, the LLMs are required to generate an appropriate commit message for new code changes.



### 5.3 Performance of IEEF-enhanced models under different settings

![Snipaste_2025-09-28_10-19-54.png](https://youke1.picui.cn/s1/2025/09/28/68d89b8830491.png)

Note: 

* IEEF-half: We halved the amount of data used in the enhancement stage.
* IEEF-reversed: We reversed the order of the augmentation tasks.