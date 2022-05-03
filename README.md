# DPGCN-main

## Requirements

Our code works with the following environment.
python=3.7
pytorch=1.4.0
cuda = 10.0
others:
nltk
numpy
sklearn

## Downloading BERT

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

For BERT, please download pre-trained BERT-Base and BERT-Large English from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

## Run on Data
Run `run.sh` to train a model on the  data under the `Data` directory.

## Train and Test
python dpgcn_main.py --train_file ./data/laptop_train.txt --test_file ./data/laptop_test.txt

## Contact me
yangjinjie@m.scnu.edu.cn
