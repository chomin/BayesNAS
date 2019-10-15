# BayesNAS
code for ICML paper *BayesNAS: A Bayesian Approach for Neural Architecture Search*

## Description
The algorithm is a one-shot based Bayesian approach for neural architecutre search. It is capable of finding efficient network architecture for image classification task. This approach is also very flexible in the sense of finding good trade-off between classification accuracy and model complexity.

## Requirements
```
Python >= 3.5.5, PyTorch == 0.3.1, torchvision == 0.2.0
```
NOTE: PyTorch 0.4 is not supported at this moment and would lead to OOM.


## RNN
Besides convolutional neural networks, we also provide theoretical analysis about sparse prior and Hessian computation for recurrent layers in [RNN](./RNN.pdf).

## Search
To run the searching algorithm, please go to [search](./search) and run *main.py* with
```bash
python main.py --lambda_child 0.01 --lambda_origin 0.01 --t_max 1
```
A folder is expected to appear in the same directory containing all parameters of our algorithm wrt. to the normal cell and reduct cell.  

## Cell Evaluation
In the [search](./search), please load the corresponding *gamma_norm.pkl* and *gamma_reduct.pkl* for cell selection and building. Then replace the *NormalCell* and *ReductCell* in *cell.py* in [CNN](./CNN) manually.

(Unfortunately we don't support automation of this process at the moment.)

## Architecture evaluation (using full-sized models)
To evaluate our best cells by training from scratch, run
```
cd cnn && python train.py --auxiliary --cutout            # CIFAR-10
cd rnn && python train.py                                 # PTB
cd rnn && python train.py --data ../data/wikitext-2 \     # WT2
            --dropouth 0.15 --emsize 700 --nhidlast 700 --nhid 700 --wdecay 5e-7
cd cnn && python train_imagenet.py --auxiliary            # ImageNet
```
Customized architectures are supported through the `--arch` flag once specified in `genotypes.py`.

The CIFAR-10 result at the end of training is subject to variance due to the non-determinism of cuDNN back-prop kernels. _It would be misleading to report the result of only a single run_. By training our best cell from scratch, one should expect the average test error of 10 independent runs to fall in the range of 2.76 +/- 0.09% with high probability.



## Pre-trained
We also provide our best pre-trained model in [pre_trained](./CNN/pre_trained) for different lambdas. You can find them according to the corresponding lambda.

