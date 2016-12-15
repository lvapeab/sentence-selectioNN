# Neural Networks for Data Selection 
 
This repository contains the code for the paper "[Neural Networks Classifier for Data Selection in Statistical Machine Translation](*url*)"
 
Built upon our fork of [Keras](https://github.com/MarcBS/keras) framework and tested for the [Theano](http://deeplearning.net/software/theano)
backend.

## Features

* Neural network-based sentence classifiers.

* BLSTMs / CNNs classifiers. Easy to extend. 

* [Glove](https://github.com/lvapeab/sentence-selectioNN/blob/master/utils/preprocess_glove_vectors.py) / [Word2Vec](https://github.com/lvapeab/sentence-selectioNN/blob/master/utils/preprocess_word2vec_vectors.py) pretrained word vectors.  


* Iterative semi-supervised selection from top/bottom scoring sentences from an out-of-domain corpus. 


## Architecture

![NN_Classifier](./docs/sentence_classifier.png)


## Installation

CNN_Sel requires the following libraries:

 - [Our version of Keras](https://github.com/MarcBS/keras) 
 - [Staged Keras Wrapper](https://github.com/MarcBS/staged_keras_wrapper) 

## Instructions:

Assuming you have a corpus:

1) Check out the inputs/outputs of your model in `data_engine/prepare_data.py`

2) Set a model configuration in  `config.py`

3) Train!:

  ``
 python main.py
 ``


## Citation

If you use this code for any purpose, please, do not forget to cite the following paper:
``
Peris Á., Chinea-Rios M., Casacuberta F. 
Neural Networks Classifier for Data Selection in Statistical Machine Translation. 
arXiv preprint arXiv:?. 2016.
``


## Contact

Álvaro Peris ([web page](http://lvapeab.github.io/)): lvapeab@prhlt.upv.es
