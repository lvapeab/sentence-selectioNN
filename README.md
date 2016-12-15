# Neural Networks for Data Selection 
 
This repository contains the code for the paper "[Neural Networks Classifier for Data Selection in Statistical Machine Translation](*url*)"
 
Built upon our fork of [Keras](https://github.com/MarcBS/keras) framework and tested for the [Theano](http://deeplearning.net/software/theano)
backend.

## Features

* Neural network-based sentence classifiers.

* BLSTMs / CNNs classifiers. Easy to extend. 

* Iterative semi-supervised selection from top/bottom scoring sentences from an out-of-domain corpus. 


## Architecture

![NN_Classifier](./sentence_classifier.png)


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

```

TODO

```

## Contact

Álvaro Peris ([web page](http://lvapeab.github.io/)): lvapeab@prhlt.upv.es
