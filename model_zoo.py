from keras.engine import Input
from keras.engine.topology import merge
from keras.layers import TimeDistributed, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, LSTMCond, AttLSTM, AttLSTMCond, AttGRUCond
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization, L2_norm
from keras.layers.convolutional import ZeroPadding1D
from keras.layers.core import Dropout, Dense, Flatten, Activation, Lambda, MaxoutDense, MaskedMean
from keras.models import model_from_json, Sequential, Graph, Model
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2, activity_l2
from keras.layers.convolutional import AveragePooling1D
from keras.optimizers import Adam, RMSprop, Nadam, Adadelta
from keras import backend as K
from keras.regularizers import l2

from keras_wrapper.cnn_model import CNN_Model

import numpy as np
import os
import logging
import shutil
import time

class Text_Classification_Model(CNN_Model):
    
    def __init__(self, params, type='Basic_Text_Classification_Model', verbose=1, structure_path=None, weights_path=None,
                 model_name=None, vocabularies=None, store_path=None):
        """
            Text_Classification_Model object constructor.

            :param params: all hyperparameters of the model.
            :param type: network name type (corresponds to any method defined in the section 'MODELS' of this class). Only valid if 'structure_path' == None.
            :param verbose: set to 0 if you don't want the model to output informative messages
            :param structure_path: path to a Keras' model json file. If we speficy this parameter then 'type' will be only an informative parameter.
            :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
            :param model_name: optional name given to the network (if None, then it will be assigned to current time as its name)
            :param vocabularies: vocabularies used for GLOVE word embedding
            :param store_path: path to the folder where the temporal model packups will be stored

            References:
                [PReLU]
                Kaiming He et al. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

                [BatchNormalization]
                Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        """
        super(self.__class__, self).__init__(type=type, model_name=model_name,
                                             silence=verbose == 0, models_path=store_path, inheritance=True)

        self.__toprint = ['_model_type', 'name', 'model_path', 'verbose']

        self.verbose = verbose
        self._model_type = type
        self.params = params
        self.vocabularies = vocabularies

        # Sets the model name and prepares the folders for storing the models
        self.setName(model_name, store_path=store_path)

        # Prepare GLOVE embedding
        if params['GLOVE_VECTORS'] is not None:
            if self.verbose > 0:
                logging.info("<<< Loading pretrained word vectors from file "+ params['GLOVE_VECTORS'] +" >>>")
            self.word_vectors = np.load(os.path.join(params['GLOVE_VECTORS'])).item()
        else:
            self.word_vectors = dict()

        # Prepare model
        if structure_path:
            # Load a .json model
            if self.verbose > 0:
                logging.info("<<< Loading model structure from file "+ structure_path +" >>>")
            self.model = model_from_json(open(structure_path).read())
        else:
            # Build model from scratch
            if hasattr(self, type):
                if self.verbose > 0:
                    logging.info("<<< Building "+ type +" Text_Classification_Model >>>")
                eval('self.'+type+'(params)')
            else:
                raise Exception('Text_Classification_Model type "'+ type +'" is not implemented.')

        # Load weights from file
        if weights_path:
            if self.verbose > 0:
                logging.info("<<< Loading weights from file "+ weights_path +" >>>")
            self.model.load_weights(weights_path)

        # Print information of self
        if verbose > 0:
            print str(self)
            self.model.summary()

        self.setOptimizer()

    def setOptimizer(self):

        """
            Sets a new optimizer for the Text_Classification_Model.
        """

        # compile differently depending if our model is 'Sequential' or 'Graph'
        if self.verbose > 0:
            logging.info("Preparing optimizer and compiling.")
        if self.params['OPTIMIZER'].lower() == 'adam':
            optimizer = Adam(lr=self.params['LR'], clipnorm=self.params['CLIP_C'])
        elif self.params['OPTIMIZER'].lower() == 'rmsprop':
            optimizer = RMSprop(lr=self.params['LR'], clipnorm=self.params['CLIP_C'])
        elif self.params['OPTIMIZER'].lower() == 'nadam':
            optimizer = Nadam(lr=self.params['LR'], clipnorm=self.params['CLIP_C'])
        elif self.params['OPTIMIZER'].lower() == 'adadelta':
            optimizer = Adadelta(lr=self.params['LR'], clipnorm=self.params['CLIP_C'])
        else:
            logging.info('\tWARNING: The modification of the LR is not implemented for the chosen optimizer.')
            optimizer = self.params['OPTIMIZER']
        self.model.compile(optimizer=optimizer, loss=self.params['LOSS'],
                           sample_weight_mode='temporal' if self.params['SAMPLE_WEIGHTS'] else None)


    def setName(self, model_name, store_path=None, clear_dirs=True):
        """
            Changes the name (identifier) of the Text_Classification_Model instance.
        """
        if model_name is None:
            self.name = time.strftime("%Y-%m-%d") + '_' + time.strftime("%X")
            create_dirs = False
        else:
            self.name = model_name
            create_dirs = True

        if store_path is None:
            self.model_path = 'Models/' + self.name
        else:
            self.model_path = store_path


        # Remove directories if existed
        if clear_dirs:
            if os.path.isdir(self.model_path):
                shutil.rmtree(self.model_path)

        # Create new ones
        if create_dirs:
            if not os.path.isdir(self.model_path):
                os.makedirs(self.model_path)


    def sample(a, temperature=1.0):
        # helper function to sample an index from a probability array
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        return np.argmax(np.random.multinomial(1, a, 1))


    def sampling(self, scores, sampling_type='max_likelihood', temperature=1.0):
        """
        Sampling words (each sample is drawn from a categorical distribution).
        Or picks up words that maximize the likelihood.
        In:
            scores - array of size #samples x #classes;
                every entry determines a score for sample i having class j
            temperature - temperature for the predictions;
                the higher the flatter probabilities and hence more random answers

        Out:
            set of indices chosen as output, a vector of size #samples
        """
        if isinstance(scores, dict):
            scores = scores['output']

        if sampling_type == 'multinomial':
            logscores = np.log(scores) / temperature
            # numerically stable version
            normalized_logscores= logscores - np.max(logscores, axis=-1)[:, np.newaxis]
            margin_logscores = np.sum(np.exp(normalized_logscores),axis=-1)
            probs = np.exp(normalized_logscores) / margin_logscores[:, np.newaxis]

            #probs = probs.astype('float32')
            draws = np.zeros_like(probs)
            num_samples = probs.shape[0]
            # we use 1 trial to mimic categorical distributions using multinomial
            for k in xrange(num_samples):
                #probs[k,:] = np.random.multinomial(1,probs[k,:],1)
                #return np.argmax(probs, axis=-1)
                draws[k, :] = np.random.multinomial(1,probs[k,:],1)
            return np.argmax(draws, axis=-1)
        elif sampling_type == 'max_likelihood':
            return np.argmax(scores, axis=-1)
        else:
            raise NotImplementedError()

    def decode_predictions(self, preds, temperature, index2word, sampling_type, verbose=0):
        """
        Decodes predictions

        In:
            preds - predictions codified as the output of a softmax activation function
            temperature - temperature for sampling
            index2word - mapping from word indices into word characters
            verbose - verbosity level, by default 0

        Out:
            Answer predictions (list of answers)
        """
        if verbose > 0:
            logging.info('Decoding prediction ...')
        flattened_preds = preds.reshape(-1, preds.shape[-1])
        flattened_answer_pred = map(lambda x: index2word[x],self.sampling(scores=flattened_preds,
                                                                          sampling_type=sampling_type,
                                                                          temperature=temperature))
        answer_pred_matrix = np.asarray(flattened_answer_pred).reshape(preds.shape[:2])
        answer_pred = []
        EOS = '<eos>'
        BOS = '<bos>'
        PAD = '<pad>'

        for a_no in answer_pred_matrix:
            init_token_pos = 0
            end_token_pos = [j for j, x in enumerate(a_no) if x==EOS or x == PAD]
            end_token_pos = None if len(end_token_pos) == 0 else end_token_pos[0]
            tmp = ' '.join(a_no[init_token_pos:end_token_pos])
            answer_pred.append(tmp)
        return answer_pred


    def decode_predictions_beam_search(self, preds, index2word, verbose=0):
        """
        Decodes predictions

        In:
            preds - predictions codified as word indices
            temperature - temperature for sampling
            index2word - mapping from word indices into word characters
            verbose - verbosity level, by default 0

        Out:
            Answer predictions (list of answers)
        """
        if verbose > 0:
            logging.info('Decoding beam search prediction ...')
        flattened_answer_pred = [map(lambda x: index2word[x], pred) for pred in preds]
        answer_pred = []
        for a_no in flattened_answer_pred:
            tmp = ' '.join(a_no[:-1])
            answer_pred.append(tmp)
        return answer_pred

    def decode_predictions_one_hot(self, preds, index2word, verbose=0):
        """
        Decodes predictionss

        In:
            preds - predictions codified as one hot vectors
            index2word - mapping from word indices into word characters
            verbose - verbosity level, by default 0

        Out:
            Answer predictions (list of answers)
        """
        if verbose > 0:
            logging.info('Decoding one hot prediction ...')
        preds = map(lambda x: np.nonzero(x)[1], preds)
        PAD = '<pad>'
        flattened_answer_pred = [map(lambda x: index2word[x], pred) for pred in preds]
        answer_pred_matrix = np.asarray(flattened_answer_pred)
        answer_pred = []

        for a_no in answer_pred_matrix:
            end_token_pos = [j for j, x in enumerate(a_no) if x == PAD]
            end_token_pos = None if len(end_token_pos) == 0 else end_token_pos[0]
            tmp = ' '.join(a_no[:end_token_pos])
            answer_pred.append(tmp)
        return answer_pred
    # ------------------------------------------------------- #
    #       VISUALIZATION
    #           Methods for visualization
    # ------------------------------------------------------- #

    def __str__(self):
        """
            Plot basic model information.
        """
        obj_str = '-----------------------------------------------------------------------------------\n'
        class_name = self.__class__.__name__
        obj_str += '\t\t'+class_name +' instance\n'
        obj_str += '-----------------------------------------------------------------------------------\n'

        # Print pickled attributes
        for att in self.__toprint:
            obj_str += att + ': ' + str(self.__dict__[att])
            obj_str += '\n'

        obj_str += '\n'
        obj_str += 'MODEL PARAMETERS:\n'
        obj_str += str(self.params)
        obj_str += '\n'

        obj_str += '-----------------------------------------------------------------------------------'

        return obj_str

    # ------------------------------------------------------- #
    #       PREDEFINED MODELS
    # ------------------------------------------------------- #

    def BLSTM_Classifier(self, params):

        # Store inputs and outputs names
        self.ids_inputs = params['INPUTS_IDS_MODEL']
        self.ids_outputs =  params['OUTPUTS_IDS_MODEL']

        # Source text
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TEXT_EMBEDDING_HIDDEN_SIZE'],
                        name='source_word_embedding',
                        W_regularizer=l2(params['WEIGHT_DECAY']),
                        mask_zero=True)(src_text)

        annotations = Bidirectional(GRU(params['LSTM_ENCODER_HIDDEN_SIZE'],
                                             W_regularizer=l2(params['WEIGHT_DECAY']),
                                             U_regularizer=l2(params['WEIGHT_DECAY']),
                                             b_regularizer=l2(params['WEIGHT_DECAY']),
                                             return_sequences=False),
                                        name='bidirectional_encoder')(src_embedding)

        # Softmax
        output = Dense(params['N_CLASSES'],
                       activation=params['CLASSIFIER_ACTIVATION'],
                       name=self.ids_outputs[0],
                       W_regularizer=l2(params['WEIGHT_DECAY']))(annotations)

        self.model = Model(input=src_text, output=output)


    # ------------------------------------------------------- #
    #       SAVE/LOAD
    #           Auxiliary methods for saving and loading the model.
    # ------------------------------------------------------- #

    def __getstate__(self):
        """
            Behaviour applied when pickling a Text_Classification_Model instance.
        """
        obj_dict = self.__dict__.copy()
        del obj_dict['model']
        return obj_dict

