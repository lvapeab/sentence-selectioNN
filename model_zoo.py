import logging
import os
import shutil
import time
import numpy as np
from keras.layers import *
from keras.models import model_from_json, Model
from keras.optimizers import Adam, RMSprop, Nadam, Adadelta
from keras.regularizers import l2
from keras_wrapper.cnn_model import CNN_Model
from keras_wrapper.extra.regularize import Regularize


class Text_Classification_Model(CNN_Model):
    def __init__(self, params, type='Basic_Text_Classification_Model', verbose=1, structure_path=None,
                 weights_path=None,
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
        if params['SRC_PRETRAINED_VECTORS'] is not None:
            if self.verbose > 0:
                logging.info(
                    "<<< Loading pretrained word vectors from file " + params['SRC_PRETRAINED_VECTORS'] + " >>>")
            self.word_vectors_src = np.load(os.path.join(params['SRC_PRETRAINED_VECTORS'])).item()
        else:
            self.word_vectors_src = dict()

        # Prepare GLOVE embedding
        if params['TRG_PRETRAINED_VECTORS'] is not None:
            if self.verbose > 0:
                logging.info(
                    "<<< Loading pretrained word vectors from file " + params['TRG_PRETRAINED_VECTORS'] + " >>>")
            self.word_vectors_trg = np.load(os.path.join(params['TRG_PRETRAINED_VECTORS'])).item()
        else:
            self.word_vectors_trg = dict()

        # Prepare model
        if structure_path:
            # Load a .json model
            if self.verbose > 0:
                logging.info("<<< Loading model structure from file " + structure_path + " >>>")
            self.model = model_from_json(open(structure_path).read())
        else:
            # Build model from scratch
            if hasattr(self, type):
                if self.verbose > 0:
                    logging.info("<<< Building " + type + " Text_Classification_Model >>>")
                eval('self.' + type + '(params)')
            else:
                raise Exception('Text_Classification_Model type "' + type + '" is not implemented.')

        # Load weights from file
        if weights_path:
            if self.verbose > 0:
                logging.info("<<< Loading weights from file " + weights_path + " >>>")
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
        obj_str += '\t\t' + class_name + ' instance\n'
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

    def CNN_Classifier(self, params):

        # Store inputs and outputs names
        self.ids_inputs = params['INPUTS_IDS_MODEL']
        self.ids_outputs = params['OUTPUTS_IDS_MODEL']

        # Prepare GLOVE vectors for text embedding initialization
        embedding_weights = np.random.rand(params['INPUT_SRC_VOCABULARY_SIZE'],
                                           params['SRC_TEXT_EMBEDDING_HIDDEN_SIZE'])
        for word, index in self.vocabularies[self.ids_inputs[0]]['words2idx'].iteritems():
            if self.word_vectors_src.get(word) is not None:
                embedding_weights[index, :] = self.word_vectors_src[word]
        self.word_vectors_src = {}

        # Source text
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]),
                         dtype='int32')
        sentence_embedding_glove = Embedding(params['INPUT_SRC_VOCABULARY_SIZE'],
                                             params['SRC_TEXT_EMBEDDING_HIDDEN_SIZE'],
                                             name='source_word_embedding_glove', weights=[embedding_weights],
                                             trainable=params['SRC_PRETRAINED_VECTORS_TRAINABLE'],
                                             W_regularizer=l2(params['WEIGHT_DECAY']),
                                             mask_zero=False)(src_text)
        convolutions = []
        for filter_len in params['FILTER_SIZES']:
            conv = Convolution1D(nb_filter=params['NUM_FILTERS'],
                                 filter_length=filter_len,
                                 activation=params['CNN_ACTIVATION'],
                                 W_regularizer=l2(params['WEIGHT_DECAY']),
                                 b_regularizer=l2(params['WEIGHT_DECAY']))(sentence_embedding_glove)
            pool = MaxPooling1D()(conv)
            # pool = Regularize(pool, params, name='pool_' + str(filter_len))
            convolutions.append(Flatten()(pool))
        if len(convolutions) > 1:
            out_layer = merge(convolutions, mode='concat')
        else:
            out_layer = convolutions[0]
        # Optional deep ouput
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            if activation.lower() == 'maxout':
                out_layer = MaxoutDense(dimension,
                                        W_regularizer=l2(params['WEIGHT_DECAY']),
                                        name='maxout_%d' % i)(out_layer)
            else:
                out_layer = Dense(dimension,
                                  activation=activation,
                                  W_regularizer=l2(params['WEIGHT_DECAY']),
                                  name=activation + '_%d' % i)(out_layer)

            out_layer = Regularize(out_layer, params, name='out_layer_' + str(i))

        # Softmax
        output = Dense(params['N_CLASSES'],
                       activation=params['CLASSIFIER_ACTIVATION'],
                       name=self.ids_outputs[0],
                       W_regularizer=l2(params['WEIGHT_DECAY']))(out_layer)

        self.model = Model(input=src_text, output=output)

    def BLSTM_Classifier(self, params):

        # Store inputs and outputs names
        self.ids_inputs = params['INPUTS_IDS_MODEL']
        self.ids_outputs = params['OUTPUTS_IDS_MODEL']

        # Prepare GLOVE vectors for text embedding initialization
        embedding_weights = np.random.rand(params['INPUT_SRC_VOCABULARY_SIZE'],
                                           params['SRC_TEXT_EMBEDDING_HIDDEN_SIZE'])
        for word, index in self.vocabularies[self.ids_inputs[0]]['words2idx'].iteritems():
            if self.word_vectors_src.get(word) is not None:
                embedding_weights[index, :] = self.word_vectors_src[word]
        self.word_vectors_src = {}

        # Source text
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')
        sentence_embedding = Embedding(params['INPUT_SRC_VOCABULARY_SIZE'], params['SRC_TEXT_EMBEDDING_HIDDEN_SIZE'],
                                       name='source_word_embedding', weights=[embedding_weights],
                                       trainable=params['SRC_PRETRAINED_VECTORS_TRAINABLE'],
                                       W_regularizer=l2(params['WEIGHT_DECAY']),
                                       mask_zero=True)(src_text)
        sentence_embedding = Regularize(sentence_embedding, params, name='source_word_embedding')

        for activation, dimension in params['ADDITIONAL_EMBEDDING_LAYERS']:
            sentence_embedding = TimeDistributed(Dense(dimension, name='%s_1' % activation, activation=activation,
                                                       W_regularizer=l2(params['WEIGHT_DECAY'])))(sentence_embedding)
            sentence_embedding = Regularize(sentence_embedding, params, name='%s_1' % activation)

        out_layer = Bidirectional(LSTM(params['LSTM_ENCODER_HIDDEN_SIZE'],
                                       W_regularizer=l2(params['WEIGHT_DECAY']),
                                       U_regularizer=l2(params['WEIGHT_DECAY']),
                                       b_regularizer=l2(params['WEIGHT_DECAY']),
                                       return_sequences=False),
                                  name='bidirectional_encoder')(sentence_embedding)
        out_layer = Regularize(out_layer, params, name='out_layer')

        # Optional deep ouput
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            if activation.lower() == 'maxout':
                out_layer = MaxoutDense(dimension,
                                        W_regularizer=l2(params['WEIGHT_DECAY']),
                                        name='maxout_%d' % i)(out_layer)
            else:
                out_layer = Dense(dimension,
                                  activation=activation,
                                  W_regularizer=l2(params['WEIGHT_DECAY']),
                                  name=activation + '_%d' % i)(out_layer)
            out_layer = Regularize(out_layer, params, name=activation + '_%d' % i)

        # Softmax
        output = Dense(params['N_CLASSES'],
                       activation=params['CLASSIFIER_ACTIVATION'],
                       name=self.ids_outputs[0],
                       W_regularizer=l2(params['WEIGHT_DECAY']))(out_layer)

        self.model = Model(input=src_text, output=output)

    def Bilingual_CNN_Classifier(self, params):

        # Store inputs and outputs names
        self.ids_inputs = params['INPUTS_IDS_MODEL']
        self.ids_outputs = params['OUTPUTS_IDS_MODEL']

        # Prepare GLOVE vectors for text embedding initialization
        embedding_weights = np.random.rand(params['INPUT_SRC_VOCABULARY_SIZE'],
                                           params['SRC_TEXT_EMBEDDING_HIDDEN_SIZE'])
        for word, index in self.vocabularies[self.ids_inputs[0]]['words2idx'].iteritems():
            if self.word_vectors_src.get(word) is not None:
                embedding_weights[index, :] = self.word_vectors_src[word]
        self.word_vectors_src = {}

        # Source text model
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]),
                         dtype='int32')
        src_sentence_embedding = Embedding(params['INPUT_SRC_VOCABULARY_SIZE'],
                                           params['SRC_TEXT_EMBEDDING_HIDDEN_SIZE'],
                                           name='source_word_embedding', weights=[embedding_weights],
                                           trainable=params['SRC_PRETRAINED_VECTORS_TRAINABLE'],
                                           W_regularizer=l2(params['WEIGHT_DECAY']),
                                           mask_zero=False)(src_text)
        src_sentence_embedding = Regularize(src_sentence_embedding, params, name='source_word_embedding')

        for activation, dimension in params['ADDITIONAL_EMBEDDING_LAYERS']:
            src_sentence_embedding = TimeDistributed(
                Dense(dimension, name='%s_1_src' % activation, activation=activation,
                      W_regularizer=l2(params['WEIGHT_DECAY'])))(src_sentence_embedding)
            src_sentence_embedding = Regularize(src_sentence_embedding, params, name='%s_1_src' % activation)

        src_convolutions = []
        for filter_len in params['FILTER_SIZES']:
            src_conv = Convolution1D(nb_filter=params['NUM_FILTERS'],
                                     filter_length=filter_len,
                                     activation=params['CNN_ACTIVATION'],
                                     W_regularizer=l2(params['WEIGHT_DECAY']),
                                     b_regularizer=l2(params['WEIGHT_DECAY']))(src_sentence_embedding)
            src_pool = MaxPooling1D()(src_conv)
            # pool = Regularize(pool, params, name='pool_' + str(filter_len))
            src_convolutions.append(Flatten()(src_pool))
        if len(src_convolutions) > 1:
            src_out_layer = merge(src_convolutions, mode='concat')
        else:
            src_out_layer = src_convolutions[0]
        # Optional deep ouput
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            if activation.lower() == 'maxout':
                src_out_layer = MaxoutDense(dimension,
                                            W_regularizer=l2(params['WEIGHT_DECAY']),
                                            name='maxout_%d_src' % i)(src_out_layer)
            else:
                src_out_layer = Dense(dimension,
                                      activation=activation,
                                      W_regularizer=l2(params['WEIGHT_DECAY']),
                                      name=activation + '_%d_src' % i)(src_out_layer)

            src_out_layer = Regularize(src_out_layer, params, name='out_layer_%s_src' % str(i))

        # Target text model
        # Prepare GLOVE vectors for text embedding initialization
        embedding_weights = np.random.rand(params['INPUT_TRG_VOCABULARY_SIZE'],
                                           params['TRG_TEXT_EMBEDDING_HIDDEN_SIZE'])
        for word, index in self.vocabularies[self.ids_inputs[1]]['words2idx'].iteritems():
            if self.word_vectors_trg.get(word) is not None:
                embedding_weights[index, :] = self.word_vectors_trg[word]
        self.word_vectors_trg = {}

        # Target text
        trg_text = Input(name=self.ids_inputs[1], batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]),
                         dtype='int32')
        trg_sentence_embedding = Embedding(params['INPUT_TRG_VOCABULARY_SIZE'],
                                           params['TRG_TEXT_EMBEDDING_HIDDEN_SIZE'],
                                           name='target_word_embedding', weights=[embedding_weights],
                                           trainable=params['TRG_PRETRAINED_VECTORS_TRAINABLE'],
                                           W_regularizer=l2(params['WEIGHT_DECAY']),
                                           mask_zero=False)(trg_text)
        trg_sentence_embedding = Regularize(trg_sentence_embedding, params, name='target_word_embedding')

        for activation, dimension in params['ADDITIONAL_EMBEDDING_LAYERS']:
            trg_sentence_embedding = TimeDistributed(
                Dense(dimension, name='%s_1_trg' % activation, activation=activation,
                      W_regularizer=l2(params['WEIGHT_DECAY'])))(trg_sentence_embedding)
            trg_sentence_embedding = Regularize(trg_sentence_embedding, params, name='%s_1_trg' % activation)

        trg_convolutions = []
        for filter_len in params['FILTER_SIZES']:
            trg_conv = Convolution1D(nb_filter=params['NUM_FILTERS'],
                                     filter_length=filter_len,
                                     activation=params['CNN_ACTIVATION'],
                                     W_regularizer=l2(params['WEIGHT_DECAY']),
                                     b_regularizer=l2(params['WEIGHT_DECAY']))(trg_sentence_embedding)
            trg_pool = MaxPooling1D()(trg_conv)
            # pool = Regularize(pool, params, name='pool_' + str(filter_len))
            trg_convolutions.append(Flatten()(trg_pool))
        if len(trg_convolutions) > 1:
            trg_out_layer = merge(trg_convolutions, mode='concat')
        else:
            trg_out_layer = trg_convolutions[0]
        # Optional deep ouput
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            if activation.lower() == 'maxout':
                trg_out_layer = MaxoutDense(dimension,
                                            W_regularizer=l2(params['WEIGHT_DECAY']),
                                            name='maxout_%d_trg' % i)(trg_out_layer)
            else:
                trg_out_layer = Dense(dimension,
                                      activation=activation,
                                      W_regularizer=l2(params['WEIGHT_DECAY']),
                                      name=activation + '_%d_trg' % i)(trg_out_layer)

                trg_out_layer = Regularize(trg_out_layer, params, name='out_layer_%s_trg' % str(i))

        out_layer = merge([src_out_layer, trg_out_layer], mode='concat')
        # Softmax
        output = Dense(params['N_CLASSES'],
                       activation=params['CLASSIFIER_ACTIVATION'],
                       name=self.ids_outputs[0],
                       W_regularizer=l2(params['WEIGHT_DECAY']))(out_layer)

        self.model = Model(input=[src_text, trg_text], output=output)

    def Bilingual_BLSTM_Classifier(self, params):

        # Store inputs and outputs names
        self.ids_inputs = params['INPUTS_IDS_MODEL']
        self.ids_outputs = params['OUTPUTS_IDS_MODEL']

        # Prepare GLOVE vectors for text embedding initialization
        embedding_weights = np.random.rand(params['INPUT_SRC_VOCABULARY_SIZE'],
                                           params['SRC_TEXT_EMBEDDING_HIDDEN_SIZE'])
        for word, index in self.vocabularies[self.ids_inputs[0]]['words2idx'].iteritems():
            if self.word_vectors_src.get(word) is not None:
                embedding_weights[index, :] = self.word_vectors_src[word]
        self.word_vectors_src = {}

        # Source text model
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')
        src_sentence_embedding = Embedding(params['INPUT_SRC_VOCABULARY_SIZE'],
                                           params['SRC_TEXT_EMBEDDING_HIDDEN_SIZE'],
                                           name='source_word_embedding', weights=[embedding_weights],
                                           trainable=params['SRC_PRETRAINED_VECTORS_TRAINABLE'],
                                           W_regularizer=l2(params['WEIGHT_DECAY']),
                                           mask_zero=True)(src_text)
        src_sentence_embedding = Regularize(src_sentence_embedding, params, name='source_word_embedding')

        for activation, dimension in params['ADDITIONAL_EMBEDDING_LAYERS']:
            src_sentence_embedding = TimeDistributed(
                Dense(dimension, name='%s_1_src' % activation, activation=activation,
                      W_regularizer=l2(params['WEIGHT_DECAY'])))(src_sentence_embedding)
            src_sentence_embedding = Regularize(src_sentence_embedding, params, name='%s_1_src' % activation)

        src_out_layer = Bidirectional(LSTM(params['LSTM_ENCODER_HIDDEN_SIZE'],
                                           W_regularizer=l2(params['WEIGHT_DECAY']),
                                           U_regularizer=l2(params['WEIGHT_DECAY']),
                                           b_regularizer=l2(params['WEIGHT_DECAY']),
                                           return_sequences=False),
                                      name='bidirectional_encoder_src')(src_sentence_embedding)
        src_out_layer = Regularize(src_out_layer, params, name='out_layer_src')

        # Optional deep ouput
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            if activation.lower() == 'maxout':
                src_out_layer = MaxoutDense(dimension,
                                            W_regularizer=l2(params['WEIGHT_DECAY']),
                                            name='maxout_%d_src' % i)(src_out_layer)
            else:
                src_out_layer = Dense(dimension,
                                      activation=activation,
                                      W_regularizer=l2(params['WEIGHT_DECAY']),
                                      name=activation + '_%d_src' % i)(src_out_layer)
            src_out_layer = Regularize(src_out_layer, params, name=activation + '_%d_src' % i)

        # Prepare GLOVE vectors for text embedding initialization
        embedding_weights = np.random.rand(params['INPUT_TRG_VOCABULARY_SIZE'],
                                           params['TRG_TEXT_EMBEDDING_HIDDEN_SIZE'])
        for word, index in self.vocabularies[self.ids_inputs[1]]['words2idx'].iteritems():
            if self.word_vectors_trg.get(word) is not None:
                embedding_weights[index, :] = self.word_vectors_trg[word]
        self.word_vectors_trg = {}

        # Target text
        trg_text = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        trg_sentence_embedding = Embedding(params['INPUT_TRG_VOCABULARY_SIZE'],
                                           params['TRG_TEXT_EMBEDDING_HIDDEN_SIZE'],
                                           name='target_word_embedding', weights=[embedding_weights],
                                           trainable=params['TRG_PRETRAINED_VECTORS_TRAINABLE'],
                                           W_regularizer=l2(params['WEIGHT_DECAY']),
                                           mask_zero=True)(trg_text)
        trg_sentence_embedding = Regularize(trg_sentence_embedding, params, name='target_word_embedding')

        for activation, dimension in params['ADDITIONAL_EMBEDDING_LAYERS']:
            trg_sentence_embedding = TimeDistributed(
                Dense(dimension, name='%s_1_trg' % activation, activation=activation,
                      W_regularizer=l2(params['WEIGHT_DECAY'])))(trg_sentence_embedding)
            trg_sentence_embedding = Regularize(trg_sentence_embedding, params, name='%s_1_trg' % activation)

        trg_out_layer = Bidirectional(LSTM(params['LSTM_ENCODER_HIDDEN_SIZE'],
                                           W_regularizer=l2(params['WEIGHT_DECAY']),
                                           U_regularizer=l2(params['WEIGHT_DECAY']),
                                           b_regularizer=l2(params['WEIGHT_DECAY']),
                                           return_sequences=False),
                                      name='bidirectional_encoder_trg')(trg_sentence_embedding)
        trg_out_layer = Regularize(trg_out_layer, params, name='out_layer_trg')

        # Optional deep ouput
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            if activation.lower() == 'maxout':
                trg_out_layer = MaxoutDense(dimension,
                                            W_regularizer=l2(params['WEIGHT_DECAY']),
                                            name='maxout_%d_trg' % i)(trg_out_layer)
            else:
                trg_out_layer = Dense(dimension,
                                      activation=activation,
                                      W_regularizer=l2(params['WEIGHT_DECAY']),
                                      name=activation + '_%d_trg' % i)(trg_out_layer)
            trg_out_layer = Regularize(trg_out_layer, params, name=activation + '_%d_trg' % i)

        out_layer = merge([src_out_layer, trg_out_layer], mode='concat')
        # Softmax
        output = Dense(params['N_CLASSES'],
                       activation=params['CLASSIFIER_ACTIVATION'],
                       name=self.ids_outputs[0],
                       W_regularizer=l2(params['WEIGHT_DECAY']))(out_layer)

        self.model = Model(input=[src_text, trg_text], output=output)

    def __getstate__(self):
        """
            Behaviour applied when pickling a Text_Classification_Model instance.
        """
        obj_dict = self.__dict__.copy()
        del obj_dict['model']
        return obj_dict
