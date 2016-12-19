
def load_parameters():
    '''
    Loads the defined parameters
    '''

    # Input data params
    DATASET_NAME = 'cnn_polarity'
    FILL = 'end'                                  # whether we fill the 'end' or the 'start' of the sentence with 0s
    SRC_LAN = 'de'                                # Input language
    TRG_LAN = 'en'                                # Outputs language
    MODE = 'semisupervised-selection'             # 'training', 'sampling', 'semisupervised-selection'

    BINARY_SELECTION = True                       # Binary classification problem (currently, 'semisupervised-selection' only supports BINARY_SELECTION)
    # Path to data
    ROOT_PATH = '/media/HDD_2TB/DATASETS/%s/' % DATASET_NAME             # Root path to the data folder
    DATA_ROOT_PATH = ROOT_PATH + 'DATA/Emea-Euro/De-En'                  # Path to the corpora folder
    DEST_ROOT_PATH = ROOT_PATH + 'Selection-Keras/' + SRC_LAN + TRG_LAN  # Path to store results
    DEBUG = False                                                        # If True, it will store temporal files
    INSTANCES_TO_ADD = 50000                                             # 'r' parameter. Number of sentences added at each iteration

    if BINARY_SELECTION:
        POSITIVE_FILENAME = 'EMEA.de-en.Sin-repetidas'                   # In-domain corpus (I)
        NEGATIVE_FILENAME = 'dev'                                        # Initial negative corpus (N_0)
        if 'semisupervised' in MODE:
            POOL_FILENAME = 'training'                                   # Initial pool of out-of-domain sentences (G_0)

    # Fill these dictionaries for a regular sentence classification task
    TEXT_FILES = {}   #{'train': 'train.' + SRC_LAN, 'val': 'val.' + SRC_LAN}
    CLASS_FILES = {}  #{'train': 'train.' + SRC_LAN.class, 'val': 'val.' + SRC_LAN.class}

    # Dataset parameters
    INPUTS_IDS_DATASET = ['input_text']           # Corresponding inputs of the dataset
    OUTPUTS_IDS_DATASET = ['class']               # Corresponding outputs of the dataset
    INPUTS_IDS_MODEL = ['input_text']             # Corresponding inputs of the built model
    OUTPUTS_IDS_MODEL = ['class']                 # Corresponding outputs of the built model

    # Evaluation params
    METRICS = ['multilabel_metrics']              # Metric used for evaluating model after each epoch (leave empty if only prediction is required)
    EVAL_ON_SETS = []                             # Possible values: 'train', 'val' and 'test' (external evaluator)
    EVAL_ON_SETS_KERAS = []                       # Possible values: 'train', 'val' and 'test' (Keras' evaluator)
    START_EVAL_ON_EPOCH = 1                       # First epoch where the model will be evaluated
    EVAL_EACH_EPOCHS = True                       # Select whether evaluate between N epochs or N updates
    EVAL_EACH = 1                                 # Sets the evaluation frequency (epochs or updates)

    # Early stop parameters
    EARLY_STOP = True                             # Turns on/off the early stop protocol
    PATIENCE = 15                                 # We'll stop if the val STOP_METRIC does not improve after this number of evaluations
    STOP_METRIC = 'accuracy'                      # Metric for the stop

    # Word representation params
    TOKENIZATION_METHOD = 'tokenize_CNN_sentence' # Select which tokenization we'll apply:

    # Input text parameters
    VOCABULARY_SIZE = 0                          # Size of the input vocabulary. Set to 0 for using all, otherwise will be truncated to these most frequent words.
    MIN_OCCURRENCES_VOCAB = 0                    # Minimum number of occurrences allowed for the words in the vocabulay. Set to 0 for using them all.
    MAX_INPUT_TEXT_LEN = 50                      # Maximum length of the input sequence

    # Input text parameters
    INPUT_VOCABULARY_SIZE = 0                    # Size of the input vocabulary. Set to 0 for using all, otherwise will be truncated to these most frequent words.
    MIN_OCCURRENCES_VOCAB = 0                    # Minimum number of occurrences allowed for the words in the vocabulay. Set to 0 for using them all.



    # Optimizer parameters (see model.compile() function)
    LOSS = 'categorical_crossentropy'
    CLASS_MODE = 'categorical'

    OPTIMIZER = 'Adam'                           # Optimizer
    LR = 0.0001                                  # (recommended values - Adam 0.001 - Adadelta 1.0
    WEIGHT_DECAY = 1e-4                          # L2 regularization
    CLIP_C = 9.                                  # During training, clip gradients to this norm
    SAMPLE_WEIGHTS = False                       # Select whether we use a weights matrix (mask) for the data outputs

    # Training parameters
    MAX_EPOCH =  10                              # Stop when computed this number of epochs
    BATCH_SIZE = 768                             # Training batch size
    N_ITER = 15                                  # Iterations to perform of the semisupervised selection

    HOMOGENEOUS_BATCHES = False                  # Use batches with homogeneous output lengths for every minibatch (Dangerous!)
    PARALLEL_LOADERS = 8                         # Parallel data batch loaders
    EPOCHS_FOR_SAVE = 1                          # Number of epochs between model saves
    WRITE_VALID_SAMPLES = True                   # Write valid samples in file


    # Model parameters
    MODEL_TYPE = 'BLSTM_Classifier'              # Model type. See model_zoo.py
    CLASSIFIER_ACTIVATION = 'softmax'            # Last layer activation
    PAD_ON_BATCH = 'CNN' not in MODEL_TYPE       # Padded batches
    N_CLASSES = 2                                # Number of classes
    DATA_AUGMENTATION = False                    # Apply data augmentation on input data (still unimplemented for text inputs)

    # Word embedding parameters
    GLOVE_VECTORS = '/media/HDD_2TB/DATASETS/cnn_polarity/DATA/word2vec.%s.npy' % SRC_LAN   # Path to pretrained vectors. Set to None if you don't want to use pretrained vectors.
    GLOVE_VECTORS_TRAINABLE = True                                                          # Finetune or not the word embedding vectors.
    TEXT_EMBEDDING_HIDDEN_SIZE = 300                                                        # When using pretrained word embeddings, this parameter must match with the word embeddings size

    # LSTM layers dimensions (Only used if needed)
    LSTM_ENCODER_HIDDEN_SIZE = 300  # For models with LSTM encoder

    # FC layers for initializing the first LSTM state
    # Here we should only specify the activation function of each layer (as they have a potentially fixed size)
    # (e.g INIT_LAYERS = ['tanh', 'relu'])
    INIT_LAYERS = ['tanh']

    # CNN layers parameters (Only used if needed)
    NUM_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    POOL_LENGTH = 2
    CNN_ACTIVATION = 'relu'

    # General architectural parameters
    # Fully-connected layers for visual embedding
    # Here we should specify the activation function and the output dimension
    # (e.g ADDITIONAL_EMBEDDING_LAYERS = [('linear', 1024)]

    ADDITIONAL_EMBEDDING_LAYERS = []

    # additional Fully-Connected layers's sizes applied before softmax.
    # Here we should specify the activation function and the output dimension
    # (e.g DEEP_OUTPUT_LAYERS = [('tanh', 600), ('relu',400), ('relu':200)])
    DEEP_OUTPUT_LAYERS = [('relu', 200), ('linear', 100)]

    # Regularizers
    USE_DROPOUT = False                           # Use dropout
    DROPOUT_P = 0.5                               # Percentage of units to drop

    USE_NOISE = True                              # Use gaussian noise during training
    NOISE_AMOUNT = 0.01                           # Amount of noise

    USE_BATCH_NORMALIZATION = True                # If True it is recommended to deactivate Dropout
    BATCH_NORMALIZATION_MODE = 1                  # See documentation in Keras' BN

    USE_PRELU = False                             # use PReLU activations as regularizer
    USE_L2 = False                                # L2 normalization on the features

    # Results plot and models storing parameters
    EXTRA_NAME = ''  # This will be appended to the end of the model name
    MODEL_NAME = DATASET_NAME + '_' + MODEL_TYPE + '_txtemb_' + str(TEXT_EMBEDDING_HIDDEN_SIZE) + \
                 '_addemb_' + '_'.join([layer[0] for layer in ADDITIONAL_EMBEDDING_LAYERS]) + \
                 '_' + str(LSTM_ENCODER_HIDDEN_SIZE) + \
                 '_deepout_' + '_'.join([layer[0] for layer in DEEP_OUTPUT_LAYERS]) + \
                 '_' + OPTIMIZER

    MODEL_NAME += EXTRA_NAME

    STORE_PATH = 'trained_models/' + MODEL_NAME + '/'  # Models and evaluation results will be stored here
    DATASET_STORE_PATH = 'datasets/'                   # Dataset instance will be stored here

    SAMPLING_SAVE_MODE = 'numpy'                       # 'list', 'numpy', 'vqa'
    VERBOSE = 1                                        # Verbosity level
    RELOAD = 0                                         # If 0 start training from scratch, otherwise the model
                                                       # Saved on epoch 'RELOAD' will be used
    REBUILD_DATASET = True                             # Build again or use stored instance

    # Extra parameters for special trainings
    TRAIN_ON_TRAINVAL = False                          # train the model on both training and validation sets combined
    FORCE_RELOAD_VOCABULARY = False                    # force building a new vocabulary from the training samples
                                                       # applicable if RELOAD > 1
    # ============================================
    parameters = locals().copy()
    return parameters
