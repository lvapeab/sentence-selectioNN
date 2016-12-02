
def load_parameters():
    '''
    Loads the defined parameters
    '''

    # Input data params
    DATASET_NAME = 'cnn_polarity'
    FILL = 'end'                                  # whether we fill the 'end' or the 'start' of the sentence with 0s
    SRC_LAN = 'sh'                                # Language of the outputs

    DATA_ROOT_PATH = '/media/HDD_2TB/DATASETS/%s/' % DATASET_NAME


    # SRC_LAN or TRG_LAN will be added to the file names
    TEXT_FILES = {'train': 'DATA/training.',
                  'val': 'DATA/val.'}

    CLASS_FILES = {'train': 'DATA/training.class',
                   'val': 'DATA/val.class'}

    # Dataset parameters
    INPUTS_IDS_DATASET = ['input_text']       # Corresponding inputs of the dataset
    OUTPUTS_IDS_DATASET = ['class']           # Corresponding outputs of the dataset
    INPUTS_IDS_MODEL = ['input_text']         # Corresponding inputs of the built model
    OUTPUTS_IDS_MODEL = ['class']             # Corresponding outputs of the built model

    # Evaluation params
    METRICS = ['multilabel_metrics']              # Metric used for evaluating model after each epoch (leave empty if only prediction is required)
    EVAL_ON_SETS = ['val']                        # Possible values: 'train', 'val' and 'test' (external evaluator)
    EVAL_ON_SETS_KERAS = []                       # Possible values: 'train', 'val' and 'test' (Keras' evaluator)
    START_EVAL_ON_EPOCH = 1                       # First epoch where the model will be evaluated
    EVAL_EACH_EPOCHS = True                       # Select whether evaluate between N epochs or N updates
    EVAL_EACH = 1                                 # Sets the evaluation frequency (epochs or updates)

    # Sampling params: Show some samples during training
    SAMPLE_ON_SETS = []                           # Possible values: 'train', 'val' and 'test'
    N_SAMPLES = 5                                 # Number of samples generated
    START_SAMPLING_ON_EPOCH = 1                   # First epoch where the model will be evaluated
    SAMPLE_EACH_UPDATES = 2500                    # Sampling frequency (default 450)


    # Early stop parameters
    EARLY_STOP = True                  # Turns on/off the early stop protocol
    PATIENCE = 15                      # We'll stop if the val STOP_METRIC does not improve after this number of evaluations
    STOP_METRIC = 'accuracy'           # Metric for the stop

    # Word representation params
    TOKENIZATION_METHOD = 'tokenize_none'         # Select which tokenization we'll apply:
                                                  # tokenize_basic, tokenize_aggressive, tokenize_soft,
                                                  # tokenize_icann or tokenize_questions
    # Input image parameters
    DATA_AUGMENTATION = False                     # Apply data augmentation on input data (noise on features)

    # Input text parameters
    VOCABULARY_SIZE = 0        # Size of the input vocabulary. Set to 0 for using all, otherwise will be truncated to these most frequent words.
    MIN_OCCURRENCES_VOCAB = 0  # Minimum number of occurrences allowed for the words in the vocabulay. Set to 0 for using them all.
    MAX_INPUT_TEXT_LEN = 50    # Maximum length of the input sequence

    CLASSIFIER_ACTIVATION = 'softmax'

    # Optimizer parameters (see model.compile() function)
    LOSS = 'categorical_crossentropy'
    CLASS_MODE = 'categorical'

    OPTIMIZER = 'Adam'      # Optimizer
    LR = 0.0001              # (recommended values - Adam 0.001 - Adadelta 1.0
    WEIGHT_DECAY = 1e-4     # L2 regularization
    CLIP_C = 10.            # During training, clip gradients to this norm
    SAMPLE_WEIGHTS = False  # Select whether we use a weights matrix (mask) for the data outputs

    # Training parameters
    MAX_EPOCH = 500         # Stop when computed this number of epochs
    BATCH_SIZE = 128        #  Training batch size

    HOMOGENEOUS_BATCHES = False  # Use batches with homogeneous output lengths for every minibatch (Dangerous!)
    PARALLEL_LOADERS = 8         # Parallel data batch loaders
    EPOCHS_FOR_SAVE = 1          # Number of epochs between model saves
    WRITE_VALID_SAMPLES = True   # Write valid samples in file


    # Input text parameters
    INPUT_VOCABULARY_SIZE = 0         # Size of the input vocabulary. Set to 0 for using all, otherwise will be truncated to these most frequent words.
    MIN_OCCURRENCES_VOCAB = 0  # Minimum number of occurrences allowed for the words in the vocabulay. Set to 0 for using them all.
    PAD_ON_BATCH = False

    # Output classes parameters
    N_CLASSES = 2

    # Model parameters
    MODEL_TYPE = 'CNN_Classifier'

    GLOVE_VECTORS = None #'/media/HDD_2TB/DATASETS/VQA/Glove/glove_300.npy'  # Path to pretrained vectors. Set to None if you don't want to use pretrained vectors.
    GLOVE_VECTORS_TRAINABLE = True    # Finetune or not the word embedding vectors.
    TEXT_EMBEDDING_HIDDEN_SIZE = 300  # When using pretrained word embeddings, this parameter must match with the word embeddings size

    # LSTM layers dimensions (Only used if needed)
    LSTM_ENCODER_HIDDEN_SIZE = 289   # For models with LSTM encoder

    # FC layers for initializing the first LSTM state
    # Here we should only specify the activation function of each layer (as they have a potentially fixed size)
    # (e.g INIT_LAYERS = ['tanh', 'relu'])
    INIT_LAYERS = ['tanh']



    # CNN layers parameters (Only used if needed)
    NUM_FILTERS = 200
    FILTER_SIZES = [3,4,5]
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
    DEEP_OUTPUT_LAYERS = [('linear', 200), ('linear', 100)]


    # Regularizers / Normalizers
    USE_DROPOUT = True                  # Use dropout (0.5)
    USE_BATCH_NORMALIZATION = False     # If True it is recommended to deactivate Dropout
    USE_PRELU = False                   # use PReLU activations
    USE_L2 = False                      # L2 normalization on the features

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
    MODE = 'training'                                  # 'training' or 'sampling' (if 'sampling' then RELOAD must
                                                       # be greater than 0 and EVAL_ON_SETS will be used)

    # Extra parameters for special trainings
    TRAIN_ON_TRAINVAL = False                          # train the model on both training and validation sets combined
    FORCE_RELOAD_VOCABULARY = False                    # force building a new vocabulary from the training samples
                                                       # applicable if RELOAD > 1

    # ============================================
    parameters = locals().copy()
    return parameters
