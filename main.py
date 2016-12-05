import logging
from timeit import default_timer as timer

from config import load_parameters
from data_engine.prepare_data import build_dataset
from model_zoo import Text_Classification_Model
from keras_wrapper.cnn_model import loadModel
from utils.semisupervised_selection import process_prediction_probs, update_config_params, process_files_binary_classification
import utils
import sys
import ast
from shutil import copyfile, rmtree
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def train_model(params):
    """
        Main function
    """

    if(params['RELOAD'] > 0):
        logging.info('Resuming training.')

    check_params(params)

    ########### Load data
    if params['BINARY_SELECTION']:
        params['POSITIVE_FILENAME'] = params['DATA_ROOT_PATH'] + '/' + params['POSITIVE_FILENAME']
        params['NEGATIVE_FILENAME'] = params['DATA_ROOT_PATH'] + '/' + params['NEGATIVE_FILENAME']
    params = process_files_binary_classification(params)
    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    ###########

    ########### Build model
    if(params['RELOAD'] == 0): # build new model
        text_class_model = Text_Classification_Model(params, type=params['MODEL_TYPE'], verbose=params['VERBOSE'],
                                              model_name=params['MODEL_NAME'], vocabularies=dataset.vocabulary,
                                              store_path=params['STORE_PATH'])

        # Define the inputs and outputs mapping from our Dataset instance to our model
        inputMapping = dict()
        for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
            pos_source = dataset.ids_inputs.index(id_in)
            id_dest = text_class_model.ids_inputs[i]
            inputMapping[id_dest] = pos_source
        text_class_model.setInputsMapping(inputMapping)

        outputMapping = dict()
        for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
            pos_target = dataset.ids_outputs.index(id_out)
            id_dest = text_class_model.ids_outputs[i]
            outputMapping[id_dest] = pos_target
        text_class_model.setOutputsMapping(outputMapping)

    else: # resume from previously trained model
        text_class_model = loadModel(params['STORE_PATH'], params['RELOAD'])
        text_class_model.setOptimizer()
    ###########


    ########### Callbacks
    callbacks = buildCallbacks(params, text_class_model, dataset)
    ###########


    ########### Training
    total_start_time = timer()

    logger.debug('Starting training!')
    training_params = {'n_epochs': params['MAX_EPOCH'],
                       'batch_size': params['BATCH_SIZE'],
                       'homogeneous_batches':params['HOMOGENEOUS_BATCHES'],
                       'shuffle': True,
                       'epochs_for_save': params['EPOCHS_FOR_SAVE'],
                       'verbose': params['VERBOSE'],
                       'eval_on_sets': params['EVAL_ON_SETS_KERAS'],
                       'n_parallel_loaders': params['PARALLEL_LOADERS'],
                       'extra_callbacks': callbacks,
                       'reload_epoch': params['RELOAD'],
                       'data_augmentation': params['DATA_AUGMENTATION']}
    text_class_model.trainNet(dataset, training_params)


    total_end_time = timer()
    time_difference = total_end_time - total_start_time
    logging.info('In total is {0:.2f}s = {1:.2f}m'.format(time_difference, time_difference / 60.0))
    ###########


def apply_Clas_model(params):
    """
        Function for using a previously trained model for sampling.
    """

    ########### Load data
    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET'][0]]
    ###########


    ########### Load model
    text_class_model = loadModel(params['STORE_PATH'], params['RELOAD'])
    text_class_model.setOptimizer()
    ###########


    ########### Apply sampling
    extra_vars = dict()
    extra_vars['tokenize_f'] = eval('dataset.' + params['TOKENIZATION_METHOD'])
    for s in params["EVAL_ON_SETS"]:

        # Apply model predictions
        params_prediction = {'batch_size': params['BATCH_SIZE'],
                             'n_parallel_loaders': params['PARALLEL_LOADERS'], 'predict_on_sets': [s]}

        predictions = text_class_model.predictNet(dataset, params_prediction)[s]

        # Store result
        filepath = text_class_model.model_path+'/'+ s +'.pred' # results file
        if params['SAMPLING_SAVE_MODE'] == 'list':
            utils.read_write.list2file(filepath, predictions)
        else:
            raise Exception, 'Only "list" is allowed in "SAMPLING_SAVE_MODE"'


        # Evaluate if any metric in params['METRICS']
        for metric in params['METRICS']:
            logging.info('Evaluating on metric ' + metric)
            filepath = text_class_model.model_path + '/' + s + '_sampling.' + metric  # results file

            # Evaluate on the chosen metric
            extra_vars[s] = dict()
            extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
            metrics = utils.evaluation.select[metric](
                pred_list=predictions,
                verbose=1,
                extra_vars=extra_vars,
                split=s)

            # Print results to file
            with open(filepath, 'w') as f:
                header = ''
                line = ''
                for metric_ in sorted(metrics):
                    value = metrics[metric_]
                    header += metric_ + ','
                    line += str(value) + ','
                f.write(header + '\n')
                f.write(line + '\n')
            logging.info('Done evaluating on metric ' + metric)


def semisupervised_selection(params):
    check_params(params)
    initial_pos_filename = params['POSITIVE_FILENAME']
    initial_neg_filename = params['NEGATIVE_FILENAME']
    initial_pool_filename = params['POOL_FILENAME']

    pos_filename_src = params['DATA_ROOT_PATH'] + '/' + initial_pos_filename+ '.' + params['SRC_LAN']
    in_domain_file = open(pos_filename_src, 'r')
    in_domain = in_domain_file.readlines()
    in_domain_file.close()

    neg_filename_src = params['DATA_ROOT_PATH'] + '/' + initial_neg_filename + '.' + params['SRC_LAN']
    pos_filename_trg = params['DATA_ROOT_PATH'] + '/' + initial_pos_filename + '.' + params['TRG_LAN']

    pool_filename_src = params['DATA_ROOT_PATH'] + '/' + initial_pool_filename + '.' + params['SRC_LAN']
    pool_filename_trg = params['DATA_ROOT_PATH'] + '/' + initial_pool_filename + '.' + params['TRG_LAN']


    for i in range(params['N_ITER']):
        print "------------------ Starting iteration", i, "------------------"
        new_pos_filename_src = params['DEST_ROOT_PATH'] + '/' + initial_pos_filename + '_' + str(i) + '.' + params['SRC_LAN']
        new_pos_filename_trg = params['DEST_ROOT_PATH'] + '/' + initial_pos_filename + '_' + str(i) + '.' + params['TRG_LAN']
        new_pos_filename_src_tmp = params['DEST_ROOT_PATH'] + '/' + initial_pos_filename + '_' + 'temp' + '.' + params['SRC_LAN']

        if params['DEBUG']:
            new_neg_filename_src_tmp = params['DEST_ROOT_PATH'] + '/' + initial_neg_filename + '_' + 'temp' + '.' + params['SRC_LAN']
        new_neg_filename_src =  params['DEST_ROOT_PATH'] + '/' + initial_neg_filename + '_' +  str(i) + '.' + params['SRC_LAN']

        new_pool_filename_src = params['DEST_ROOT_PATH'] + '/' + initial_pool_filename + '_' + str(i) + '.' + params['SRC_LAN']
        new_pool_filename_trg = params['DEST_ROOT_PATH'] + '/' + initial_pool_filename + '_' + str(i) + '.' + params['TRG_LAN']

        if i > 0:
            copyfile(pos_filename_src, new_pos_filename_src_tmp)
            copyfile(pos_filename_src, new_pos_filename_src)
            copyfile(pos_filename_trg, new_pos_filename_trg)

        with open(new_pos_filename_src_tmp, "a") as f:
            for line in in_domain:
                f.write(line)

        copyfile(neg_filename_src, new_neg_filename_src)
        copyfile(pool_filename_src, new_pool_filename_src)
        copyfile(pool_filename_trg, new_pool_filename_trg)

        params = update_config_params(params, new_pos_filename_src_tmp, new_neg_filename_src, new_pool_filename_src)
        params = process_files_binary_classification(params, i=i)
        ########### Load data
        dataset = build_dataset(params)
        params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
        ###########

        ########### Build model
        text_class_model = Text_Classification_Model(params, type=params['MODEL_TYPE'], verbose=params['VERBOSE'],
                                              model_name=params['MODEL_NAME'], vocabularies=dataset.vocabulary,
                                              store_path=params['STORE_PATH'])

        # Define the inputs and outputs mapping from our Dataset instance to our model
        inputMapping = dict()
        for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
            pos_source = dataset.ids_inputs.index(id_in)
            id_dest = text_class_model.ids_inputs[i]
            inputMapping[id_dest] = pos_source
        text_class_model.setInputsMapping(inputMapping)

        outputMapping = dict()
        for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
            pos_target = dataset.ids_outputs.index(id_out)
            id_dest = text_class_model.ids_outputs[i]
            outputMapping[id_dest] = pos_target
        text_class_model.setOutputsMapping(outputMapping)

        ########### Callbacks
        callbacks = buildCallbacks(params, text_class_model, dataset)
        ###########

        ########### Training
        total_start_time = timer()

        logger.debug('Starting training!')
        training_params = {'n_epochs': params['MAX_EPOCH'], 'batch_size': params['BATCH_SIZE'],
                           'homogeneous_batches':params['HOMOGENEOUS_BATCHES'],
                           'shuffle': False if 'train' in params['EVAL_ON_SETS'] else True,
                           'epochs_for_save': params['EPOCHS_FOR_SAVE'], 'verbose': params['VERBOSE'],
                           'eval_on_sets': params['EVAL_ON_SETS_KERAS'], 'n_parallel_loaders': params['PARALLEL_LOADERS'],
                           'extra_callbacks': callbacks, 'reload_epoch': params['RELOAD'],
                           'data_augmentation': params['DATA_AUGMENTATION']}
        text_class_model.trainNet(dataset, training_params)
        total_end_time = timer()
        time_difference = total_end_time - total_start_time
        logging.info('In total is {0:.2f}s = {1:.2f}m'.format(time_difference, time_difference / 60.0))
        ###########


        # Apply model predictions
        params_prediction = {'batch_size': params['BATCH_SIZE'],
                             'n_parallel_loaders': params['PARALLEL_LOADERS'],
                             'predict_on_sets': ['test']}

        prediction_probs = text_class_model.predictNet(dataset, params_prediction)['test']
        positive_lines_src, positive_lines_trg, negative_lines, neutral_lines_src, neutral_lines_trg = \
        process_prediction_probs(prediction_probs, params['INSTANCES_TO_ADD'], pool_filename_src, pool_filename_trg)

        print "Adding", len(positive_lines_src), "positive lines"
        print "Positive sample:", positive_lines_src[0], "---", positive_lines_trg[0]
        print "Adding", len(negative_lines), "negative lines"
        print "Negative sample:", negative_lines[0]

        print "Adding", len(neutral_lines_src), "neutral lines"
        print "Neutral sample:", neutral_lines_src[0], "---", neutral_lines_trg[0]

        new_pos_file_src = open(new_pos_filename_src, 'a')
        new_pos_file_trg = open(new_pos_filename_trg, 'a')

        new_neg_file = open(new_neg_filename_src, 'a')
        if params['DEBUG']:
            new_neg_file_tmp = open(new_neg_filename_src_tmp, 'a')

        new_pool_file_src = open(new_pool_filename_src, 'w')
        new_pool_file_trg = open(new_pool_filename_trg, 'w')

        for line in positive_lines_src:
            new_pos_file_src.write(line)
        for line in positive_lines_trg:
            new_pos_file_trg.write(line)

        for line in negative_lines:
            new_neg_file.write(line)
            if params['DEBUG']:
                new_neg_file_tmp.write(line)
        for line in neutral_lines_src:
            new_pool_file_src.write(line)
        for line in neutral_lines_trg:
            new_pool_file_trg.write(line)

        new_pos_file_src.close()
        new_pos_file_trg.close()

        new_neg_file.close()

        new_pool_file_src.close()
        new_pool_file_trg.close()

        if params['DEBUG']:
            new_neg_file_tmp.close()

        pos_filename_src = new_pos_filename_src
        pos_filename_trg = new_pos_filename_trg

        neg_filename_src = new_neg_filename_src

        pool_filename_src = new_pool_filename_src
        pool_filename_trg = new_pool_filename_trg


def buildCallbacks(params, model, dataset):
    """
        Builds the selected set of callbacks run during the training of the model
    """

    callbacks = []

    if params['METRICS']:
        # Evaluate training
        extra_vars = {'n_parallel_loaders': params['PARALLEL_LOADERS']}
        for s in params['EVAL_ON_SETS']:
            extra_vars[s] = dict()
            extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
            if dataset.dic_classes.get(params['OUTPUTS_IDS_DATASET'][0]):
                extra_vars['n_classes'] = len(dataset.dic_classes[params['OUTPUTS_IDS_DATASET'][0]])

        if params['EVAL_EACH_EPOCHS']:
            callback_metric = utils.callbacks.PrintPerformanceMetricOnEpochEnd(model, dataset,
                                                           gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                                           metric_name=params['METRICS'],
                                                           set_name=params['EVAL_ON_SETS'],
                                                           batch_size=params['BATCH_SIZE'],
                                                           each_n_epochs=params['EVAL_EACH'],
                                                           extra_vars=extra_vars,
                                                           reload_epoch=params['RELOAD'],
                                                           save_path=model.model_path,
                                                           start_eval_on_epoch=params['START_EVAL_ON_EPOCH'],
                                                           write_samples=True,
                                                           write_type=params['SAMPLING_SAVE_MODE'],
                                                           early_stop=params['EARLY_STOP'],
                                                           patience=params['PATIENCE'],
                                                           stop_metric=params['STOP_METRIC'],
                                                           verbose=params['VERBOSE'])

        else:
            callback_metric = utils.callbacks.PrintPerformanceMetricEachNUpdates(model, dataset,
                                                           gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                                           metric_name=params['METRICS'],
                                                           set_name=params['EVAL_ON_SETS'],
                                                           batch_size=params['BATCH_SIZE'],
                                                           each_n_updates=params['EVAL_EACH'],
                                                           extra_vars=extra_vars,
                                                           reload_epoch=params['RELOAD'],
                                                           save_path=model.model_path,
                                                           start_eval_on_epoch=params['START_EVAL_ON_EPOCH'],
                                                           write_samples=True,
                                                           write_type=params['SAMPLING_SAVE_MODE'],
                                                           early_stop=params['EARLY_STOP'],
                                                           patience=params['PATIENCE'],
                                                           stop_metric=params['STOP_METRIC'],
                                                           verbose=params['VERBOSE'])
        callbacks.append(callback_metric)

    return callbacks

def check_params(params):
    if 'Glove' in params['MODEL_TYPE'] and params['GLOVE_VECTORS'] is None:
        logger.warning("You set a model that uses pretrained word vectors but you didn't specify a vector file."
                       "We'll train WITHOUT pretrained embeddings!")
    if params['MODE'] == 'semisupervised-selection' and not params['BINARY_SELECTION']:
        raise AttributeError, 'When MODE = %s, BINARY_SELECTION must be set to True'
if __name__ == "__main__":

    params = load_parameters()

    try:
        for arg in sys.argv[1:]:
            k, v = arg.split('=')
            params[k] = ast.literal_eval(v)
    except:
        print 'Overwritten arguments must have the form key=Value'
        exit(1)

    if(params['MODE'] == 'training'):
        logging.info('Running training.')
        utils.read_write.create_dir_if_not_exists(params['DEST_ROOT_PATH'])
        train_model(params)
    elif(params['MODE'] == 'sampling'):
        logging.info('Running sampling.')
        apply_Clas_model(params)
    elif(params['MODE'] == 'semisupervised-selection'):
        logging.info('Running semisuprevised selection.')
        utils.read_write.create_dir_if_not_exists(params['DEST_ROOT_PATH'])
        semisupervised_selection(params)

    logging.info('Done!')
