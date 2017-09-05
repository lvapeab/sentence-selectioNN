import ast
import logging
import sys
from shutil import copyfile
from timeit import default_timer as timer

from config import load_parameters
from data_engine.prepare_data import build_dataset
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.extra import evaluation, read_write
from keras_wrapper.extra.callbacks import PrintPerformanceMetricOnEpochEndOrEachNUpdates
from model_zoo import Text_Classification_Model
from utils.semisupervised_selection import process_prediction_probs, update_config_params, \
    process_files_binary_classification

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def train_model(params):
    """
        Main function
    """

    if params['RELOAD'] > 0:
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
    if params['RELOAD'] == 0:  # build new model
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

    else:  # resume from previously trained model
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
                       'homogeneous_batches': params['HOMOGENEOUS_BATCHES'],
                       'shuffle': True,
                       'epochs_for_save': params['EPOCHS_FOR_SAVE'],
                       'verbose': params['VERBOSE'],
                       'eval_on_sets': params['EVAL_ON_SETS_KERAS'],
                       'n_parallel_loaders': params['PARALLEL_LOADERS'],
                       'extra_callbacks': callbacks,
                       'reload_epoch': params['RELOAD'],
                       'data_augmentation': params.get('DATA_AUGMENTATION', False)}
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
    params['INPUT_SCR_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
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
        filepath = text_class_model.model_path + '/' + s + '.pred'  # results file
        if params['SAMPLING_SAVE_MODE'] == 'list':
            read_write.list2file(filepath, predictions)
        else:
            raise Exception, 'Only "list" is allowed in "SAMPLING_SAVE_MODE"'

        # Evaluate if any metric in params['METRICS']
        for metric in params['METRICS']:
            logging.info('Evaluating on metric ' + metric)
            filepath = text_class_model.model_path + '/' + s + '_sampling.' + metric  # results file

            # Evaluate on the chosen metric
            extra_vars[s] = dict()
            extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
            metrics = evaluation.select[metric](
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

    pos_filename = params['DATA_ROOT_PATH'] + '/' + initial_pos_filename
    in_domain_file_src = open(pos_filename + '.' + params['SRC_LAN'], 'r')
    in_domain_src = in_domain_file_src.readlines()
    in_domain_file_src.close()
    if params['BILINGUAL_SELECTION']:
        in_domain_file_trg = open(pos_filename + '.' + params['TRG_LAN'], 'r')
        in_domain_trg = in_domain_file_trg.readlines()
        in_domain_file_trg.close()

    neg_filename = params['DATA_ROOT_PATH'] + '/' + initial_neg_filename

    pool_filename = params['DATA_ROOT_PATH'] + '/' + initial_pool_filename

    for i in range(params['N_ITER']):
        print "------------------ Starting iteration", i, "------------------"
        new_pos_filename = params['DEST_ROOT_PATH'] + '/' + initial_pos_filename + '_' + str(i)
        new_pos_filename_tmp = params['DEST_ROOT_PATH'] + '/' + initial_pos_filename + '_' + 'temp'
        if params['DEBUG']:
            new_neg_filename_tmp = params['DEST_ROOT_PATH'] + '/' + initial_neg_filename + '_' + 'temp'
        new_neg_filename = params['DEST_ROOT_PATH'] + '/' + initial_neg_filename + '_' + str(i)

        new_pool_filename = params['DEST_ROOT_PATH'] + '/' + initial_pool_filename + '_' + str(i)

        if i > 0:
            copyfile(pos_filename + '.' + params['SRC_LAN'], new_pos_filename_tmp + '.' + params['SRC_LAN'])
            copyfile(pos_filename + '.' + params['SRC_LAN'], new_pos_filename + '.' + params['SRC_LAN'])
            copyfile(pos_filename + '.' + params['TRG_LAN'], new_pos_filename + '.' + params['TRG_LAN'])
            if params['BILINGUAL_SELECTION']:
                copyfile(pos_filename + '.' + params['TRG_LAN'], new_pos_filename_tmp + '.' + params['TRG_LAN'])

        with open(new_pos_filename_tmp + '.' + params['SRC_LAN'], "a") as f:
            for line in in_domain_src:
                f.write(line)

        if params['BILINGUAL_SELECTION']:
            with open(new_pos_filename_tmp + '.' + params['TRG_LAN'], "a") as f:
                for line in in_domain_trg:
                    f.write(line)

        copyfile(neg_filename + '.' + params['SRC_LAN'], new_neg_filename + '.' + params['SRC_LAN'])
        if params['BILINGUAL_SELECTION'] or params['DEBUG']:
            copyfile(neg_filename + '.' + params['TRG_LAN'], new_neg_filename + '.' + params['TRG_LAN'])
        copyfile(pool_filename + '.' + params['SRC_LAN'], new_pool_filename + '.' + params['SRC_LAN'])
        copyfile(pool_filename + '.' + params['TRG_LAN'], new_pool_filename + '.' + params['TRG_LAN'])

        if params['BILINGUAL_SELECTION']:
            params = update_config_params(params,
                                          new_pos_filename_tmp,
                                          new_neg_filename,
                                          new_pool_filename)
        params = process_files_binary_classification(params, i=i)
        ########### Load data
        dataset = build_dataset(params)
        params['INPUT_SRC_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
        if params['BILINGUAL_SELECTION']:
            params['INPUT_TRG_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][1]]
        ###########

        ########### Build model
        text_class_model = Text_Classification_Model(params,
                                                     type=params['MODEL_TYPE'],
                                                     model_name=params['MODEL_NAME'],
                                                     vocabularies=dataset.vocabulary,
                                                     store_path=params['STORE_PATH'],
                                                     verbose=params['VERBOSE'])

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
                           'homogeneous_batches': params['HOMOGENEOUS_BATCHES'],
                           'shuffle': False if 'train' in params['EVAL_ON_SETS'] else True,
                           'epochs_for_save': params['EPOCHS_FOR_SAVE'], 'verbose': params['VERBOSE'],
                           'eval_on_sets': params['EVAL_ON_SETS_KERAS'],
                           'n_parallel_loaders': params['PARALLEL_LOADERS'],
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
        positive_lines_src, positive_lines_trg, negative_lines_src, negative_lines_trg, neutral_lines_src, neutral_lines_trg = \
            process_prediction_probs(prediction_probs, params['INSTANCES_TO_ADD'],
                                     pool_filename + '.' + params['SRC_LAN'],
                                     pool_filename + '.' + params['TRG_LAN'],
                                     verbose=params['VERBOSE'])

        print "Adding", len(positive_lines_src), "positive lines"
        print "Positive sample:", positive_lines_src[0], "---", positive_lines_trg[0]
        print "Adding", len(negative_lines_trg), "negative lines"
        print "Negative sample:", negative_lines_src[0], "---", negative_lines_trg[0]

        print "Adding", len(neutral_lines_src), "neutral lines"
        print "Neutral sample:", neutral_lines_src[0], "---", neutral_lines_trg[0]

        new_pos_file_src = open(new_pos_filename + '.' + params['SRC_LAN'], 'a')
        new_pos_file_trg = open(new_pos_filename + '.' + params['TRG_LAN'], 'a')

        new_neg_file_src = open(new_neg_filename + '.' + params['SRC_LAN'], 'a')
        new_neg_file_trg = open(new_neg_filename + '.' + params['TRG_LAN'], 'a')

        new_pool_file_src = open(new_pool_filename + '.' + params['SRC_LAN'], 'w')
        new_pool_file_trg = open(new_pool_filename + '.' + params['TRG_LAN'], 'w')

        for line in positive_lines_src:
            new_pos_file_src.write(line)
        for line in positive_lines_trg:
            new_pos_file_trg.write(line)

        for line in negative_lines_src:
            new_neg_file_src.write(line)
        for line in negative_lines_trg:
            new_neg_file_trg.write(line)
        for line in neutral_lines_src:
            new_pool_file_src.write(line)
        for line in neutral_lines_trg:
            new_pool_file_trg.write(line)

        new_pos_file_src.close()
        new_pos_file_trg.close()

        new_neg_file_src.close()
        new_neg_file_trg.close()

        new_pool_file_src.close()
        new_pool_file_trg.close()

        pos_filename = new_pos_filename
        neg_filename = new_neg_filename
        pool_filename = new_pool_filename

        if len(neutral_lines_src) < 2 * params['INSTANCES_TO_ADD']:
            logger.warning("We got out of neutral sentences (from the pool) to classify!. Stopping the process.")
            break


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
            callback_metric = PrintPerformanceMetricOnEpochEndOrEachNUpdates(model,
                                                                             dataset,
                                                                             gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                                                             metric_name=params['METRICS'],
                                                                             set_name=params['EVAL_ON_SETS'],
                                                                             batch_size=params['BATCH_SIZE'],
                                                                             each_n_epochs=params['EVAL_EACH'],
                                                                             extra_vars=extra_vars,
                                                                             reload_epoch=params['RELOAD'],
                                                                             save_path=model.model_path,
                                                                             start_eval_on_epoch=params[
                                                                                 'START_EVAL_ON_EPOCH'],
                                                                             write_samples=True,
                                                                             write_type=params['SAMPLING_SAVE_MODE'],
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
    read_write.clean_dir(params['DEST_ROOT_PATH'])
    if params['MODE'] == 'training':
        logging.info('Running training.')
        train_model(params)
    elif params['MODE'] == 'sampling':
        logging.info('Running sampling.')
        apply_Clas_model(params)
    elif params['MODE'] == 'semisupervised-selection':
        logging.info('Running semisupervised selection.')
        semisupervised_selection(params)

    logging.info('Done!')
