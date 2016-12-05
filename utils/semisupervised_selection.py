import numpy as np

def process_prediction_probs(prediction_probs, n_intances_to_add, pool_src, pool_trg):
    probs = np.array([], dtype="float32")
    for batch in prediction_probs:
        probs=np.append(probs, batch)
    probs = probs.reshape(-1, 2)
    top_positive_positions = probs.argsort(axis=0)[:, 0][:n_intances_to_add]
    top_negative_positions = probs.argsort(axis=0)[:, 1][:n_intances_to_add]
    positive_lines_src = []
    positive_lines_trg = []
    negative_lines = []
    neutral_lines_src = []
    neutral_lines_trg = []

    pool_file_src = open(pool_src)
    pool_file_trg = open(pool_trg)

    for i, (line_src, line_trg) in enumerate(zip (pool_file_src, pool_file_trg)):
        if i in top_negative_positions:
            negative_lines.append(line_src)
        elif i in top_positive_positions:
            positive_lines_src.append(line_src)
            positive_lines_trg.append(line_trg)
        else:
            neutral_lines_src.append(line_src)
            neutral_lines_trg.append(line_trg)

    pool_file_src.close()
    pool_file_trg.close()

    return positive_lines_src, positive_lines_trg, negative_lines, neutral_lines_src, neutral_lines_trg

def update_config_params(params,
                         pos_filename,
                         neg_filename,
                         pool_filename):
    params['POSITIVE_FILENAME'] = pos_filename
    params['NEGATIVE_FILENAME'] = neg_filename
    params['POOL_FILENAME'] = pool_filename

    return params


def process_files_binary_classification(params):
    for (split, filename) in params['TEXT_FILES'].iteritems():
        params['TEXT_FILES'][split] = params['DATA_ROOT_PATH'] + '/' + filename + '.' + params['SRC_LAN']
    for (split, filename) in params['CLASS_FILES'].iteritems():
        params['CLASS_FILES'][split] = params['DATA_ROOT_PATH'] + '/' + filename
    if params['BINARY_SELECTION']:
        pos_filename = params['POSITIVE_FILENAME'] + '.' + params['SRC_LAN']
        neg_filename = params['NEGATIVE_FILENAME'] + '.' + params['SRC_LAN']

        dest_sentences_filename = params['DEST_ROOT_PATH'] + '/training.' + params['SRC_LAN']
        dest_classes_filename = params['DEST_ROOT_PATH'] + '/training.class.' + params['SRC_LAN']
        dest_sentences_file = open(dest_sentences_filename, 'w')
        dest_classes_file = open(dest_classes_filename, 'w')

        positive_file = open(pos_filename, 'r')
        for line in positive_file:
            dest_sentences_file.write(line)
            dest_classes_file.write('1\n')
        negative_file = open(neg_filename, 'r')
        for line in negative_file:
            dest_sentences_file.write(line)
            dest_classes_file.write('0\n')

        positive_file.close()
        negative_file.close()
        dest_sentences_file.close()
        dest_classes_file.close()
        params['TEXT_FILES']['train'] = dest_sentences_filename
        params['CLASS_FILES']['train'] = dest_classes_filename

    return  params