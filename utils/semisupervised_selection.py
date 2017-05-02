import numpy as np


def process_prediction_probs(prediction_probs, n_intances_to_add, pool_src, pool_trg):
    probs = np.array([], dtype="float32")
    for batch in prediction_probs:
        probs = np.append(probs, batch)
    probs = probs.reshape(-1, 2)
    top_positive_positions = probs.argsort(axis=0)[:, 0][:n_intances_to_add]
    top_negative_positions = probs.argsort(axis=0)[:, 1][:n_intances_to_add]
    positive_lines_src = []
    positive_lines_trg = []
    negative_lines_src = []
    negative_lines_trg = []
    neutral_lines_src = []
    neutral_lines_trg = []

    pool_file_src = open(pool_src)
    pool_file_trg = open(pool_trg)

    for i, (line_src, line_trg) in enumerate(zip(pool_file_src, pool_file_trg)):
        if i in top_negative_positions:
            negative_lines_src.append(line_src)
            negative_lines_trg.append(line_trg)
        elif i in top_positive_positions:
            positive_lines_src.append(line_src)
            positive_lines_trg.append(line_trg)
        else:
            neutral_lines_src.append(line_src)
            neutral_lines_trg.append(line_trg)

    pool_file_src.close()
    pool_file_trg.close()

    return positive_lines_src, positive_lines_trg, negative_lines_src, negative_lines_trg, neutral_lines_src, neutral_lines_trg


def update_config_params(params,
                         pos_filename,
                         neg_filename,
                         pool_filename):
    params['POSITIVE_FILENAME'] = pos_filename
    params['NEGATIVE_FILENAME'] = neg_filename
    params['POOL_FILENAME'] = pool_filename

    return params


def process_files_binary_classification(params, i=0):
    if i == 0:
        for (split, filename) in params['TEXT_FILES'].iteritems():
            params['TEXT_FILES'][split] = params['DATA_ROOT_PATH'] + '/' + filename + '.' + params['SRC_LAN']
            if params['BILINGUAL_SELECTION']:
                params['TEXT_FILES'][split].append(params['DATA_ROOT_PATH'] + '/' + filename + '.' + params['TRG_LAN'])
        for (split, filename) in params['CLASS_FILES'].iteritems():
            params['CLASS_FILES'][split] = params['DATA_ROOT_PATH'] + '/' + filename

    if params['BINARY_SELECTION']:
        pos_filename_src = params['POSITIVE_FILENAME'] + '.' + params['SRC_LAN']
        neg_filename_src = params['NEGATIVE_FILENAME'] + '.' + params['SRC_LAN']

        if params['BILINGUAL_SELECTION']:
            pos_filename_trg = params['POSITIVE_FILENAME'] + '.' + params['TRG_LAN']
            neg_filename_trg = params['NEGATIVE_FILENAME'] + '.' + params['TRG_LAN']

        dest_sentences_src_filename = params['DEST_ROOT_PATH'] + '/training_pos_neg_' + str(i) + '_tmp' + '.' + params[
            'SRC_LAN']
        dest_sentences_file_src = open(dest_sentences_src_filename, 'w')

        if params['BILINGUAL_SELECTION']:
            dest_sentences_trg_filename = params['DEST_ROOT_PATH'] + '/training_pos_neg_' + str(i) + '_tmp' + '.' + \
                                          params['TRG_LAN']
            dest_sentences_trg_file = open(dest_sentences_trg_filename, 'w')

        dest_classes_filename = params['DEST_ROOT_PATH'] + '/training_pos_neg_%d_tmp.class' % i

        dest_classes_file = open(dest_classes_filename, 'w')
        positive_file_src = open(pos_filename_src, 'r')
        for line in positive_file_src:
            dest_sentences_file_src.write(line)
            dest_classes_file.write('1\n')
        negative_file_src = open(neg_filename_src, 'r')
        for line in negative_file_src:
            dest_sentences_file_src.write(line)
            dest_classes_file.write('0\n')

        if params['BILINGUAL_SELECTION']:
            positive_file_trg = open(pos_filename_trg, 'r')
            for line in positive_file_trg:
                dest_sentences_trg_file.write(line)
            negative_file_trg = open(neg_filename_trg, 'r')
            for line in negative_file_trg:
                dest_sentences_trg_file.write(line)
            dest_sentences_trg_file.close()
            positive_file_trg.close()
            negative_file_trg.close()

        positive_file_src.close()
        negative_file_src.close()
        dest_sentences_file_src.close()
        dest_classes_file.close()
        params['POOL_FILENAME'] = [params['POOL_FILENAME'] + '.' + params['SRC_LAN'],
                                   params['POOL_FILENAME'] + '.' + params['TRG_LAN']] if params['BILINGUAL_SELECTION'] \
            else [params['POOL_FILENAME'] + '.' + params['TRG_LAN']]
        params['TEXT_FILES']['train'] = [dest_sentences_src_filename,
                                         dest_sentences_trg_filename] if params['BILINGUAL_SELECTION'] \
            else [dest_sentences_src_filename]
        params['CLASS_FILES']['train'] = dest_classes_filename

    return params
