import pathlib
import os.path as osp
import pandas as pd
import numpy as np
from ast import literal_eval

from .vocabulary import build_vocab, Vocabulary
from dataset.in_out.data_generation.nr3d import decode_stimulus_string
from six.moves import cPickle

def read_lines(file_name):
    trimmed_lines = []
    with open(file_name) as fin:
        for line in fin:
            trimmed_lines.append(line.rstrip())
    return trimmed_lines

def unpickle_data(file_name, python2_to_3=False):
    """
    Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: an generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            
            import sys
            import dataset.in_out as in_out

            sys.modules['referit3d'] = sys.modules[__name__]
            sys.modules['referit3d.in_out'] = in_out
            yield cPickle.load(in_file)
    in_file.close()

def scannet_official_train_val(valid_views=None, verbose=True):
    """
    :param valid_views: None or list like ['00', '01']
    :return:
    """
    pre_fix = osp.split(pathlib.Path(__file__).parent.absolute())[0]
    train_split = osp.join(pre_fix, 'data/scannet/splits/official/v2/scannetv2_train.txt')
    train_split = read_lines(train_split)
    test_split = osp.join(pre_fix, 'data/scannet/splits/official/v2/scannetv2_val.txt')
    test_split = read_lines(test_split)

    if valid_views is not None:
        train_split = [sc for sc in train_split if sc[-2:] in valid_views]
        test_split = [sc for sc in test_split if sc[-2:] in valid_views]

    if verbose:
        print('#train/test scans:', len(train_split), '/', len(test_split))

    scans_split = dict()
    scans_split['train'] = set(train_split)
    scans_split['test'] = set(test_split)
    return scans_split


def objects_counter_percentile(scan_ids, all_scans, prc):
    all_obs_len = list()
    for scan_id in all_scans:
        if scan_id in scan_ids:
            all_obs_len.append(len(all_scans[scan_id].three_d_objects))
    return np.percentile(all_obs_len, prc)


def mean_color(scan_ids, all_scans):
    mean_rgb = np.zeros((1, 3), dtype=np.float32)
    n_points = 0
    for scan_id in scan_ids:
        color = all_scans[scan_id].color
        mean_rgb += np.sum(color, axis=0)
        n_points += len(color)
    mean_rgb /= n_points
    return mean_rgb


def load_referential_data(referit_csv, scans_split):
    """
    :param config:
    :param referit_csv:
    :param scans_split:
    :return:
    """
    referit_data = pd.read_csv(referit_csv)

    if True: #config.mentions_target_class_only:
        n_original = len(referit_data)
        referit_data = referit_data[referit_data['mentions_target_class']]
        referit_data.reset_index(drop=True, inplace=True)
        print('Dropping utterances without explicit '
              'mention to the target class {}->{}'.format(n_original, len(referit_data)))

    referit_data = referit_data[['tokens', 'instance_type', 'scan_id',
                                 'dataset', 'target_id', 'utterance', 'stimulus_id', 'anchor_ids', 'distractor_ids', 'need_distractors']]
    referit_data.tokens = referit_data['tokens'].apply(literal_eval)

    # Add the is_train data to the pandas data frame (needed in creating data loaders for the train and test)
    is_train = referit_data.scan_id.apply(lambda x: x in scans_split['train'])
    referit_data['is_train'] = is_train

    # Trim data based on token length
    train_token_lens = referit_data.tokens[is_train].apply(lambda x: len(x))
    print('{}-th percentile of token length for remaining (training) data'
          ' is: {:.1f}'.format(95, np.percentile(train_token_lens, 95)))
    n_original = len(referit_data)
    referit_data = referit_data[referit_data.tokens.apply(lambda x: len(x) <= 24)] # config.max_seq_len
    referit_data.reset_index(drop=True, inplace=True)
    print('Dropping utterances with more than {} tokens, {}->{}'.format(24, n_original, len(referit_data))) # config.max_seq_len

    
    context_size = referit_data[~referit_data.is_train].stimulus_id.apply(lambda x: decode_stimulus_string(x)[2])
    print('(mean) Random guessing among target-class test objects {:.4f}'.format( (1 / context_size).mean() ))

    return referit_data


def load_scan_related_data(preprocessed_scannet_file, verbose=True, add_pad=True):
    _, all_scans = unpickle_data(preprocessed_scannet_file)
    if verbose:
        print('Loaded in RAM {} scans'.format(len(all_scans)))

    instance_labels = set()
    for scan in all_scans:
        idx = np.array([o.object_id for o in scan.three_d_objects])
        instance_labels.update([o.instance_label for o in scan.three_d_objects])
        assert np.all(idx == np.arange(len(idx)))  # assert the list of objects-ids -is- the range(n_objects).
                                                   # because we use this ordering when we sample objects from a scan.
    all_scans = {scan.scan_id: scan for scan in all_scans}  # place scans in dictionary


    class_to_idx = {}
    i = 0
    for el in sorted(instance_labels):
        class_to_idx[el] = i
        i += 1

    if verbose:
        print('{} instance classes exist in these scans'.format(len(class_to_idx)))

    # Add the pad class needed for object classification
    if add_pad:
        class_to_idx['pad'] = len(class_to_idx)

    scans_split = scannet_official_train_val()

    return all_scans, scans_split, class_to_idx


def compute_auxiliary_data(referit_data, all_scans):
    """Given a train-split compute useful quantities like mean-rgb, a word-vocabulary.
    :param referit_data: pandas Dataframe, as returned from load_referential_data()
    :param all_scans:
    :param config:
    :return:
    """
    # Vocabulary
    if False: #config.vocab_file:
        vocab = Vocabulary.load(config.vocab_file)
        print(('Using external, provided vocabulary with {} words.'.format(len(vocab))))
    else:
        train_tokens = referit_data[referit_data.is_train]['tokens']
        vocab = build_vocab([x for x in train_tokens], 3) # config.min_word_freq
        print(('Length of vocabulary, with min_word_freq={} is {}'.format(3, len(vocab)))) #config.min_word_freq

    if all_scans is None:
        return vocab

    # Mean RGB for the training
    training_scan_ids = set(referit_data[referit_data['is_train']]['scan_id'])
    print('{} training scans will be used.'.format(len(training_scan_ids)))
    mean_rgb = mean_color(training_scan_ids, all_scans)

    # Percentile of number of objects in the training data
    prc = 90
    obj_cnt = objects_counter_percentile(training_scan_ids, all_scans, prc)
    print('{}-th percentile of number of objects in the (training) scans'
          ' is: {:.2f}'.format(prc, obj_cnt))

    # Percentile of number of objects in the testing data
    prc = 99
    testing_scan_ids = set(referit_data[~referit_data['is_train']]['scan_id'])
    obj_cnt = objects_counter_percentile(testing_scan_ids, all_scans, prc)
    print('{}-th percentile of number of objects in the (testing) scans'
          ' is: {:.2f}'.format(prc, obj_cnt))
    return mean_rgb, vocab


def trim_scans_per_referit3d_data(referit_data, scans):
    # remove scans not in referit_data
    in_r3d = referit_data.scan_id.unique()
    to_drop = []
    for k in scans:
        if k not in in_r3d:
            to_drop.append(k)
    for k in to_drop:
        del scans[k]
    print('Dropped {} scans to reduce mem-foot-print.'.format(len(to_drop)))
    return scans

