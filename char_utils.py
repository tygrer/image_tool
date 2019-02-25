import os
import numpy as np


charset = open('/home/gytang/project/rcnn/words/chinese_word_list_kuaidi.txt', encoding='utf8').read().strip('\r\n')
charset = [c for c in charset]
UNKNOWN_INDEX = len(charset)

UNKNOWN_TOKEN = chr(9617) # '░'
charset.append(UNKNOWN_TOKEN)
encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset):
    encode_maps[char] = i
    decode_maps[i] = char

BLANK_INDEX = len(charset)
BLANK_TOKEN = ''
decode_maps[BLANK_INDEX] = BLANK_TOKEN
num_classes = len(charset) + 1

def decode_label(label, ignore_value=-1):
    return(''.join([decode_maps[j] for j in label if j != ignore_value]))

def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=False):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        origin_label = decode_label(original_seq[i], ignore_value=ignore_value)
        decoded_label = decode_label(decoded_seq[i], ignore_value=ignore_value)
        if origin_label == decoded_label:
            count += 1

    return count * 1.0 / len(original_seq)


def char_accuracy_calculation(original_seq, decoded_seq, ignore_value=-1):
    import editdistance
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    total_char_num = 0
    total_distance = 0
    for i, origin_label in enumerate(original_seq):
        origin_label = decode_label(original_seq[i], ignore_value=ignore_value)
        decoded_label = decode_label(decoded_seq[i], ignore_value=ignore_value)
        total_char_num += len(origin_label)
        total_distance += editdistance.eval(origin_label, decoded_label)

    return 1 - (total_distance / total_char_num)


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def eval_expression(encoded_list):
    """
    :param encoded_list:
    :return:
    """

    eval_rs = []
    for item in encoded_list:
        try:
            rs = str(eval(item))
            eval_rs.append(rs)
        except:
            eval_rs.append(item)
            continue

    with open('./result.txt') as f:
        for ith in range(len(encoded_list)):
            f.write(encoded_list[ith] + ' ' + eval_rs[ith] + '\n')

    return eval_rs
