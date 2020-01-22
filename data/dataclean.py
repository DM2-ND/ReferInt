import re
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import nltk
from nltk import word_tokenize
import numpy as np


def normalize(s):

    s = re.sub(r"==\[.*\]==", r"cit", s)
    s = re.sub(r"-", r" ", s)
    s = re.sub(r"\[.*?\]", r" ", s)
    s = re.sub(r"([,.!?])", r" \1", s)
    s = s.strip()
    s = s.lower()

    s = ' '.join(word_tokenize(s))

    return s


f = open(os.path.join(BASE_DIR, 'data', 'citations.tsv'), 'r')
lines = [line.strip('\n').split('\t') for line in f][1:]
idx, cid, cid_list, cont, lb = [_ for _ in range(5)]

# train = open(os.path.join(BASE_DIR, 'data', 'academic_train'), 'w')
# test = open(os.path.join(BASE_DIR, 'data', 'academic_test'), 'w')

labels = []
for index, line in enumerate(lines):

    citation = normalize(line[cont])

    label = line[lb][:2]
    if label == 'ro': label = 'h-'
    if label == 'r-': label = 'h-'

    citing_id, cited_id = line[cid], line[cid_list]

    labels.append(label)
    # if index % 4 != 0:
    #     train.write('{}\t{}\t{}\t{}\n'.format(label, citing_id, cited_id, citation))
    # else:
    #     test.write('{}\t{}\t{}\t{}\n'.format(label, citing_id, cited_id, citation))

from collections import Counter
print(Counter(labels))