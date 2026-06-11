#!/usr/bin/env python3

import numpy as np
import os

create_confusion_matrix = __import__('0-create_confusion').create_confusion_matrix

if __name__ == '__main__':

    # always load relative to script location
    base = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base, 'labels_logits.npz')

    lib = np.load(file_path)
    labels = lib['labels']
    logits = lib['logits']

    np.set_printoptions(suppress=True)

    confusion = create_confusion_matrix(labels, logits)
    print(confusion)

    np.savez_compressed('confusion.npz', confusion=confusion)