import matplotlib.pyplot as plt
#from scipy.io import loadmat
import numpy as np
#import pandas
from os.path import join,exists
from os import mkdir
import cv2
import math
import os
from scipy.spatial import distance

import time

NOMBRE_DE_SEQUENCES = 11935

data = []
for i in range(NOMBRE_DE_SEQUENCES):
    file = join('../Train', 'train_3', str(i) + '.txt')
    f = open(file, 'r').read().split()
    datait = [float(x) for x in f]
    datait = datait[2:]
    dataitem = []
    for i in range(len(datait)):
        if i%4 != 0:
            dataitem.append(datait[i])
    data_seq = np.asarray(dataitem)
    data.append(data_seq.reshape((1000, 3)))
np.save('train_embedding_1000_3', data)

def to_category(actions, labels):
    categories = []
    y = []
    for i in range(len(actions)):
        cat_vec = np.zeros(10)
        cat_vec[i] = 1
        categories.append(cat_vec)
    for label in labels:
        indx = actions.index(label)
        y.append(categories[indx])
    return y


