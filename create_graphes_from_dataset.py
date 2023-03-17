
import numpy as np
from scipy.spatial import distance

import time
from sklearn.model_selection import train_test_split

# CONSTANTS
intraframe_edges = [
    (0, 1), (0, 12), (0, 16),
    (1, 20),
    (2, 3), (2, 20),
    (4, 20), (4, 5),
    (5, 6),
    (6, 7), (6, 22),
    (7, 21),
    (8, 20), (8, 9),
    (9, 10),
    (10, 11), (10, 24),
    (11, 23),
    (12, 13),
    (13, 14),
    (14, 15),
    (16, 17),
    (17, 18),
    (18, 19)
]
joint_names = [
    "SPINBASE",
    "SPINMID",
    "NECK",
    "HEAD",
    "SHOULDERLEFT",
    "ELBOWLEFT",
    "WRISTLEFT",
    "HANDLEFT",
    "SOULDERRIGHT",
    "ELBOWRIGHT",
    "WRISTRIGHT",
    "HANDRIGHT",
    "HIPLEFT",
    "KNEELEFT",
    "ANKLELEFT",
    "FOOTLEFT",
    "HIPRIGHT",
    "KNEERIGHT",
    "ANKLERIGHT",
    "FOOTTIGHT",
    "SPINSHOULDER",
    "HANDTIPLEFT",
    "THUMBLEFT",
    "HAN0DTIPRIGHT",
    "THUMBRIGHT"
]
NODE_PER_FRAME = 25

DATA_PATH = 'C:\\Users\\Mohamed\\PycharmProjects\\DATA'

train_path = DATA_PATH+'/Train'
test_path = DATA_PATH+'/Test'

X_train_file = train_path + "/x_train"
y_train_file = train_path + "/y_train"
X_test_file = test_path + "/x_test"
y_test_file = test_path + "/y_test"

def recharger_train_data():
    X, y = np.load(X_train_file + '.npy'), np.load(y_train_file + '.npy')
    return X, y

def recharger_test_data():
    X, y = np.load(X_test_file + '.npy'), np.load(y_test_file + '.npy')
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.1, random_state=123)

import networkx as nx
def transform_graph(graph_data, fc = True):
    if fc:
        frame = 0
        graphe = nx.Graph()
        for data in graph_data:
            # connexion intra-frame
            for u, v in intraframe_edges:
                w = distance.euclidean(data[u], data[v])
                graphe.add_edge((NODE_PER_FRAME * frame) + u, (NODE_PER_FRAME * frame) + v, weight=w)
            for i in range(NODE_PER_FRAME):
                for j in range(NODE_PER_FRAME):
                    if i != j:
                        w = distance.euclidean(data[i], data[j])
                        graphe.add_edge((NODE_PER_FRAME * frame) + i, (NODE_PER_FRAME * frame) + j, weight=w)

            if frame > 0:
                # connect every joint to itself in t+1
                for i in range(NODE_PER_FRAME):
                    u = i + frame * NODE_PER_FRAME
                    v = i + (frame - 1) * NODE_PER_FRAME
                    w = distance.euclidean(data_t0[i], data[i])
                    graphe.add_edge(u, v, weight=w)
            data_t0 = data
            frame += 1
    else:
        frame = 0
        graphe = nx.Graph()
        for data in graph_data:
            # connexion intra-frame
            for u, v in intraframe_edges:
                w = distance.euclidean(data[u], data[v])
                graphe.add_edge((NODE_PER_FRAME * frame) + u, (NODE_PER_FRAME * frame) + v, weight=w)
            if frame > 0:
                # connect every joint to itself in t+1
                for i in range(NODE_PER_FRAME):
                    u = i + frame * NODE_PER_FRAME
                    v = i + (frame - 1) * NODE_PER_FRAME
                    w = distance.euclidean(data_t0[i], data[i])
                    graphe.add_edge(u, v, weight=w)
            data_t0 = data
            frame += 1

    print("graph created with", len(graphe.nodes), "and", len(graphe.edges), "edges.")
    return graphe

def create_graphe_dataset():
    X_train, y_train = recharger_train_data()
    X_test, y_test = recharger_test_data()
    print("Creating graphs for training subset: ")
    for i in range(len(y_train)):
        start = time.time()
        nx.write_edgelist(transform_graph(X_train[i], True), DATA_PATH+'/graphes/train/graphe_'+str(i))
        print("Create", i, "Time: ", time.time()-start)

    print("Creating graphs for testing subset: ")
    for i in range(len(y_test)):
        start = time.time()
        nx.write_edgelist(transform_graph(X_test[i], True), DATA_PATH+'/graphes/test/graphe_' + str(i))
        print("Create", i, "Time: ", time.time()-start)

"""

X,y = recharger_train_data()
X_train, X_val, y_train, y_val = split_data(X, y)
np.save(DATA_PATH+"/X_train_split", X_train)
np.save(DATA_PATH+"/y_train_split", y_train)
np.save(DATA_PATH+"/X_test_split", X_val)
np.save(DATA_PATH+"/y_test_split", y_val)

file_path = DATA_PATH + '/Train/graph_10percent_train'
np.save(DATA_PATH+"/Train/y_10percent_train", y_val)
"""

import pickle

def save_as_pickle(X_train, graph_path):
    t0 = time.time()
    file = open(graph_path, 'wb')
    graphes = []
    for elm in X_train:
        graphes.append(transform_graph(elm))
        print("Time:", time.time()-t0)
    pickle.dump(graphes, file)

X_train = np.load('10percentOfX_train.npy')
y_train = np.load('10percentOfy_train.npy')
save_as_pickle(X_train, "10percentOfGraph_train")