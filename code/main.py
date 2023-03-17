from os import mkdir
from os.path import join, exists

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from node2vec import Node2Vec

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

# importration des données
def get_data(file):
    try:
        f = open(file, 'r').read().split()
        datait = [float(x) for x in f]
        data = np.asarray(datait)
        data = data.reshape((NODE_PER_FRAME, 3))
        return data
    except:
        print('Ex', file)
        return None


def get_file(directory=0, file=55):
    return "data/" + str(directory) + "/skeleton/" + str(file) + ".txt"


def get_label_file(directory):
    return "data/" + str(directory) + "/label/label.txt"


def get_files_from(directory, start, end):
    filenames = [get_file(directory, i) for i in range(start, end+1)]
    return filenames


def get_labels(file):
    labels = open(file, 'r').read().splitlines()
    prev_action = None
    actions = []
    for line in labels:
        if line.replace(' ', '').isalpha():
            prev_action = line.strip()
        else:
            tab = line.split(' ')
            actions.append((int(tab[0]), int(tab[1]), prev_action))
    return actions


def get_graph_label(start,end,labels):
    index = (start+end)//2
    for s,e,a in labels:
        if s <= index and index <= e:
            return a
    return None


# la fonction permet de créer les frames et lier les noeuds entre frames cons
def clean_data(directory=0, window_size = 40):
    label_path = "data/"+str(directory)+"/label/label.txt"
    label_data = get_labels(label_path)
    start = min(label_data)[0] - window_size // 2
    end = max(label_data)[1] + window_size // 2
    files = get_files_from(directory, start, end)
    corpus = [get_data(file) for file in files]
    graph_data = [corpus[i:i + window_size] for i in range(len(corpus) - window_size + 1)]
    lab = [get_graph_label(i, i + window_size, label_data) for i in range(start, end - window_size + 2)]
    # delete non labeled sequences
    # print(len(graph_data), len(lab), len(corpus), window_size, len(corpus) - window_size + 1)
    i = 0
    while i <len(lab):
        if lab[i] is None:
            del lab[i]
            del graph_data[i]
        else:
            i += 1
    # delete sequences with jumps
    # print(len(graph_data), len(lab), len(corpus), window_size, len(corpus) - window_size + 1)
    i = 0
    while i < len(graph_data):
        jumped = False
        for x in graph_data[i]:
            if x is None:
                print(x is None)
            if x is None or not x.shape == (25,3):
                    del lab[i]
                    del graph_data[i]
                    jumped = True
                    break
        if not jumped:
            i+=1
    print(len(graph_data), len(lab), len(corpus), window_size, len(corpus) - window_size + 1)
    return graph_data, lab


def construct_graphes(graph_data):
    frame = 0
    graphe = nx.Graph()
    for data in graph_data:
        # connexion intra-frame
        for u, v in intraframe_edges:
            w = np.sqrt(sum(data[u] - data[v]) ** 2)
            graphe.add_edge((NODE_PER_FRAME * frame) + u, (NODE_PER_FRAME * frame) + v, weight=w)
        for i in range(NODE_PER_FRAME):
            graphe.nodes[(NODE_PER_FRAME * frame) + i]['name'] = joint_names[i]
            graphe.nodes[(NODE_PER_FRAME * frame) + i]['x'] = data[i][0]
            graphe.nodes[(NODE_PER_FRAME * frame) + i]['y'] = data[i][1]
            graphe.nodes[(NODE_PER_FRAME * frame) + i]['z'] = data[i][2]
        frame += 1
    # connect every joint to itself in t+1
    for i in range(window_size-1):
        for j in range(NODE_PER_FRAME):
            u = j + i * NODE_PER_FRAME
            v = j + (i + 1) * NODE_PER_FRAME
            w = np.sqrt(
                        (graphe.nodes[u]['x'] - graphe.nodes[v]['x'])**2
                        + (graphe.nodes[u]['y'] - graphe.nodes[v]['y'])**2
                        + (graphe.nodes[u]['z'] - graphe.nodes[v]['z'])**2
                        )
            graphe.add_edge(u, v, weight = w)
    print("total number of graph nodes: ", len(graphe.nodes))
    print("total number of graph edges: ", len(graphe.edges))
    return graphe


def viz_graph(graphe, edge_labels=False):
    pos = nx.kamada_kawai_layout(graphe)

    # nodes
    nx.draw_networkx_nodes(graphe, pos, node_size=100)

    # edges
    nx.draw_networkx_edges(
        graphe, pos, edgelist=graphe.edges, width=4
    )

    # node labels
    nx.draw_networkx_labels(
        graphe, pos, font_size=8, font_family="sans-serif"
    )
    if (edge_labels):
        # edge weight labels
        edge_labels = nx.get_edge_attributes(graphe, "weight")
        nx.draw_networkx_edge_labels(graphe, pos, edge_labels, font_size=6)

    plt.show()


def transform_graph(data_path):#,label_path,out_path):
    skeleton_data, label = clean_data(data_path, window_size=40)
    X = []
    y = []
    for i in range(len(label)):
        print(" <========== ",i," ===========> ")
        graphe = construct_graphes(skeleton_data[i])
        node2vec = Node2Vec(graphe, dimensions=64, walk_length=30, num_walks=20, workers=8)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        X.append(model.wv)
        y.append(label[i])
    return X, y


data_path = '../data'
train_path = "Train_node2vec"
test_path = "Test_node2vec"
train_sub = [1, 2, 3, 4, 7, 8, 9, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 32, 33, 34, 35, 37, 38, 39, 49, 50, 51, 54, 57, 58]
test_sub  = [0, 10, 13, 17, 21, 26, 27, 28, 29, 36, 40, 41, 42, 43, 44, 45, 52, 53, 55, 56]
window_size = 40;

if not exists(train_path):
    mkdir(train_path)
if not exists(test_path):
    mkdir(test_path)

"""dataset = []
for k in range(58):
    X, y = transform_graph(k)
    for i in range(len(X)):
        data = []
        target = y[i]
        for j in range(len(X[i])):
            data.append(X[i][j])
        x_y = (data, target)
        dataset.append(x_y)"""

X_train_file = "x_train"
y_train_file = "y_train"
X_test_file = "x_test"
y_test_file = "y_test"
def recharger_train_data():
    X, y = np.load(X_train_file + '.npy'), np.load(y_train_file + '.npy')
    return X, y
def recharger_test_data():
    X, y = np.load(X_test_file + '.npy'), np.load(y_test_file + '.npy')
    return X, y

