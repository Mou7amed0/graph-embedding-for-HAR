

from os.path import exists
import networkx as nx
import time
import numpy as np
from node2vec import Node2Vec

DATA_PATH = 'C:\\Users\\Mohamed\\PycharmProjects\\DATA'
y_train = np.load('C:\\Users\\Mohamed\\PycharmProjects\\DATA/Train/y_train.npy')
y_test = np.load('C:\\Users\\Mohamed\\PycharmProjects\\DATA/Test/y_test.npy')
Train_set_size = len(y_train)
Test_set_size = len(y_test)

def embed_graph(graphe, dim=8, walk_l=25, walks=64): # , workers=1, window=10, min_count=1, batch_words=4):
    node2vec = Node2Vec(graphe, dimensions=dim, walk_length=walk_l, num_walks=walks)
    model = node2vec.fit()
    return model.wv.vectors


def load_train_graphes():
    train_graph_path = DATA_PATH + "/graphes/train/graphe_"
    X_train = []
    for i in range(Train_set_size):
        X_train.append(nx.read_edgelist(train_graph_path+str(i)))
    return X_train


def load_test_graphes():
    test_graph_path = DATA_PATH + "/graphes/test/graphe_"
    X_test = []
    for i in range(Test_set_size):
        X_test.append(nx.read_edgelist(test_graph_path+str(i)))
    return X_test

def save_embedding(train = True, dim = 4, walk_l = 6, walks = 8):
    embeddings_path = DATA_PATH+'/embeddings'
    train_graph_path = DATA_PATH + "/graphes/train/graphe_"
    test_graph_path = DATA_PATH + "/graphes/test/graphe_"
    t0 = time.time()
    if train:
        for i in range(Train_set_size):
            step_start = time.time()
            file = embeddings_path+'/train/emb_'+str(dim)+'dim_'+str(walk_l)+'len_'+str(walks)+'walks_'+str(i)
            if not exists(file+'.npy'):
                np.save(file,embed_graph(nx.read_edgelist(train_graph_path+str(i)), dim=dim, walk_l= walk_l, walks= walks))
            print(i, " ===> Time: ", time.time()-t0, " step time: ", time.time()-step_start)
    else:
        for i in range(Test_set_size):
            step_start = time.time()
            file = embeddings_path+'/test/emb_'+str(dim)+'dim_'+str(walk_l)+'len_'+str(walks)+'walks_'+str(i)

            if not exists(file+'.npy'):
                np.save(file,embed_graph(nx.read_edgelist(test_graph_path+str(i)), dim=dim, walk_l=walk_l, walks=walks))
            print(i, " ===> Time: ", time.time()-t0, " step time: ", time.time()-step_start)


x_file = DATA_PATH + '/Train/graph_10percent_train'
y_file = DATA_PATH+"/Train/y_10percent_train.npy"



t0 = time.time()
"""X_train = np.load('10percentOfX_train.npy')
y_train = np.load('10percentOfy_train.npy')"""
import pickle
file = open("10percentOfGraph_train", 'rb')
embeddings_path = DATA_PATH+'/embeddings'
graphes = pickle.load(file)
embedding_file = '10percentOfEmbedding_train16_25_64'
embdns = []
print("Chargement des données:",time.time()-t0)
t1 = time.time()
for graph in graphes:
    embdns.append(embed_graph(graph, 16))
    print("Time:", time.time()-t1)
print("Temps de l'embedding:",time.time()-t1)
np.save(embedding_file, embdns)
print("Temps globale:", time.time()-t0)



