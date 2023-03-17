
import numpy as np
from os.path import join,exists
from os import mkdir
from scipy.spatial import distance

import time

rho = 1

actions = ['sweeping', 'gargling', 'opening cupboard', 'washing hands',
           'eating', 'writing', 'wiping',
           'drinking','opening microwave oven', 'Throwing trash']

def get_data(file,type='no_order'):
    try:
        f = open(file,'r').read().split()
        datait = [float(x) for x in f]
        if type=='no_order':
            data = np.asarray(datait)
            data = data.reshape((25,3))
        else:
            spine_base = datait[0:3]
            spine_mid = datait[3:6]
            neck = datait[6:9]
            head = datait[9:12]
            shoulder_left = datait[12:15]
            elbow_left = datait[15:18]
            wrist_left = datait[18:21]
            hand_left = datait[21:24]
            shoulder_right = datait[24:27]
            elbow_right = datait[27:30]
            wrist_right = datait[30:33]
            hand_right = datait[33:36]
            hip_left = datait[36:39]
            knee_left = datait[39:42]
            ankle_left = datait[42:45]
            foot_left = datait[45:48]
            hip_right = datait[48:51]
            knee_right = datait[51:54]
            ankle_right = datait[54:57]
            foot_right = datait[57:60]
            spine_shoulder = datait[60:63]
            handtip_left = datait[63:66]
            thumb_left = datait[66:69]
            handtip_right = datait[69:72]
            thumb_right = datait[72:75]

            if type=='head_to_feet':
                data=np.stack((head, neck, spine_shoulder,
                               shoulder_left, shoulder_right, elbow_left,
                               elbow_right, wrist_left, wrist_right,
                               thumb_left, thumb_right, hand_left,
                               hand_right, handtip_left, handtip_right,
                               spine_mid, spine_base, hip_left,
                               hip_right, knee_left, knee_right,
                               ankle_left, ankle_right, foot_left, foot_right))
            else : # foot_to_foot
                data=np.stack((foot_left, ankle_left, knee_left,
                               hip_left, spine_base, handtip_left,
                               thumb_left, hand_left, wrist_left,
                               elbow_left, shoulder_left,
                               spine_shoulder,head,neck,
                               shoulder_right,elbow_right,
                               wrist_right,   hand_right,thumb_right,
                               handtip_right, spine_mid, hip_right,
                               knee_right, ankle_right,foot_right))
        return data
    except:
        print('Ex',file)
        return None

def normalize(array):
    min_ = np.min(array,0)
    max_ = np.max(array,0)
    return (array-min_)/(max_-min_)

def get_sequence_energy(sequence):
    energy = np.zeros((len(sequence),25))
    # energy = np.zeros(25)
    for i in range(len(sequence)):
        for k in range(25):
            if i == 0:
                energy[i][k] = np.linalg.norm(sequence[i][k] - sequence[i + 1][k])
            elif i == len(sequence)-1:
                energy[i][k] = np.linalg.norm(sequence[i][k] - sequence[i - 1][k])
            else:
                energy[i][k] = (np.linalg.norm(sequence[i][k] - sequence[i + 1][k])+np.linalg.norm(sequence[i][k] - sequence[i - 1][k]))/2
            # if i == 0:
            #     continue
            # energy[k]+= np.linalg.norm(sequence[i][k]- sequence[i][k - 1])
    E = normalize(energy)
    w = rho*E + (1-rho)
    return w

def get_labels(file):
    labels = open(file,'r').read().splitlines()
    prev_action=None
    start =[]
    end = []
    actions=[]
    for line in labels:
        if line.replace(' ','').isalpha():
            prev_action = line.strip()
        else:
            tab = line.split(' ')
            start.append(int(tab[0]))
            end.append(int(tab[1]))
            actions.append(prev_action)
    return (start,end,actions)

def get_image_label(start,end,labels):
    index = (start+end)//2
    for s,e,a in set(zip(labels[0],labels[1],labels[2])):
        if s <= index and index <= e:
            return a
    return None

"""def transform_image_ludl(image,path,name,weights):
    RGB = image
    from copy import deepcopy
    height = image.shape[1]
    width = image.shape[0]
    X = np.arange(height)
    Y = np.arange(width)
    RGB = np.squeeze(RGB)
    En = deepcopy(RGB)
    if len(weights.shape)==1:
        weights = np.expand_dims(weights,0)
    white = np.ones((width,height))*255
    for i in range(3):
        RGB[:,:,i] = np.floor(255 * (RGB[:,:,i] - np.min(RGB[:,:,i])) / (np.max(RGB[:,:,i]) - np.min(RGB[:,:,i])))
        En[:, :, i] = RGB[:, :, i]*weights+(1-weights)*white

    img = np.zeros((2*height, width, 3), dtype=np.uint8)
    for i in X:
        for j in Y:
            img[i,j]=RGB[j,i]
    for i in X:
        for j in Y:
            img[i+25, j] = En[j, i]
    cv2.imwrite(join(path,name+'_.png'),img)
    return img"""

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

NODE_PER_FRAME = 25

DATA_PATH = 'C:\\Users\\Mohamed\\PycharmProjects\\DATA'

import networkx as nx
def transform_graph(graph_data, fc = False):
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

def to_nassim(data_path,labels,window_length=40,type_='foot_to_foot'):
    start_frame = min(labels[0]) - window_length//2
    end_frame = max(labels[1]) + window_length //2
    data = []
    for i in range(start_frame,end_frame+1):
        data.append(get_data(data_path+'/'+str(i)+'.txt',type_))
    images = [data[i:i + window_length] for i in range(len(data) - window_length + 1)]
    lab = [get_image_label(i,i+window_length,labels) for i in range(start_frame,end_frame - window_length+2)]
    i=0
    while i <len(lab):
        if lab[i] is None:
            del lab[i]
            del images[i]
        else:
            i+=1
    i = 0
    while i < len(images):
        jumped = False
        for x in images[i]:
            if x is None:
                print(x is None)
            if x is None or not x.shape==(25,3):
                    del lab[i]
                    del images[i]
                    jumped = True
                    break
        if not jumped:
            i+=1
    print(len(images), len(lab), len(data), window_length, len(data) - window_length + 1)
    return np.asarray(images), np.asarray(lab)


def transform_nassim(data_path,label_path):
    images, labels = to_nassim(data_path, get_labels(label_path), window_length=40)
    data = []
    lab = []
    for i in range(len(images)):
        data.append(transform_graph(images[i]))
        lab.append(actions.index(labels[i]))
    data = np.asarray(data)
    labels = np.asarray(lab)
    return data, labels


data_path='data'
train_path = 'Train'
test_path = 'Test'
if not exists(train_path):
    mkdir(train_path)
if not exists(test_path):
    mkdir(test_path)
train_sub = [1, 2, 3, 4, 7, 8, 9, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 32, 33, 34, 35, 37, 38, 39, 49, 50, 51, 54, 57, 58]
test_sub  = [0, 10, 13, 17, 21, 26, 27, 28, 29, 36, 40, 41, 42, 43, 44, 45, 52, 53, 55, 56]


def load_dataset(subset):
    X = []
    y = []
    for i in subset:
        path = join(data_path, str(i))
        label_path = join(path,'label','label.txt')
        labels = get_labels(label_path)
        image_path = join(path,'skeleton')
        print('Processing sequence num ===========>',i)
        data, label  = to_nassim(image_path, labels)
        for x in data:
            X.append(x)
        for l in label:
            y.append(l)
    return X, y

X_train_file = train_path + "/x_train"
y_train_file = train_path + "/y_train"
X_test_file = test_path + "x_test"
y_test_file = test_path + "y_test"

def import_data():

    if not exists(X_train_file+'.npy'):
        X_train, y_train = load_dataset(train_sub)
        np.save(X_train_file, X_train)
        np.save(y_train_file, y_train)
    if not exists(X_test_file+".npy"):
        X_test, y_test = load_dataset(test_sub)
        np.save(X_test_file, X_test)
        np.save(y_test_file, y_test)

def recharger_train_data():
    X, y = np.load(X_train_file + '.npy'), np.load(y_train_file + '.npy')
    return X, y

def recharger_test_data():
    X, y = np.load(X_test_file + '.npy'), np.load(y_test_file + '.npy')
    return X, y

from node2vec import Node2Vec

def embed_graph(graphe, dim=10, walk_l=15, walks=25, workers=8, window=10, min_count=1, batch_words=4):
    node2vec = Node2Vec(graphe, dimensions=dim, walk_length=walk_l, num_walks=walks, workers=workers)
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
    return model.wv

def save_embedding(embedding, dimension = 10, filename = 0):
    train_path  = join('Train','train_'+str(dimension))
    if not exists(train_path):
        mkdir(train_path)
    path = join(train_path,str(filename)+".txt")
    embedding.save_word2vec_format(path)

def save_all_embedding(dimension, train = True, fc = True):
    start = time.time()
    if train:
        X, y = recharger_train_data()
        train_path = join('Train', 'train_' + str(dimension))
    else:
        X, y = recharger_train_data()
        train_path = join('Test', 'test_' + str(dimension))
    embeddings = []
    for i in range(len(y)):
        step_time = time.time()
        print(" <========== ", i, " ===========> ")
        node2vec = Node2Vec(transform_graph(X[i], fc),
                            dimensions=dimension,
                            walk_length=15,
                            num_walks=5,
                            workers=1)
        model = node2vec.fit(window=10,
                             min_count=1,
                             batch_words=4)
        print('Step time: ', time.time()-step_time, "Total time: ", time.time()-start)
        embeddings.append(model.wv.vectors)
    np.save(train_path, embeddings)
    return time.time() - start

import sys

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


import_data()
X, y = recharger_train_data()
X_test, y_test = recharger_test_data()
# save_all_embedding(4)
















'''
from keras.utils.np_utils import to_categorical
test_label = to_categorical(test_label)
train_label = to_categorical(train_label)
print(test_label.shape)
np.save('train_x_{}_base_seq.npy'.format(rho),train)
np.save('test_x_{}_base_seq.npy'.format(rho),test)
np.save('train_y_{}_base_seq.npy'.format(rho),train_label)
np.save('test_y_{}_base_seq.npy'.format(rho),test_label)

Y = np.argmax(train_label,axis=1)
print(Y.shape)
unique, counts = np.unique(Y, return_counts=True)
print(dict(zip(unique, counts)))

Y = np.argmax(test_label,axis=1)
print(Y.shape)
unique, counts = np.unique(Y, return_counts=True)
print(dict(zip(unique, counts)))
print(train.shape,train_label.shape,test.shape,test_label.shape)
#29126,)
# (23912,)'''