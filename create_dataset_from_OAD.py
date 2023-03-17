
import numpy as np
from os.path import join,exists
from os import mkdir

data_path='C:\\Users\\Mohamed\\PycharmProjects\\DATA'
train_path = 'Train'
test_path = 'Test'

if not exists(train_path):
    mkdir(train_path)
if not exists(test_path):
    mkdir(test_path)
train_sub = [1, 2, 3, 4, 7, 8, 9, 14, 15, 16, 18, 19, 20, 22, 23, 24, 25, 32, 33, 34, 35, 37, 38, 39, 49, 50, 51, 54, 57, 58]
test_sub  = [0, 10, 13, 17, 21, 26, 27, 28, 29, 36, 40, 41, 42, 43, 44, 45, 52, 53, 55, 56]


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


def get_labels(file):
    labels = open(file, 'r').read().splitlines()
    prev_action = None
    start = []
    end = []
    actions = []
    for line in labels:
        if line.replace(' ', '').isalpha():
            prev_action = line.strip()
        else:
            tab = line.split(' ')
            start.append(int(tab[0]))
            end.append(int(tab[1]))
            actions.append(prev_action)
    return (start, end, actions)


def get_image_label(start, end, labels):
    index = (start + end) // 2
    for s, e, a in set(zip(labels[0], labels[1], labels[2])):
        if s <= index and index <= e:
            return a
    return None


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
    print("Data is completely stored as NumPy Array.")
