import argparse
import numpy as np
import yaml
import os
import cv2

def cal_supp_locate_edge(other_states):
    '''
    This func is used for obtaining the pixels of the bounding box of each supplement agents
    :param other_states:
    :return: the pixels of the bounding boxes -- x, y, vx, vy
    '''
    locate_states = []
    if len(other_states) < 1:
        return locate_states

    for i in range(len(other_states)):
        x_min, y_min, x_max, y_max = other_states[i, 7:11].astype(int)

        vertex = np.zeros((4, 5))
        vertex[0, :2] = np.array([x_min, y_min])
        vertex[1, :2] = np.array([x_max, y_min])
        vertex[2, :2] = np.array([x_min, y_max])
        vertex[3, :2] = np.array([x_max, y_max])

        x_range = np.arange(x_min+1, x_max)
        y_range = np.arange(y_min+1, y_max)

        edge1 = np.zeros((len(x_range), 5))
        edge2 = np.zeros((len(x_range), 5))
        edge3 = np.zeros((len(y_range), 5))
        edge4 = np.zeros((len(y_range), 5))
        edge1[:, 0] = x_range
        edge1[:, 1] = y_min
        vertex = np.vstack((vertex, edge1))
        edge2[:, 0] = x_range
        edge2[:, 1] = y_max
        vertex = np.vstack((vertex, edge2))
        edge3[:, 0] = x_min
        edge3[:, 1] = y_range
        vertex = np.vstack((vertex, edge3))
        edge4[:, 0] = x_max
        edge4[:, 1] = y_range
        vertex = np.vstack((vertex, edge4))
        vertex[:, 2:4] = other_states[i, 5:7]
        vertex[:, -1] = i
        locate_states.append(vertex)

    return np.concatenate(locate_states, axis=0)

parser = argparse.ArgumentParser(description='SocialCVAE')
parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--config_filename', '-cfn', type=str, default='sdd.yaml')
parser.add_argument('--save_file', '-sf', type=str, default='sdd_model.pt')
parser.add_argument('--verbose', '-v', action='store_true')

args = parser.parse_args()

with open(f"config/{args.config_filename}", 'r') as file:
    try:
        params = yaml.load(file, Loader=yaml.FullLoader)
    except Exception:
        params = yaml.load(file)
file.close()

DATA_NAME=params['dataset']

TRAIN_DATA_PATH = f'data/{DATA_NAME}/train.pickle'
TEST_DATA_PATH = f'data/{DATA_NAME}/test.pickle'

## load raw data & semantic segmentations
train_data = np.load(TRAIN_DATA_PATH, allow_pickle=True)
test_data = np.load(TEST_DATA_PATH, allow_pickle=True)

delta_t = 0.4
ped_r = params['ped_r']
ped_r_plus = ped_r + ped_r
ped_r_plus2 = ped_r_plus * ped_r_plus

METAID_PATH = f'data/{DATA_NAME}/metaid.pickle'
metaid_int2str = np.load(METAID_PATH, allow_pickle=True)


semantic_path = os.path.join('data', DATA_NAME, 'masks')
semantic_scenes = os.listdir(semantic_path)
semantic_images = {}
for scene in semantic_scenes:
    img_label, _ = scene.split('_mask', 1)
    img = cv2.imread(os.path.join(semantic_path, scene), 0)
    semantic_images[img_label] = img

train_images = []
test_images  = []
print('load train images... ...')
for ped_traj, _, _ in train_data:
    scene_meta_str = metaid_int2str[int(ped_traj[0, 0, -1])]
    scene, video = scene_meta_str.split('video', 1)
    train_images.append(semantic_images[scene+video])
print('DONE: load train images')

print('load test images... ...')
for ped_traj, _, _ in test_data:
    scene_meta_str = metaid_int2str[int(ped_traj[0, 0, -1])]
    scene, video = scene_meta_str.split('video', 1)
    test_images.append(semantic_images[scene+video])
print('DONE: load test images')

train_other_edge = []
test_other_edge = []
for _, _, other_data in train_data:
    batch_other_locate = []
    for t in range(params['past_length'] + params['future_length']):
        other_num = int(other_data[t, -1, -1])
        t_other_locate = cal_supp_locate_edge(other_data[t, :other_num, :])
        batch_other_locate.append(t_other_locate)
    batch_other_locate = np.array(batch_other_locate, dtype=object)
    train_other_edge.append(batch_other_locate)

for _, _, other_data in test_data:
    batch_other_locate = []
    for t in range(params['past_length'] + params['future_length']):
        other_num = int(other_data[t, -1, -1])
        t_other_locate = cal_supp_locate_edge(other_data[t, :other_num, :])
        batch_other_locate.append(t_other_locate)
    batch_other_locate = np.array(batch_other_locate, dtype=object)
    test_other_edge.append(batch_other_locate)