import copy
import numpy as np
import pandas as pd
import os
import math
import pickle

class_dic = {'Pedestrian': 0, 'Car': 1, 'Bus': 2, 'Cart': 3, 'Biker': 4, 'Skater': 5}
SDD_cols = ['trackId', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']

step = 12
delta_t = 0.4
min_ped = 1

def cal_v(data, delta_t=0.4):
    v = copy.deepcopy(data[:, 5:7])
    agents = np.unique(data[:, 0])
    for agent in agents:
        if agent == 72:
            aaaa = -1
        a_index = np.argwhere(data[:, 0] == agent)
        if len(a_index) > 1:
            xy = data[a_index, 3:5]
            v[a_index[1:], :] = (xy[1:, :] - xy[:-1, :]) / delta_t
            v[a_index[0], :] = v[a_index[1], :]
        # deltav = v - data[:, 5:7]
    return v

def rad2vec(angle, length):
    vx = length * np.cos(angle)
    vy = length * np.sin(angle)
    return vx, vy

acc_vel_step = []

def process_sdd_per_step(path, max_peds=200, obs_len=8, pred_len=12):
    seq_len = obs_len + pred_len
    skip = seq_len
    scenes = os.listdir(path)
    scenes.sort()
    success_num = 0

    metaid_str2int = {}
    metaid_int2str = []
    mid = 0
    for scene in scenes:
        scene_path = path + scene
        videos = os.listdir(scene_path)
        videos.sort()
        for video in videos:
            meta_str = scene + '_' + video
            metaid_str2int[meta_str] = mid
            metaid_int2str.append(meta_str)
            mid += 1

        for video in videos:
            video_data_per_fseq = []
            video_path = scene_path + '/' + video
            file_path = video_path + '/' + 'annotations.txt'
            scene_df = pd.read_csv(file_path, header=0, names=SDD_cols, delimiter=' ')
            scene_df['x'] = (scene_df['xmax'] + scene_df['xmin']) / 2
            scene_df['y'] = (scene_df['ymax'] + scene_df['ymin']) / 2

            scene_df = scene_df[scene_df['lost'] == 0]

            scene_df['frame_rest'] = scene_df['frame'] % step
            scene_df = scene_df[scene_df['frame_rest'] == 0]



            scene_df['vx'] = np.zeros_like(scene_df['x'])
            scene_df['vy'] = np.zeros_like(scene_df['x'])

            scene_df['x_min'] = scene_df['xmin']
            scene_df['y_min'] = scene_df['ymin']
            scene_df['x_max'] = scene_df['xmax']
            scene_df['y_max'] = scene_df['ymax']

            scene_df = scene_df.drop(columns=['xmin', 'xmax', 'ymin', 'ymax', 'occluded', 'generated', 'lost', 'frame_rest'])

            metaID_str = scene + '_' + video
            scene_df['metaID'] = metaid_str2int[metaID_str]

            scene_np = scene_df.to_numpy()

            scene_np[:, 1] = scene_np[:, 1] / step

            for i in range(scene_np.shape[0]):
                scene_np[i, 2] = class_dic[scene_np[i, 2]]


            agent_types = np.unique(scene_np[:, 2])
            for atype in agent_types:
                scene_np[scene_np[:, 2]==atype, 5:7] = cal_v(scene_np[scene_np[:, 2]==atype, :], delta_t)


            frames = np.unique(scene_np[:, 1]).tolist()
            frame_data = [scene_np[frame == scene_np[:, 1], :] for frame in frames]
            num_sequences = int(math.ceil((len(frames) - seq_len + 1) / skip))

            for fseq_id in range(0, num_sequences * skip + 1, skip):
                if len(frame_data[fseq_id:fseq_id + seq_len]) < 1:
                    continue

                curr_seq_all = np.concatenate(frame_data[fseq_id:fseq_id + seq_len], axis=0)

                curr_ped_all = curr_seq_all[curr_seq_all[:, 2] == 0, :]

                ped_exists = np.unique(curr_ped_all[:, 0])
                if len(ped_exists) < 1:
                    continue

                curr_veh_all = curr_seq_all[curr_seq_all[:, 2] != 0, :]

                curr_fseq_ped_supp = []
                curr_fseq_ped_data = [] # the active pedestrians that exist during the whole sequence

                for ped_id in ped_exists:
                    ### record the traj info of pedestrian ped_id
                    curr_ped_seq = curr_ped_all[curr_ped_all[:, 0] == ped_id, :]
                    if len(curr_ped_seq) != seq_len:
                        curr_fseq_ped_supp.append(curr_ped_seq)
                    else:
                        curr_fseq_ped_data.append(curr_ped_seq)

                if len(curr_fseq_ped_data) < min_ped: 
                    continue

                curr_fseq_ped_data = np.array(curr_fseq_ped_data)
                curr_min_f = int(np.min(curr_ped_all[:, 1]))
                curr_fseq_supplement_data = np.zeros((seq_len, max_peds, scene_np.shape[1]))
                for curr_ped_seq in curr_fseq_ped_supp:
                    for data_term in curr_ped_seq:
                        time_t = int(data_term[1]) - curr_min_f
                        curr_fseq_supplement_data[time_t, int(curr_fseq_supplement_data[time_t, -1, -1]), :] = data_term
                        curr_fseq_supplement_data[time_t, -1, -1] += 1

                curr_fseq_veh_data = np.zeros((seq_len, max_peds, scene_np.shape[1]))
                f_in_vehsupp = np.unique(curr_veh_all[:, 1])
                for f in f_in_vehsupp:
                    time_t = int(f) - curr_min_f
                    curr_veh_supplement = curr_veh_all[curr_veh_all[:, 1] == f, :]
                    curr_veh_supp_num = len(curr_veh_supplement)
                    curr_fseq_veh_data[time_t, :curr_veh_supp_num, :] = curr_veh_supplement
                    curr_fseq_veh_data[time_t, -1, -1] = curr_veh_supp_num

                curr_fseq = [curr_fseq_ped_data, curr_fseq_supplement_data, curr_fseq_veh_data]
                video_data_per_fseq.append(curr_fseq)

            success_num += 1
            print(
                'success ',
                scene,
                ' video',
                video,
                ', metaid ',
                metaid_str2int[metaID_str],
                ', batch size ',
                len(video_data_per_fseq)
            )
            out_path_per_fseq = './data/SDD_ini/processed/data/' + scene + '_' + video + '2.pickle'
            with open(out_path_per_fseq, 'wb') as f:
                pickle.dump(video_data_per_fseq, f)

    out_path_metaid = './data/SDD/metaid.pickle'
    with open(out_path_metaid, 'wb') as f:
        pickle.dump(metaid_int2str, f)

def obtain_train_and_test_dataset():
    train_mask_path = './data/SDD/train_masks'
    test_mask_path = './data/SDD/test_masks'
    mask_paths = [train_mask_path, test_mask_path]

    for i, mask_path in enumerate(mask_paths):
        batch_num = 0
        save_data = []
        save_data_name = 'train.pickle' if i==0 else 'test.pickle'
        video_masks = os.listdir(mask_path)
        video_masks.sort()
        for video_mask in video_masks:
            data_name, video_index, _ = video_mask.split('_', 2)
            data_path = './data/SDD_ini/processed/data/' + data_name + '_video' + video_index + '.pickle'
            data = np.load(data_path, allow_pickle=True)
            if len(data) < 1:
                continue
            batch_num += len(data)
            save_data.extend(data)
        out_put_path = f'./data/SDD/{save_data_name}'
        with open(out_put_path, 'wb') as f:
            pickle.dump(save_data, f)


path = './data/SDD_ini/annotations/'

process_sdd_per_step(path, max_peds=120)
obtain_train_and_test_dataset()
