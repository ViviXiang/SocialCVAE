import os
import math
import numpy as np
import copy
import pickle
import codecs
import re
import cv2
# import matplotlib.pyplot as plt

def world2image(traj_w, scene, H_inv): # transfer the meter position to image position
    # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
    
    traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T
    # to camera frame
    traj_cam = np.matmul(H_inv, traj_homog)
    # to pixel coords
    traj_uvz = np.transpose(traj_cam/traj_cam[2])
    
    if scene in ['eth', 'hotel']: # eth and hotel have different coordinate system than ucy data
        traj_uvz[:, [0, 1]] = traj_uvz[:, [1, 0]]
    
    return traj_uvz[:, :2]

def image2world(traj_img, scene, H): # transfer world coordiantes (pixel) to image coordinates (meter)
    # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
    traj_homog = np.hstack((traj_img, np.ones((traj_img.shape[0], 1))))
    if scene in ['eth', 'hotel']: # eth and hotel have different coordinate system than ucy data
        traj_homog[:, [0, 1]] = traj_homog[:, [1, 0]]
    # to camera frame
    traj_cam = np.matmul(H, np.transpose(traj_homog))
    # to meter coords
    traj_uvz = np.transpose(traj_cam/traj_cam[2])
    return traj_uvz[:, :2]

def calculate_v(x_seq):
    length = x_seq.shape[1]
    peds = x_seq.shape[0]
    x_seq_velocity = np.zeros_like(x_seq)
    episa = 1e-6
    for i in range(1, length):
        for j in range(peds):
            position = x_seq[j][i]
            before_position = x_seq[j][i-1]
            position_norm = np.linalg.norm(position)
            before_position_norm = np.linalg.norm(before_position)
            if position_norm < episa:
                velocity = np.array([0,0])
            else:
                if before_position_norm < episa:
                    velocity = np.array([0, 0])
                else:
                    velocity = (position - before_position)/0.4
            x_seq_velocity[j][i] = velocity
    return x_seq_velocity

def calculate_a(vel):
    length = vel.shape[1]
    peds = vel.shape[0]
    x_seq_acce = np.zeros_like(vel)
    episa = 1e-6
    for i in range(2, length):
        for j in range(peds):
            position = vel[j][i]
            before_position = vel[j][i - 1]
            position_norm = np.linalg.norm(position)
            before_position_norm = np.linalg.norm(before_position)
            if position_norm < episa:
                acce = np.array([0, 0])
            else:
                if before_position_norm < episa:
                    acce = np.array([0, 0])
                else:
                    acce = (position - before_position) / 0.4
            x_seq_acce[j][i] = acce
    return x_seq_acce


def outlier_test(seq, seq_remains):
    seq = np.swapaxes(seq, 1, 2)  # peds*20*2
    seq_remains = np.swapaxes(seq_remains, 1, 2)  # peds*20*3
    seq_remains_label = np.max(seq_remains[:,:,-1], axis=1) #peds
    seq_remains = seq_remains[seq_remains_label  == 1,:,:] #peds*20*3, all the ped_remains who are humans
    seq_remains = seq_remains[:,:,:-1] #peds*20*2
    seq_all = np.concatenate([seq, seq_remains]) #peds*20*2
    vel_all = calculate_v(seq_all) #peds*20*2
    acc_all = calculate_a(vel_all) #peds*20*2
    vel_all_norm = np.linalg.norm(vel_all, axis=-1) #peds*20
    acc_all_norm = np.linalg.norm(acc_all, axis=-1) #peds*20
    vel_all_norm_max = np.amax(vel_all_norm) #1
    acc_all_norm_max = np.amax(acc_all_norm) #1
    if vel_all_norm_max > 300:
        bool_out = True
    else:
        if vel_all_norm_max > 100 and vel_all_norm_max <= 300:
            if acc_all_norm_max > 150:
                bool_out = True
            else:
                bool_out = False
        else:
            bool_out = False

    return bool_out

scenes = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
paths = []
for scene in scenes:
    path = f'data/ETHUCY_ini/{scene}/'
    paths.append(path)
    
data_scenes = ['eth', 'hotel', 'students001', 'students003', 'uni_examples', 'zara1', 'zara2', 'zara3']
H = {}
H_inv = {}
for data_scene in data_scenes:
    h = np.loadtxt(f'data/ETHUCY_ini/{data_scene}_H.txt')
    H[data_scene] = h
    H_inv[data_scene] = np.linalg.inv(h)
    
data_name = {'biwi_hotel':'hotel', 'crowds_zara01':'zara1', 'crowds_zara02':'zara2',
                     'crowds_zara03':'zara3', 'students001':'students001', 'students003':'students003',
                     'uni_examples':'uni_examples', 'biwi_eth':'eth'}


obs_len = 8
pred_len = 12
seq_len = obs_len + pred_len
skip = 20
min_ped = 1
max_peds = 120
num_success = 0
delta_t=0.4


num_seqs = 0  # the index of a traj sequence: mapping to a time interval
episa = 1e-6

for label_set, (scene_path, scene) in enumerate(zip(paths, scenes)):
    save_path = f'data/ETHUCY/{scene}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_files = ['train', 'test', 'val']
    # all_files = ['test']
    for tfile in all_files: 
        # the obtained 'train', 'test' or 'val' data
        t_seq_list = []
        t_supplement = []
        t_scene_name = []
        num_agents = 0
        num_batches = 0

        file_path = f'{scene_path}{tfile}'
  
        all_videos = os.listdir(file_path)
        for vid, video in enumerate(all_videos):
            
            processed_data_path = f'{file_path}/{video}'
            filecp = codecs.open(processed_data_path, encoding='cp1252')

            data = np.loadtxt(filecp)

            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - 20 + 1)))
            
            if tfile == 'test':
                data_name_raw,_ = video.split('.txt')
            else:
                data_name_raw,_ = video.split(f'_{tfile}')
            curr_data_scene_name = data_name[data_name_raw]
            
            for idx in range(0, num_sequences + 1, 1):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # the ids of each pedestrian: 1*peds
                curr_seq = np.zeros((len(peds_in_curr_seq), 2,
                                     seq_len))  # (data sequence in this time interval) the trajs for each ped: peds*2*20

                curr_fseq_ped_data = [] # the active pedestrians that exist during the whole sequence
                curr_fseq_supplement_data = np.zeros((seq_len, max_peds, 10))
                curr_fseq_scene_name = []
                num_peds_considered = 0
                curr_fseq_supp = []
                for i, ped_id in enumerate(peds_in_curr_seq):
                    
                    ### record the traj info of pedestrian ped_id
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]  # the data sequence of the ped ped_id
                    
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    
                    curr_ped_vel = np.zeros((curr_ped_seq.shape[0],2))
                    curr_ped_vel[1:, :] = (curr_ped_seq[1:, 2:4] - curr_ped_seq[0:-1, 2:4]) / delta_t
                    if curr_ped_seq.shape[0] > 1:
                        curr_ped_vel[0, :] = curr_ped_vel[1, :]

                    curr_ped_seq = np.hstack((curr_ped_seq, curr_ped_vel))
                   
                    current_ped_pos_image = world2image(curr_ped_seq[:, 2:4], curr_data_scene_name, H_inv[curr_data_scene_name])
                    
                    # test the image coord to meter coord
                    test_meter_pos = image2world(current_ped_pos_image, curr_data_scene_name, H[curr_data_scene_name])
                    test_offset = np.sum(np.linalg.norm((test_meter_pos - curr_ped_seq[:, 2:4]), axis=1))
                    assert test_offset < 1e-6

                    current_ped_vel_image = np.zeros((curr_ped_seq.shape[0],2))
                    current_ped_vel_image[1:, :] = (current_ped_pos_image[1:, :] - current_ped_pos_image[0:-1, :]) / delta_t
                    if curr_ped_seq.shape[0] > 1:
                        current_ped_vel_image[0, :] = current_ped_vel_image[1, :]
                    
                    current_ped_seq_image = np.hstack((current_ped_pos_image, current_ped_vel_image))
                    
                    curr_ped_seq = np.hstack((curr_ped_seq, current_ped_seq_image))
                    
                    
                    if pad_end - pad_front != 20:
                        curr_fseq_supp.append(curr_ped_seq)
                        continue
                    
                    curr_fseq_ped_data.append(curr_ped_seq)
                        
                if len(curr_fseq_ped_data) <= min_ped:
                    continue
                
                curr_min_f = int(curr_fseq_ped_data[0][0, 0]/10)
                for curr_ped_seq_supp in curr_fseq_supp:
                    for data_term in curr_ped_seq_supp:
                        time_t = int(data_term[0]/10) - curr_min_f
                        curr_fseq_supplement_data[time_t, int(curr_fseq_supplement_data[time_t, -1, -1]), :] = data_term
                        curr_fseq_supplement_data[time_t, -1, -1] += 1
                
                curr_fseq_scene_name.append(curr_data_scene_name)
                num_agents += num_peds_considered
                curr_fseq_ped_data = np.array(curr_fseq_ped_data)
                t_seq_list.append(curr_fseq_ped_data)
                t_supplement.append(curr_fseq_supplement_data)
                t_scene_name.append(curr_fseq_scene_name)
                
        ## save the processed data
        save_name = f'{save_path}/{tfile}.pkl'

        with open(save_name, 'wb') as f:
            pickle.dump([t_seq_list, t_supplement, t_scene_name], f)

        print(f'success {scene}-{tfile}:, {len(t_seq_list)} batches')
