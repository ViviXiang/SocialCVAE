import numpy as np
import copy
import torch
import torch.nn.functional as F
import numpy as np
import copy
import torch
from utils.preprocessing_sdd import delta_t, ped_r, ped_r_plus


def calculate_loss_cvae(mean, log_var, criterion, future, predictions):
    # reconstruction loss
    ADL_traj = criterion(future, predictions) # better with l2 loss
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return KLD, ADL_traj


EPSILON = 10e-6

def select_nei_in_fov_recArea(xij, fov_angle=2*np.pi, fov_r=200):
    '''
    We regard the fov of an agent as a rectangle:
    the default is a square with the current agent as center
    '''
    if fov_angle == 2*np.pi:
        abs_xij_max = np.max(np.abs(xij), axis=1)
        return np.where(abs_xij_max < fov_r)[0]
    elif fov_angle == np.pi:
        return np.arange(0, len(xij))


def cal_ped_vision_area(curr_x, curr_y, fov_r):
    x_min = curr_x - fov_r + 1
    x_max = curr_x + fov_r - 1
    y_min = curr_y - fov_r + 1
    y_max = curr_y + fov_r - 1

    return np.array([x_min, x_max, y_min, y_max])

def cal_nei_ped_area(nx, ny, ped_r, map_h, map_w):
    x_min = max(nx - ped_r, 0)
    x_max = min(nx + ped_r + 1, map_w-1)
    y_min = max(ny - ped_r, 0)
    y_max = min(ny + ped_r + 1, map_h-1)

    return x_min, x_max, y_min, y_max

def cal_em_ped(current_state, pred_vel, current_ped_nei, pred_ped_nei, fov_r=50, map_h=100, map_w=100):
    em_ped_rep = np.zeros((map_h, map_w))

    xij = current_state[3:5] - current_ped_nei[:, 3:5]
    abs_xij_max = np.max(np.abs(xij), axis=1)
    ped_in_fov = np.where(abs_xij_max < fov_r)[0]
    if len(ped_in_fov) < 1:
        return em_ped_rep

    xij = xij[ped_in_fov, :]
   
    vij = pred_vel - current_ped_nei[ped_in_fov, 5:7]
    vij_sqr = np.sum(vij**2, axis=1) + EPSILON

    ttc = -np.sum(np.multiply(xij, vij), axis=1) / vij_sqr
    ttc[ttc<0] = delta_t
    ttc[ttc>delta_t] = delta_t

    delta_xij = copy.deepcopy(vij)
    delta_xij[:, 0] *= ttc
    delta_xij[:, 1] *= ttc
    xij_pred = delta_xij + xij
    xij_pred_sqrt = np.linalg.norm(xij_pred, axis=1) - ped_r_plus


    energy_ped_rep = np.exp(-xij_pred_sqrt/ped_r)
    
    map_center_index = np.array([fov_r, fov_r]).astype(int)
    curr_ped_global_index = np.round(current_state[3:5]).astype(int)
    nei_locate_norm = np.round(pred_ped_nei[ped_in_fov]).astype(int) - curr_ped_global_index + map_center_index

    for i, nei_locate in enumerate(nei_locate_norm):
        nx, ny = nei_locate[0], nei_locate[1]

        nx_min, nx_max, ny_min, ny_max = cal_nei_ped_area(nx, ny, ped_r, map_h, map_w)
        em_ped_rep[nx_min:nx_max+1, ny_min:ny_max+1] += energy_ped_rep[i]

    return em_ped_rep

def cal_em_envobs(current_state, pred_vel, semantic_map, ped_vision_area, fov_r=50, map_h=100, map_w=100):
    scene_h, scene_w = semantic_map.shape[0], semantic_map.shape[1]
    em_envobs = np.zeros((map_h, map_w))

    vision_xmin, vision_xmax, vision_ymin, vision_ymax = ped_vision_area

    if vision_ymax < 0 or vision_xmax < 0:
        return em_envobs
    if vision_xmin > scene_w-1 or vision_ymin > scene_w-1:
        return em_envobs

    wmin, wmax, hmin, hmax = max(vision_xmin, 0), min(vision_xmax, scene_w), max(vision_ymin, 0), min(vision_ymax, scene_h)

    local_semantic = semantic_map[hmin:hmax, wmin:wmax]

    obs_locate_norm = np.argwhere(local_semantic == 3)


    if len(obs_locate_norm) < 1:
        return em_envobs

    # cal the accurate relative index in the ped_for (401*401 map with (200, 200) as center)
    if vision_xmin < 0: # width index
        obs_locate_norm[:, 1] += abs(vision_xmin)
    if vision_ymin < 0: # height index
        obs_locate_norm[:, 0] += abs(vision_ymin)
    # flip the image index to 2d-cord of the ped
    obs_locate_norm = np.flip(obs_locate_norm, axis=1)

    ped_locate_norm = np.array([fov_r, fov_r])
    xij = ped_locate_norm - obs_locate_norm
    vij = pred_vel
    vij_sqr = np.sum(vij**2) + EPSILON

    ttc = -np.sum(np.multiply(xij, vij), axis=1) / vij_sqr
    ttc[ttc<0] = delta_t
    ttc[ttc>delta_t] = delta_t
    delta_xij = np.zeros((len(ttc), 2))

    for i in range(len(ttc)):
        delta_xij[i, :] = ttc[i] * vij
    
    xij_pred = delta_xij + xij
    xij_pred_sqrt = np.linalg.norm(xij_pred, axis=1) - ped_r
    energy_obs_rep = np.exp(-xij_pred_sqrt/ped_r)

    for i, obs_locate in enumerate(obs_locate_norm):
        em_envobs[obs_locate[0], obs_locate[1]] += energy_obs_rep[i]

    return em_envobs


'''
In the SDD dataset, other types of dynamic environmental objects are considered as cicles with radii = max(ymax-ymin, x_max-xmin)/2
the calculation of the energy map is similar to that of agent-agent interactions
'''
def cal_em_other(current_state, pred_vel, other_locates, fov_r=200, map_h=401, map_w=401):

    em_other_rep = np.zeros((map_h, map_w))

    xij = current_state[3:5] - other_locates[:, :2]

    # we employ a disk-shape surrounding area
    other_in_fov = select_nei_in_fov_recArea(xij, fov_angle=np.pi*2, fov_r=fov_r)

    if len(other_in_fov) < 1:
        return em_other_rep

    xij = xij[other_in_fov, :]

    pred_other_nei = other_locates[other_in_fov, :2] + other_locates[other_in_fov, 2:4] * delta_t

    vij = pred_vel - other_locates[other_in_fov, 2:4]
    vij_sqr = np.sum(vij**2, axis=1) + EPSILON

    ttc = -np.sum(np.multiply(xij, vij), axis=1) / vij_sqr
    ttc[ttc<0] = delta_t
    ttc[ttc>delta_t] = delta_t

    delta_xij = copy.deepcopy(vij)
    delta_xij[:, 0] *= ttc
    delta_xij[:, 1] *= ttc
    xij_pred = delta_xij + xij
    xij_pred_sqrt = np.linalg.norm(xij_pred, axis=1) - ped_r

    energy_other_rep = np.exp(-xij_pred_sqrt/ped_r)

    map_center_index = np.array([fov_r, fov_r]).astype(int)
    curr_ped_global_index = np.round(current_state[3:5]).astype(int)
    nei_locate_norm = np.round(pred_other_nei).astype(int) - curr_ped_global_index + map_center_index

    # em_ped_anti = np.zeros((map_r*2, map_r*2))
    for i, nei_locate in enumerate(nei_locate_norm):
        nx, ny = nei_locate[0], nei_locate[1]
        if nx >= 0 and nx < map_w:
            if ny >=0 and ny < map_h:
                em_other_rep[nx, ny] += energy_other_rep[i]

    return em_other_rep

def cal_em_veh(current_state, pred_vel, current_other_nei, current_other_scene, ped_vision_area, fov_r=200, veh_h=130, veh_w=55, map_h=401, map_w=401):
    scene_h, scene_w = current_other_scene.shape[0], current_other_scene.shape[1]
    curr_ped_int = current_state[3:5].astype(int)
    vision_xmin, vision_xmax, vision_ymin, vision_ymax = ped_vision_area

    wmin, wmax, hmin, hmax = max(vision_xmin, 0), min(vision_xmax, scene_w), max(vision_ymin, 0), min(vision_ymax, scene_h)

    curr_other_locate_norm = np.argwhere(current_other_scene[hmin:hmax, wmin:wmax] > 0)
    em_veh = np.zeros((map_h, map_w))

    if len(curr_other_locate_norm) < 1:
        return em_veh

    # cal the accurate relative index in the ped_for (401*401 map with (200, 200) as center)
    if vision_xmin < 0: # width index
        curr_other_locate_norm[:, 1] += abs(vision_xmin)
    if vision_ymin < 0: # height index
        curr_other_locate_norm[:, 0] += abs(vision_ymin)

    # flip the image index to 2d-cord of the ped
    curr_other_locate_norm = np.flip(curr_other_locate_norm, axis=1)

    ped_locate_norm = np.array((fov_r, fov_r)).astype(int)
    curr_other_pos = curr_other_locate_norm - ped_locate_norm + curr_ped_int

    pred_other_pos = np.zeros((curr_other_pos.shape[0], 2))
    veh_vel = np.zeros((curr_other_pos.shape[0], 2))
    for i, curr_veh in enumerate(curr_other_pos):
        veh_idw, veh_idh = curr_veh
        veh_id = int(current_other_scene[veh_idh, veh_idw]) - 1
        if veh_id < 0:
            print('error energymap veh')
            return -1
        veh_vel[i] = current_other_nei[veh_id, 5:7]
        pred_other_pos[i] = curr_veh + current_other_nei[veh_id, 5:7] * delta_t

    pred_other_dis_curr = np.linalg.norm(current_state[3:5]-pred_other_pos, axis=1, keepdims=True)

    if len(pred_other_dis_curr) < 1:
        return em_veh

    curr_other_pos = curr_other_pos[pred_other_dis_curr < fov_r]
    veh_vel = veh_vel[pred_other_dis_curr < fov_r]
    pred_other_pos = pred_other_pos[pred_other_dis_curr < fov_r]

    pred_other_pos = np.round(pred_other_pos).astype(int)
    pred_other_locate_norm = pred_other_pos - curr_ped_int + np.array([fov_r, fov_r])

    xij = current_state[3:5] - curr_other_pos
    vij = pred_vel - veh_vel
    vij_sqr = np.sum(vij**2, axis=1) + EPSILON

    ttc = -np.sum(np.multiply(xij, vij), axis=1) / vij_sqr
    ttc[ttc<0] = delta_t
    ttc[ttc>delta_t] = delta_t
    delta_xij = copy.deepcopy(vij)
    delta_xij[:, 0] *= ttc
    delta_xij[:, 1] *= ttc
    xij_pred = delta_xij + xij
    xij_pred_sqrt = np.linalg.norm(xij_pred, axis=1) - ped_r

    energy_other_rep = np.exp(-xij_pred_sqrt/ped_r)

    # energy_other_rep = np.zeros(len(xij_pred_sqrt))
    # energy_other_rep[xij_pred_sqrt > 0] = 1 / xij_pred_sqrt[xij_pred_sqrt > 0]
    # energy_other_rep[xij_pred_sqrt <= 0] = 1000000
    for i, obs_locate in enumerate(pred_other_locate_norm):
        em_veh[obs_locate[0], obs_locate[1]] += energy_other_rep[i]

    return em_veh

def cal_energymap_agent(current_state, pred_vel, semantic_map, current_other_nei, current_ped_nei, fov_r=50, map_h=100, map_w=100):
    ped_vision_area = cal_ped_vision_area(int(round(current_state[3])), int(round(current_state[4])), fov_r)

    if len(current_ped_nei) > 0:
        pred_ped_nei = current_ped_nei[:, 3:5] + current_ped_nei[:, 5:7] * delta_t
        e_ped = cal_em_ped(current_state, pred_vel, current_ped_nei, pred_ped_nei, fov_r=fov_r, map_h=map_h, map_w=map_w)
    else:
        e_ped = np.zeros((map_h, map_w))

    if len(current_other_nei) > 0:
        e_other = cal_em_other(current_state, pred_vel, current_other_nei, fov_r=fov_r, map_h=map_h, map_w=map_w)
    else:
        e_other = np.zeros((map_h, map_w))


    e_obs = cal_em_envobs(current_state, pred_vel, semantic_map, ped_vision_area, fov_r=fov_r, map_h=map_h, map_w=map_w)

    return np.array((e_ped, e_obs, e_other))


def cal_energymap(current_state, pred_vel, current_ped_nei, current_ped_supp_nei, current_other_nei, semantic, fov_r=50, map_h=100, map_w=100, device='cpu'):
    current_ped_neighbor = np.vstack((current_ped_nei, current_ped_supp_nei))

    energy_map = cal_energymap_agent(current_state, pred_vel, 
                                     semantic,
                                     current_other_nei, 
                                     current_ped_neighbor,
                                     fov_r=fov_r,
                                     map_h=map_h, map_w=map_w)

    return energy_map

def generate_velspace(curr_vel, angle_interval=9, mag_intertval=3, mag_split=0.2, fov_ang = np.pi):
    curr_speed = np.linalg.norm(curr_vel, axis=1, keepdims=True)

    curr_dir = curr_vel / (curr_speed+1e-6)

    zero_id = np.argwhere(curr_speed==0)
    if len(zero_id) > 1:
        curr_dir[zero_id[:,0], :] = np.array([1,0])

    speed_space = np.zeros((curr_vel.shape[0], angle_interval, mag_intertval, 1))
    dir_space = np.zeros((curr_vel.shape[0], angle_interval, mag_intertval, 2))
    angle_split = fov_ang / (angle_interval-1)
    
    splited_angles = [angle_split*(angle_interval//2-i) for i in range(angle_interval)]
    splited_speeds = [mag_split*(mag_intertval//2-i) for i in range(mag_intertval)]

    for i, rot_angle in enumerate(splited_angles):
        if rot_angle == 0:
            new_dir = curr_dir
        else:
            new_dir = rotate_dir_vector(curr_dir, rot_angle) # N 2
        new_dir = np.tile(new_dir, (1, mag_intertval)) # N mag_inter*2
        new_dir = np.reshape(new_dir, (curr_vel.shape[0], mag_intertval, -1))
        dir_space[:, i, :] = new_dir

    for j, acc_speed in enumerate(splited_speeds):
        new_speed = curr_speed + acc_speed
        new_speed = np.tile(new_speed, (1, angle_interval))
        new_speed = np.reshape(new_speed, (curr_vel.shape[0], angle_interval, -1))

        dzero_speed_id = np.argwhere(new_speed < 0)
        if len(dzero_speed_id) > 0:
            curr_max_speed = np.max(speed_space, axis=-2)
            new_speed[dzero_speed_id[:,0], dzero_speed_id[:,1]] = curr_max_speed[dzero_speed_id[:,0], dzero_speed_id[:,1]] + mag_split

        speed_space[:, :, j] = new_speed

    vel_space = dir_space * speed_space
    vel_space = np.reshape(vel_space, (vel_space.shape[0], -1, vel_space.shape[-1]))

    return vel_space

def rotate_dir_vector(vector,theta):
    
    new_vector = np.zeros_like(vector) 
    new_vector[:, 0] = (vector[:, 0]*np.cos(theta)) - (vector[:, 1]*np.sin(theta))
    new_vector[:, 1] = (vector[:, 1]*np.cos(theta)) + (vector[:, 0]*np.sin(theta))        
         
    return new_vector