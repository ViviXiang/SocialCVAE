from this import d
import numpy as np
import copy
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from utils.preprocessing_sdd import params, ped_r, ped_r_plus, delta_t

def calculate_loss(fine_pred_trajs, coarse_pred_trajs,
                   gt_coarse_trajs, gt_trajs,
                   coarse_vel_scores, gt_labels,
                   criterion_traj, criterion_score):
    
    # reconstruction loss
    coarse_adl_loss = criterion_traj(coarse_pred_trajs, gt_coarse_trajs) # better with l2 loss
    fine_adl_loss = criterion_traj(fine_pred_trajs, gt_trajs) # better with l2 loss
    # cross entropy loss
    cel_loss = criterion_score(coarse_vel_scores, gt_labels)
    
    total_loss = coarse_adl_loss + fine_adl_loss + cel_loss
    return coarse_adl_loss, fine_adl_loss, cel_loss, total_loss

def calculate_loss_coarse(coarse_pred_trajs, gt_coarse_trajs,
                   coarse_vel_scores, gt_labels,
                   criterion_traj, criterion_score,
                   gt_trajs, fine_pred_trajs=None):     
    # reconstruction loss
    coarse_adl_loss = criterion_traj(coarse_pred_trajs, gt_coarse_trajs) # better with l2 loss
    if fine_pred_trajs is None:
        fine_adl_loss = criterion_traj(coarse_pred_trajs, gt_trajs)
    else:
        fine_adl_loss = criterion_traj(fine_pred_trajs, gt_trajs)
    # cross entropy loss
    cel_loss = criterion_score(coarse_vel_scores, gt_labels)

    total_loss = coarse_adl_loss + fine_adl_loss + cel_loss
    return coarse_adl_loss, fine_adl_loss, cel_loss, total_loss

def calculate_loss_coarse1(coarse_pred_trajs, gt_trajs,
                   coarse_vel_scores, gt_labels,
                   criterion_traj, criterion_score,
                   fine_pred_trajs=None):
    # reconstruction loss
    coarse_adl_loss = criterion_traj(coarse_pred_trajs, gt_trajs) # better with l2 loss

    cel_loss = criterion_score(coarse_vel_scores, gt_labels)

    if fine_pred_trajs is None:
        total_loss = coarse_adl_loss + cel_loss
        return coarse_adl_loss, cel_loss, total_loss

    fine_adl_loss = criterion_traj(fine_pred_trajs, gt_trajs)
    total_loss = coarse_adl_loss + fine_adl_loss + cel_loss

    return coarse_adl_loss, fine_adl_loss, cel_loss, total_loss


def calculate_loss_cvae(mean, log_var, criterion, future, predictions):
    # reconstruction loss
    ADL_traj = criterion(future, predictions) # better with l2 loss
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    total_loss = ADL_traj + KLD

    return KLD, ADL_traj, total_loss

def rotate_dir_vector(vector,theta):
    
    new_vector = np.zeros_like(vector) 
    new_vector[:, 0] = (vector[:, 0]*np.cos(theta)) - (vector[:, 1]*np.sin(theta))
    new_vector[:, 1] = (vector[:, 1]*np.cos(theta)) + (vector[:, 0]*np.sin(theta))        
         
    return new_vector

def generate_velspace_full(curr_vel, angle_interval=8, mag_intertval=3, mag_split=0.2):
    curr_speed = np.linalg.norm(curr_vel, axis=1, keepdims=True)

    curr_dir = curr_vel / (curr_speed+1e-6)

    zero_id = np.argwhere(curr_speed==0)
    if len(zero_id) > 1:
        curr_dir[zero_id[:,0], :] = np.array([1,0])

    speed_space = np.zeros((curr_vel.shape[0], angle_interval, mag_intertval, 1))
    dir_space = np.zeros((curr_vel.shape[0], angle_interval, mag_intertval, 2))
    angle_split = 2*np.pi / angle_interval
    
    splited_angles = [angle_split*i for i in range(angle_interval)]
    splited_speeds = [mag_split*(mag_intertval//2-i) for i in range(mag_intertval)]

    for i, rot_angle in enumerate(splited_angles):
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

def normalize_data(data, first_frame=[]):
    if len(first_frame) == 0:
        first_frame = data[:, 0, :]
    out_data = copy.deepcopy(data)
    for i, first_pos in enumerate(first_frame):
        out_data[i, :, :2] = data[i, :, :2] - first_pos[:2]

    return out_data


EPSILON = 10e-6

def zero_softmax(x, dim=0, eps=1e-5):
    x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    x = x_exp / (x_exp_sum + eps)

    return x

def linear_normalize(x):
    min_x = np.min(x, dim=0, keepdims=True)
    max_x = np.max(x, dim=0,  keepdims=True)
    x = (x - min_x) / (max_x - min_x + EPSILON)
    
    return x
    
def cal_total_energy(current_state, candidate_vel,
                     curr_ped_supp, 
                     curr_veh_supp, 
                     semantic, 
                     k_vel, k_env, k_veh,k_ped,
                     device='cpu'):
    """
    current_state: N 4
    candidate_vel: N n 2
    curr_ped_supp: N N_ped 4
    curr_veh_supp N N_veh 4
    semantic H W
    """
    e_rep_env = cal_energy_pedenv(current_state, candidate_vel, semantic, device=device)
    e_vel = cal_energy_velContinuity(current_state[:,5:7], candidate_vel, device=device)
    e_rep_ped = cal_energy_pedped(current_state, candidate_vel, curr_ped_supp, device=device)
    e_rep_veh = cal_energy_pedveh(current_state, candidate_vel, curr_veh_supp, device=device)
    
    energy_mask = 1/(k_vel*e_vel + k_ped*e_rep_ped + k_env*e_rep_env + k_veh*e_rep_veh + 1)
    # energy_mask = zero_softmax(energy_mask, dim=-1)
    return energy_mask

def cal_energy_velContinuity(current_vel, candidate_vel, device='cpu'):
    vel_diff_vec = current_vel[:, np.newaxis, :] - candidate_vel
    energy = np.linalg.norm(vel_diff_vec, axis=-1)
    
    return torch.DoubleTensor(energy).to(device)

def cal_energy_pedped(current_state, candidate_vel, curr_ped_supp, device='cpu'):
    # candi_num = candidate_vel.shape[1]
    ped_num = current_state.shape[0]
    supp_num = curr_ped_supp.shape[0]
    nei_num = ped_num+supp_num-1

    current_nei = np.zeros((ped_num, nei_num, 4))

    for i in range(ped_num):
        ped_id = current_state[i,0]
        current_nei[i] = np.concatenate((current_state[current_state[:,0]!=ped_id, 3:7], curr_ped_supp[:,3:7]), axis=0)
    # current_nei = current_nei[:, np.newaxis, :, :] # N 1 nei_num 4

    xij = current_state[:,np.newaxis,3:5] - current_nei[:,:,:2] # N nei_num 2

    dis_xij = np.linalg.norm(xij, axis=-1) # N nei_num
    ped_in_fov = np.argwhere(dis_xij <= params['ped_fov_r'])

    if len(ped_in_fov)<1:
        return torch.zeros(current_state.shape[0], candidate_vel.shape[1]).to(device)

    energy = np.zeros((current_state.shape[0], candidate_vel.shape[1]))

    nei_idx_0, nei_idx_1 = ped_in_fov[:,0], ped_in_fov[:,1]
    xij = xij[nei_idx_0, nei_idx_1] # len(ped_in_fov) 2
    # xij = np.repeat(xij, (1, candi_num))
    # xij
    current_nei_vel = current_nei[nei_idx_0, nei_idx_1, 2:] # len(ped_in_fov) 2
    vij = candidate_vel[nei_idx_0, :, :] - current_nei_vel[:,np.newaxis,:] # len(ped_in_fov) candi_num 2

    # assert xij.shape==vij.shape

    vij_sqr = np.sum(vij**2, axis=-1) + EPSILON # len(ped_in_fov) candi_num
    ttc = -np.sum(np.multiply(xij[:,np.newaxis,:], vij), axis=-1) / vij_sqr
    ttc[ttc<0] = delta_t
    ttc[ttc>delta_t] = delta_t
    delta_xij = vij * ttc[:,:,np.newaxis]
    xij_pred = delta_xij + xij[:,np.newaxis,:]
    xij_pred_sqrt = np.linalg.norm(xij_pred, axis=-1) # len(ped_in_fov) candi_num
    energys = np.exp(-xij_pred_sqrt/ped_r)

    for i in range(ped_num):
        energy[i,:] += np.sum(energys[nei_idx_0==i, :], axis=0)

    return torch.DoubleTensor(energy).to(device)

def cal_energy_pedenv(current_state, candidate_vel, semantic_map, device='cpu'):

    scene_h, scene_w = semantic_map.shape[0], semantic_map.shape[1]

    energy = np.zeros((current_state.shape[0], candidate_vel.shape[1]))
    xi_locate = np.zeros_like(current_state[:,:2]) + np.array([params["ped_fov_r"], params["ped_fov_r"]])
    for i in range(current_state.shape[0]):
        vision_xmin, vision_xmax, vision_ymin, vision_ymax =\
            cal_ped_vision_area(int(round(current_state[i, 3])), int(round(current_state[i, 4])), params["ped_fov_r"])

        if vision_ymax < 0 or vision_xmax < 0:
            continue
        if vision_xmin > scene_w-1 or vision_ymin > scene_w-1:
            continue

        wmin, wmax, hmin, hmax = max(vision_xmin, 0), min(vision_xmax, scene_w), max(vision_ymin, 0), min(vision_ymax, scene_h)
        local_semantic = semantic_map[hmin:hmax, wmin:wmax]

        obs_locate_norm = np.argwhere(local_semantic == 3)
        if len(obs_locate_norm) < 1:
            continue
        # flip the image index to 2d-cord of the ped
        obs_locate_norm = np.flip(obs_locate_norm, axis=1) # nei_num 2
        vj = np.zeros_like(obs_locate_norm) # nei_num 2
        xij = xi_locate[i,:] - obs_locate_norm # nei_num 2
        vij = candidate_vel[i, :, np.newaxis, :] - vj[np.newaxis, :,:] # candi_num nei_num 2
        vij_sqr = np.sum(vij**2, axis=-1) + EPSILON # candi_num nei_num
        ttc = -np.sum(np.multiply(xij[np.newaxis,:,:], vij), axis=-1) / vij_sqr # candi_num nei_num
        ttc[ttc<0] = delta_t
        ttc[ttc>delta_t] = delta_t
        delta_xij = vij * ttc[:,:,np.newaxis] # candi_num nei_num 2
        xij_pred = delta_xij + xij[np.newaxis,:,:] # candi_num nei_num 2
        xij_pred_sqrt = np.linalg.norm(xij_pred, axis=-1) # candi_num nei_num
        energys = np.exp(-xij_pred_sqrt/ped_r)
        energy[i,:] = np.sum(energys, axis=-1)

    return torch.DoubleTensor(energy).to(device)

def cal_energy_pedveh(current_state, candidate_vel, other_locates, device='cpu'):
    if len(other_locates)<1:
        return torch.zeros(current_state.shape[0], candidate_vel.shape[1]).to(device)
    
    current_nei = other_locates[:,:4] # nei_num 4

    xij = current_state[:,np.newaxis,3:5] - current_nei[np.newaxis,:,:2] # N nei_num 2

    dis_xij = np.linalg.norm(xij, axis=-1) # N nei_num
    ped_in_fov = np.argwhere(dis_xij <= params['ped_fov_r'])
    if len(ped_in_fov) < 1:
        return torch.zeros(current_state.shape[0], candidate_vel.shape[1]).to(device)

    energy = np.zeros((current_state.shape[0], candidate_vel.shape[1])) # N candi_num col_num

    nei_idx_0, nei_idx_1 = ped_in_fov[:,0], ped_in_fov[:,1]
    xij = xij[nei_idx_0, nei_idx_1] # len(ped_in_fov) 2

    current_nei_vel = current_nei[nei_idx_1, 2:] # len(ped_in_fov) 2
    vij = candidate_vel[nei_idx_0, :, :] - current_nei_vel[:,np.newaxis,:] # len(ped_in_fov) candi_num 2

    vij_sqr = np.sum(vij**2, axis=-1) + EPSILON # len(ped_in_fov) candi_num
    ttc = -np.sum(np.multiply(xij[:,np.newaxis,:], vij), axis=-1) / vij_sqr
    ttc[ttc<0] = delta_t
    ttc[ttc>delta_t] = delta_t
    delta_xij = vij * ttc[:,:,np.newaxis]
    xij_pred = delta_xij + xij[:,np.newaxis,:]
    xij_pred_sqrt = np.linalg.norm(xij_pred, axis=-1) # len(ped_in_fov) candi_num
    energys = np.exp(-xij_pred_sqrt/ped_r)

    for i in range(current_state.shape[0]):
        energy[i,:] += np.sum(energys[nei_idx_0==i, :], axis=0)

    return torch.DoubleTensor(energy).to(device)


def select_nei_in_fov(current_state, xij, dis_ij, fov_angle=2*np.pi, fov_r=200):
    '''We regard the fov of an agent as a sector:
        the default fov is a circle around the ped
    '''
    if fov_angle == 2*np.pi:
        return np.where(dis_ij <= fov_r)[0]
    curr_v = np.linalg.norm(current_state[5:7])
    curr_dir = current_state[5:7] / curr_v
    norm_dis_ij = -xij / dis_ij
    angle_ij = np.arccos(norm_dis_ij * curr_dir)
    return np.where(angle_ij <= fov_angle/2 & dis_ij <= fov_r)[0]

def select_nei_in_fov_recArea(xij, fov_angle=2*np.pi, fov_r=200):
    '''
    We regard the fov of an agent as a rectangle:
    the default is a square with the current agent as center
    '''
    if fov_angle == 2*np.pi:
        abs_xij_max = np.max(np.abs(xij), axis=1)
        return np.where(abs_xij_max <= fov_r)[0]
    elif fov_angle == np.pi:
        # TBD
        return np.arange(0, len(xij))


'''
Function cal_ped_vision_area(): calculate the ped's sense area
curr_x: the ped's x coordination, corresponding to the row index of the semantic map
curr_y: the ped's y corrdination, corresponding to the column index of the semantic map
'''
# def cal_ped_vision_area(curr_x, curr_y, fov_r, scene_h, scene_w):
#     x_min = max(curr_x-fov_r, 0)
#     x_max = min(curr_x+fov_r + 1, scene_w)
#     y_min = max(curr_y-fov_r, 0)
#     y_max = min(curr_y+fov_r + 1, scene_h)
#
#     return np.array([x_min, x_max, y_min, y_max])

def cal_ped_vision_area(curr_x, curr_y, fov_r):
    x_min = curr_x - fov_r
    x_max = curr_x + fov_r + 1
    y_min = curr_y - fov_r
    y_max = curr_y + fov_r + 1

    return np.array([x_min, x_max, y_min, y_max])

def cal_ped_vision_area_tensor(curr_x, curr_y, fov_r):
    x_min = curr_x - fov_r
    x_max = curr_x + fov_r + 1
    y_min = curr_y - fov_r
    y_max = curr_y + fov_r + 1

    return torch.Tensor(([x_min, x_max, y_min, y_max]), dtype=torch.int32)


# def cal_nei_other_area(current_state, nx, ny, map_h, map_w):
#     x_min = max(nx - ped_r, 0)
#     x_max = min(nx + ped_r + 1, map_w-1)
#     y_min = max(ny - ped_r, 0)
#     y_max = min(ny + ped_r + 1, map_h-1)
#
#     return x_min, x_max, y_min, y_max

def cal_nei_ped_area(nx, ny, ped_r, map_h, map_w):
    x_min = max(nx - ped_r, 0)
    x_max = min(nx + ped_r + 1, map_w-1)
    y_min = max(ny - ped_r, 0)
    y_max = min(ny + ped_r + 1, map_h-1)

    return x_min, x_max, y_min, y_max

def cal_em_ped(current_state, pred_vel, current_ped_nei, pred_ped_nei, fov_r=200, map_h=401, map_w=401):
    em_ped_rep = np.zeros((map_h, map_w))

    xij = current_state[3:5] - current_ped_nei[:, 3:5]
    abs_xij_max = np.max(np.abs(xij), axis=1)
    ped_in_fov = np.where(abs_xij_max <= fov_r)[0]
    if len(ped_in_fov) < 1:
        return em_ped_rep

    xij = xij[ped_in_fov, :]
    # xij_sqrt = np.linalg.norm(xij, axis=1)
    # xij_sqr = xij_sqrt ** 2
    # we employ a disk-shape surrounding area
    # ped_in_fov = select_nei_in_fov(current_state, xij, xij_sqrt, fov_angle=np.pi*2, fov_r=fov_r)



    # xij = xij[ped_in_fov, :]
    # xij_sqr = xij_sqr[ped_in_fov, :]
    vij = pred_vel - current_ped_nei[ped_in_fov, 5:7]
    vij_sqr = np.sum(vij**2, axis=1) + EPSILON

    # # for anticipatory collision avoidance
    # xij_dot_vij = sum(xij * vij)
    # time_to_collision = (-xij_dot_vij - math.sqrt(xij_dot_vij*xij_dot_vij - vij_sqr*(xij_sqr-ped_r_plus_sqr))) / vij_sqr
    # enery_ped_anti = np.exp(-time_to_collision / ttc_crop)
    # for repulsion

    ttc = -np.sum(np.multiply(xij, vij), axis=1) / vij_sqr
    ttc[ttc<0] = delta_t
    ttc[ttc>delta_t] = delta_t
    # ttc = np.clip(ttc, 0, delta_t)
    delta_xij = copy.deepcopy(vij)
    delta_xij[:, 0] *= ttc
    delta_xij[:, 1] *= ttc
    xij_pred = delta_xij + xij
    xij_pred_sqrt = np.linalg.norm(xij_pred, axis=1) - ped_r_plus


    energy_ped_rep = np.exp(-xij_pred_sqrt/ped_r)
    # energy_ped_rep = np.zeros(len(xij_pred_sqrt))

    # energy_ped_rep[xij_pred_sqrt > 0] = 1 / xij_pred_sqrt[xij_pred_sqrt > 0]
    # energy_ped_rep[xij_pred_sqrt <= 0] = 1000000

    # energy_ped_rep[]
    map_center_index = np.array([fov_r, fov_r]).astype(int)
    curr_ped_global_index = np.round(current_state[3:5]).astype(int)
    nei_locate_norm = np.round(pred_ped_nei[ped_in_fov]).astype(int) - curr_ped_global_index + map_center_index

    # em_ped_anti = np.zeros((map_r*2, map_r*2))

    for i, nei_locate in enumerate(nei_locate_norm):
        nx, ny = nei_locate[0], nei_locate[1]
        # for anticipatory collision avoidance
        # em_ped_anti[nx, ny] = enery_ped_anti[i]
        # for repulsion
        nx_min, nx_max, ny_min, ny_max = cal_nei_ped_area(nx, ny, ped_r, map_h, map_w)
        em_ped_rep[nx_min:nx_max+1, ny_min:ny_max+1] += energy_ped_rep[i]

    return em_ped_rep

def cal_em_envobs(current_state, pred_vel, semantic_map, ped_vision_area, fov_r=200, map_h=401, map_w=401):
    scene_h, scene_w = semantic_map.shape[0], semantic_map.shape[1]
    # curr_pos_int = np.round(current_state[3:5]).astype(int)
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

    # the global coordinate of the obstacles in the fov of this ped
    # obs_pos = obs_locate_norm - np.array([fov_r, fov_r]) + curr_pos_int

    # center_index = np.array((fov_r, fov_r)).astype(int)
    # obs_locate_norm = obs_index - curr_pos_int + center_index
    # obs_locate_norm = np.vstack((obs_index1, obs_index2))
    ped_locate_norm = np.array([fov_r, fov_r])
    xij = ped_locate_norm - obs_locate_norm
    vij = pred_vel
    vij_sqr = np.sum(vij**2) + EPSILON

    ttc = -np.sum(np.multiply(xij, vij), axis=1) / vij_sqr
    ttc[ttc<0] = delta_t
    ttc[ttc>delta_t] = delta_t
    # ttc = np.clip(ttc, 0, delta_t)
    delta_xij = np.zeros((len(ttc), 2))

    for i in range(len(ttc)):
        delta_xij[i, :] = ttc[i] * vij
    # delta_xij[0] *= ttc
    # delta_xij[1] *= ttc
    xij_pred = delta_xij + xij
    xij_pred_sqrt = np.linalg.norm(xij_pred, axis=1) - ped_r
    energy_obs_rep = np.exp(-xij_pred_sqrt/ped_r)

    # energy_obs_rep = np.zeros(len(xij_pred_sqrt))
    # energy_obs_rep[xij_pred_sqrt > 0] = 1 / xij_pred_sqrt[xij_pred_sqrt > 0]
    # energy_obs_rep[xij_pred_sqrt <= 0] = 1000000
    for i, obs_locate in enumerate(obs_locate_norm):
        em_envobs[obs_locate[0], obs_locate[1]] += energy_obs_rep[i]

    return em_envobs

def cal_other_radius(other_state):
    x_r = (other_state[:, 9] - other_state[:, 7])/2
    y_r = (other_state[:, 10] - other_state[:, 8])/2
    r = np.max(np.array([x_r, y_r]), axis=0)
    return r

def cal_other_locate_edge(other_states):
    '''
    This func is used for obtaining the pixels of the bounding box of each supplement agents
    :param other_states:
    :return: the pixels of the bounding boxes -- x, y, vx, vy
    '''
    locate_states = []
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

    locate_states = np.concatenate(locate_states, axis=0)
    return locate_states

def cal_other_locate_area(other_states):
    '''
    :param other_states:
    :return: pixel_num * 4 -> x, y, vx, vy
    '''
    locate_states = []
    for i in range(len(other_states)):
        x_length = int(other_states[i, 9]) - int(other_states[i, 7]) + 1
        x_coord = np.arange(int(other_states[i, 7]), int(other_states[i, 9])+1)
        y_length = int(other_states[i, 10]) - int(other_states[i, 8]) + 1
        y_coord = np.arange(int(other_states[i, 8]), int(other_states[i, 10])+1)

        agent_locate = np.zeros((x_length*y_length, 4))
        # agent_locate[:, 0] = other_states[i, 0]
        for j, x in enumerate(x_coord):
            agent_locate[j*y_length:(j+1)*(y_length), 0] = x
            agent_locate[j*y_length:(j+1)*(y_length), 0] = y_coord
            agent_locate[j*y_length:(j+1)*(y_length), 2:] = other_states[i, 5:7]
        locate_states.append(agent_locate)

    locate_states = np.concatenate(locate_states, axis=0)
    return locate_states


'''
In the SDD dataset, other types of dynamic environmental objects are considered as cicles with radii = max(ymax-ymin, x_max-xmin)/2
the calculation of the energy map is similar to that of agent-agent interactions
'''
def cal_em_other(current_state, pred_vel, other_locates, fov_r=200, map_h=401, map_w=401):

    em_other_rep = np.zeros((map_h, map_w))

    # other_locates = cal_other_locate_edge(current_other_nei)
    # xij_sqrt = np.linalg.norm(all_xij, axis=1)
    # xij_sqr = xij_sqrt ** 2
    xij = current_state[3:5] - other_locates[:, :2]
    # we employ a disk-shape surrounding area
    other_in_fov = select_nei_in_fov_recArea(xij, fov_angle=np.pi*2, fov_r=fov_r)

    if len(other_in_fov) < 1:
        return em_other_rep

    xij = xij[other_in_fov, :]
    # xij_sqrt = np.linalg.norm(xij, axis=1)
    # ped_other_rplus = ped_r

    pred_other_nei = other_locates[other_in_fov, :2] + other_locates[other_in_fov, 2:4] * delta_t

    # xij_sqr = xij_sqr[ped_in_fov, :]
    vij = pred_vel - other_locates[other_in_fov, 2:4]
    vij_sqr = np.sum(vij**2, axis=1) + EPSILON

    ttc = -np.sum(np.multiply(xij, vij), axis=1) / vij_sqr
    ttc[ttc<0] = delta_t
    ttc[ttc>delta_t] = delta_t
    # ttc = np.clip(ttc, 0, delta_t)
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

def cal_energymap_agent(current_state, pred_vel, semantic_map, current_other_nei, current_ped_nei, fov_r=200, map_h=401, map_w=401):
    ped_vision_area = cal_ped_vision_area(int(round(current_state[3])), int(round(current_state[4])), fov_r)

    if len(current_ped_nei) > 0:
        pred_ped_nei = current_ped_nei[:, 3:5] + current_ped_nei[:, 5:7] * delta_t
        e_ped = cal_em_ped(current_state, pred_vel, current_ped_nei, pred_ped_nei, fov_r=fov_r, map_h=map_h, map_w=map_w)
    else:
        e_ped = np.zeros((map_h, map_w))

    if len(current_other_nei) > 0:
        e_other = cal_em_other(current_state, pred_vel, current_other_nei, fov_r=fov_r, map_h=map_h, map_w=map_w)
        # pred_other_nei = current_other_nei[:, 3:5] + current_other_nei[:, 5:7] * delta_t
        # e_veh = cal_em_veh(current_state, pred_vel, current_other_nei, pred_other_nei, semantic_map, ped_vision_area, fov_r=params['ped_fov_r'],
        #                veh_h=params['veh_h'], veh_w=params['veh_w'], map_h=map_h, map_w=map_w)
    else:
        e_other = np.zeros((map_h, map_w))


    e_obs = cal_em_envobs(current_state, pred_vel, semantic_map, ped_vision_area, fov_r=fov_r, map_h=map_h, map_w=map_w)


    return np.array((e_ped, e_obs, e_other))


def cal_energymap(current_state, pred_vel, current_ped_nei, current_ped_supp_nei, current_other_nei, semantic, fov_r=200, map_h=401, map_w=401, device='cpu'):
    current_ped_neighbor = np.vstack((current_ped_nei, current_ped_supp_nei))

    energy_map = cal_energymap_agent(current_state, pred_vel, 
                                     semantic,
                                     current_other_nei, 
                                     current_ped_neighbor,
                                     fov_r=fov_r,
                                     map_h=map_h, map_w=map_w)

    return energy_map