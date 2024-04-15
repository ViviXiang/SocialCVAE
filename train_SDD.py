import time

import torch
import torch.optim as optim
from model_coarse2fine import *
from utils.preprocessing_sdd import args, params, train_data, test_data, train_images, test_images, delta_t, train_other_edge, test_other_edge
from utils.utils_sdd import cal_energymap, generate_velspace, calculate_loss_cvae
import numpy as np
from torch.autograd import Variable
import copy
from tqdm import tqdm
import os


dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)

print(params)

seq_len = params['past_length'] + params['future_length']

fov_r = params['ped_fov_r'] # the range of a pedestrian's neighborhood
energymap_h = fov_r * 2
energymap_w = energymap_h

fov_ang = np.pi/3*2

angle_interval = params['angle_interval']
mag_intertval = params['mag_intertval']
candiadate_num = angle_interval * mag_intertval
mag_split=params['mag_split']

model = heter_nn(params["lstm_input_size"], params["lstm_embedding_size"], params["lstm_hidden_size"],
                 params["lstm_output_size"], params['future_length'],
                 params['cvae_enc_past_size'], params["cvae_enc_emap_size"],
                 params["cvae_enc_bias_size"], params["cvae_enc_latent_size"], params["cvae_dec_size"],
                 params["cvae_fdim"], params["cvae_zdim"], params["cvae_sigma"],
                 params["past_length"], params["future_length"], energymap_h, energymap_w)

best_test_loss = 9999999

k_veh = torch.tensor(1.0).to(device)
k_obs = torch.tensor(1.0).to(device)

model = model.double().to(device)
k_veh.requires_grad = True
k_obs.requires_grad = True

optimizer = optim.Adam([{'params': model.parameters()},
                        {'params': k_veh},
                        {'params': k_obs}],
                       lr= params["learning_rate"])

sample_N = params['sample_N']

criterion = nn.MSELoss()

def train(epoch):
    model.train()
    total_loss = 0

    batchs = range(len(train_data))
    with tqdm(batchs) as epoch_bar:
        for batch, (train_traj, ped_supp, _), batch_semantic, other_supp in zip(epoch_bar, train_data, train_images, train_other_edge):

            epoch_bar.set_description(f"train epoch {epoch}")
            num_peds = train_traj.shape[0]
            ped_traj = copy.deepcopy(train_traj)
            ped_traj = ped_traj.astype(float)

            traj = torch.DoubleTensor(ped_traj).to(device)
           
            coarse_predictions = torch.zeros(num_peds, params['future_length'], 2).to(device)
            hidden_states = Variable(torch.zeros(num_peds, params['lstm_hidden_size']))
            cell_states = Variable(torch.zeros(num_peds, params['lstm_hidden_size']))
            hidden_states = hidden_states.to(device)
            cell_states = cell_states.to(device)

            for m in range(params['past_length']):
                current_step = traj[:, m, 3:5]
                current_vel = traj[:, m, 5:7]
                input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
                lstm_pred_vel, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)

            total_loss_cvae = 0
            current_t = params['past_length'] - 1
            for t in range(params['future_length']):
                if t > 0:
                    input_lstm = torch.cat((current_step, current_vel), dim=1)
                    lstm_pred_vel, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)

                candidate_vels = generate_velspace(lstm_pred_vel.clone().cpu().detach().numpy(),
                                                   angle_interval=angle_interval, mag_intertval=mag_intertval,
                                                   mag_split=mag_split, fov_ang=fov_ang)
                candidate_vels = torch.DoubleTensor(candidate_vels).to(device)

                x_past_traj = copy.deepcopy(traj[:, t:params['past_length'] + t, 3:5])
                x_past_traj = torch.reshape(x_past_traj, (-1, x_past_traj.shape[1] * x_past_traj.shape[2])).to(device)

                coarse_pred_vel = model.forward_coarse(x_past_traj, candidate_vels, device=device)
                coarse_pred_position = current_step + coarse_pred_vel * delta_t
                coarse_predictions[:, t, :] = coarse_pred_position

                ped_supp_num = int(ped_supp[current_t, -1, -1])
                pred_vel = coarse_pred_vel.clone().cpu().detach().numpy()
                energy_maps = []
                for id in range(num_peds):
                    ped_id = ped_traj[id, current_t, 0]
                    em = cal_energymap(ped_traj[id, current_t, :], pred_vel[id, :],
                               ped_traj[ped_traj[:, current_t, 0]!=ped_id, current_t, :],
                               ped_supp[current_t, :ped_supp_num, :],
                               other_supp[current_t],
                               batch_semantic,
                               fov_r=fov_r,
                               map_h=energymap_h, map_w=energymap_w)
                    energy_maps.append(em)

                energy_maps_all = torch.DoubleTensor(np.array(energy_maps)).to(device)

                energy_map = model.forward_energy(energy_maps_all, k_obs, k_veh, energymap_h, energymap_w, device=device)

                x_energ_map = torch.reshape(energy_map, (-1, energy_map.shape[1]*energy_map.shape[2])).to(device)


                alpha = (traj[:, params['past_length'] + t, 3:5] - coarse_pred_position)

                alpha_recon, mu, var = model.forward_cvae(x_energ_map, x_past_traj, next_step=alpha, device=device)

                prediction_cvae = alpha_recon + coarse_pred_position

                current_vel = (prediction_cvae - current_step) / delta_t
                current_step = prediction_cvae
                current_t += 1
                
                kld, adl = calculate_loss_cvae(mu, var, criterion, alpha, alpha_recon)
                loss_cvae = kld * params['w_kld'] + adl * params['w_adl']

                total_loss_cvae += loss_cvae

            loss_coarse = criterion(traj[:, params['past_length']:seq_len, 3:5], coarse_predictions)
            batch_loss = loss_coarse * params['w_coarse'] + total_loss_cvae

            optimizer.zero_grad()
            batch_loss.backward()
            total_loss += batch_loss.item()
            epoch_bar.set_postfix(current_total_loss=total_loss)
            optimizer.step()
            
        return total_loss

def test(epoch):
    model.eval()
    with torch.no_grad():
        ade_all = []
        fde_all = []
        batchs = range(len(test_data))
        with tqdm(batchs) as epoch_bar:
            for batch, (test_traj, ped_supp, _), other_supp in zip(epoch_bar, test_data, test_other_edge):

                epoch_bar.set_description(f"test epoch {epoch}")
                num_peds = test_traj.shape[0]

                batch_semantic = test_images[batch]

                test_traj = test_traj.astype(float)
             
                ped_traj = copy.deepcopy(test_traj)
                traj = torch.DoubleTensor(test_traj).to(device)
                
                traj_copy = copy.deepcopy(traj)
                y = traj[:, params['past_length']:, 3:5]  # peds*future_length*2
                y = y.cpu().numpy()

                predictions = torch.zeros(num_peds, params['future_length'], 2).to(device)
                hidden_states = Variable(torch.zeros(num_peds, params['lstm_hidden_size']))
                cell_states = Variable(torch.zeros(num_peds, params['lstm_hidden_size']))
                hidden_states = hidden_states.to(device)
                cell_states = cell_states.to(device)

                for m in range(params['past_length']):
                    current_step = traj[:, m, 3:5]
                    current_vel = traj[:, m, 5:7]
                    input_lstm = torch.cat((current_step, current_vel), dim=1)  # peds*4
                    lstm_pred_vel, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)

                current_t = params['past_length'] - 1
                for t in range(params['future_length']):
                    if t > 0:
                        input_lstm = torch.cat((current_step, current_vel), dim=1)
                        lstm_pred_vel, hidden_states, cell_states = model.forward_lstm(input_lstm, hidden_states, cell_states)
                    
                    candidate_vels = generate_velspace(lstm_pred_vel.clone().cpu().detach().numpy(),
                                                   angle_interval=angle_interval, mag_intertval=mag_intertval,
                                                   mag_split=mag_split, fov_ang=fov_ang)
                    candidate_vels = torch.DoubleTensor(candidate_vels).to(device)

                    x_past_traj = copy.deepcopy(traj_copy[:, t:params['past_length'] + t, 3:5])

                    x_past_traj = torch.reshape(x_past_traj, (-1, x_past_traj.shape[1] * x_past_traj.shape[2])).to(device)

                    coarse_pred_vel = model.forward_coarse(x_past_traj, candidate_vels, device=device)
                    coarse_pred_position = current_step + coarse_pred_vel * delta_t

                    ped_supp_num = int(ped_supp[current_t, -1, -1])
                    pred_vel = coarse_pred_vel.clone().cpu().detach().numpy()
                    energy_maps = []
                    for id in range(num_peds):
                        ped_id = ped_traj[id, current_t, 0]
                        em = cal_energymap(ped_traj[id, current_t, :], pred_vel[id, :],
                                       ped_traj[ped_traj[:, current_t, 0]!=ped_id, current_t, :],
                                       ped_supp[current_t, :ped_supp_num, :],
                                       other_supp[current_t],
                                       batch_semantic,
                                       fov_r=fov_r,
                                       map_h=energymap_h, map_w=energymap_w)
                        energy_maps.append(em)
                        
                    energy_maps_all = torch.DoubleTensor(np.array(energy_maps)).to(device)

                    energy_map = model.forward_energy(energy_maps_all, k_obs, k_veh, energymap_h, energymap_w, device=device)

                    x_energ_map = torch.reshape(energy_map, (-1, energy_map.shape[1]*energy_map.shape[2])).to(device)

                    alpha_step = torch.zeros(sample_N, len(traj), 2).to(device)
                    for n in range(sample_N):
                        alpha_recon = model.forward_cvae(x_energ_map, x_past_traj, device=device)
                        alpha_step[n, :, :] = alpha_recon
                    alpha_step[-1, :, :] = torch.zeros_like(alpha_step[-1, :, :])
                    prediction_correct = alpha_step + coarse_pred_position
                    predictions_norm = torch.norm((prediction_correct - traj[:, params['past_length'] + t, 3:5]), dim=-1)
                    _, indices = torch.min(predictions_norm, dim=0)  # peds
                    prediction_cvae = prediction_correct[indices, [x for x in range(len(traj))], :]  # peds*2

                    predictions[:, t, :] = prediction_cvae

                    current_vel = (prediction_cvae - current_step) / delta_t
                    current_step = prediction_cvae
                    current_t += 1

                    # update the ped information in the next step
                    traj_copy[:, current_t, 3:5] = current_step
                    traj_copy[:, current_t, 5:7] = current_vel
                    ped_traj[:, current_t, 3:7] = traj_copy[:, current_t, 3:7].cpu().detach().numpy()

                predictions = predictions.cpu().numpy()
                test_ade = np.mean(np.linalg.norm(y - predictions, axis=2), axis=1)  # peds
                test_fde = np.linalg.norm((y[:, -1, :] - predictions[:, -1, :]), axis=1)  # peds
                ade_all.append(test_ade)
                fde_all.append(test_fde)
            ade = np.mean(np.concatenate(ade_all))
            fde = np.mean(np.concatenate(fde_all))

        return ade, fde

if not os.path.exists(f'saved_models/sdd/'):
    os.makedirs(f'saved_models/sdd/')

save_path = f'saved_models/sdd/{args.save_file}.pt'

for e in range(0, params['num_epochs']):
    # train
    # total_loss = train(e)

    # validation
    test_ade, test_fde = test(e)
    print('epoch: ', e, ' test ade: ', test_ade, ' fde: ', test_fde, '\n')

    if test_ade < best_test_loss:
        print('################## BEST PERFORMANCE EPOCH {:d} ADE {:0.2f} FDE {:0.2f} ########'.format(e, test_ade, test_fde))
        print('k_obs: ', k_obs)
        print('k_veh: ', k_veh)
        
        torch.save({
                    'model_state_dict': model.state_dict(),
                    'k_obs': k_obs,
                    'k_veh': k_veh,
                    'trained_epochs': e,
                    'best_ade': test_ade,
                    'best_fde': test_fde
                    }, save_path)
        
        print(f"Saved model to:\n{save_path}")
        best_test_loss = test_ade
        best_fde = test_fde
        best_epoch = e

