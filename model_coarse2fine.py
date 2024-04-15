import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F
import math

class MLP(nn.Module):
    """ 
        The code is based on the implementation from https://github.com/realcrane/Human-Trajectory-Prediction-via-Neural-Social-Physics
        Paper name: Human Trajectory Prediction via Neural Social Physics
        Publication: ECCV2022
    """
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

class SelfAttention(nn.Module):
    """
    The code is based on the implementation from https://github.com/lssiair/SIT
    Paper name: Social Interpretable Tree for Pedestrian Trajectory Prediction
    Publication: AAAI2022
    """

    def __init__(self, in_size, hidden_size=256, out_size=64, non_linear=True):
        super(SelfAttention, self).__init__()

        self.query = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, out_size)
        ) if non_linear else nn.Linear(in_size, out_size)

        self.key = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, out_size)
        ) if non_linear else nn.Linear(in_size, out_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, mask=None, interaction=True):

        assert len(query.shape) == 3

        query = self.query(query)  # N 1 d_model
        query = query / float(math.sqrt(query.shape[-1]))
        key = self.key(key)      # N n_candidate d_model
        attention = torch.matmul(query, key.permute(0, 2, 1))  # (batch_size, 1, n_candidate)

        if interaction is True:
            attention = F.softmax(attention, dim=-1)
            attention = F.normalize(attention.squeeze(1), p=1, dim=-1)  
            return attention.unsqueeze(1)

        if mask is not None:
            attention = attention.squeeze(1) * mask

        return attention

class heter_nn(nn.Module):

    def __init__(self, lstm_input_size, lstm_embedding_size, lstm_hidden_size, lstm_output_size, lstm_output_seq,
                 cvae_enc_past_size, cvae_enc_emap_size, cvae_enc_bias_size, cvae_enc_latent_size, cvae_dec_size, cvae_fdim,
                 cvae_zdim, cvae_sigma, past_length, future_length, emap_h, emap_w):
        '''
        The model
        '''
        super(heter_nn, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_input_embedding_size = lstm_embedding_size

        self.gcn_layer = 1
        enc_size = 64
        hidden1=1024
        hidden2 = 512
        mlp_hid_pastTraj = [hidden1, hidden2]
        mlp_hid_velspace = [hidden1, hidden2]
        self.encoder_pastTraj = MLP(input_dim=past_length*2, output_dim=enc_size, hidden_size=mlp_hid_pastTraj)
        self.encoder_velspace = MLP(input_dim=2, output_dim=enc_size, hidden_size=mlp_hid_velspace)
        self.coarse_attention = nn.ModuleList(
            [SelfAttention(enc_size, hidden_size=hidden2, out_size=enc_size) for _ in range(self.gcn_layer)]
        )

        coarse_mlp_size = [hidden1, hidden2]
        self.coarse_prediction = MLP(enc_size, lstm_output_size, coarse_mlp_size)


        # LSTM for temporal motion tendency learning
        self.lstm_cell = nn.LSTMCell(lstm_embedding_size, lstm_hidden_size)
        self.input_embedding_layer = nn.Linear(lstm_input_size, lstm_embedding_size)
        self.lstm_output_layer = nn.Linear(lstm_hidden_size, lstm_output_size)

        # self.lstm_output_layer1 = nn.Linear(lstm_hidden_size, lstm_output_seq)
        # self.lstm_output_layer2 = nn.Linear(lstm_hidden_size, lstm_output_seq)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        # CVAE for motion refinement for heter-interaction
        self.zdim = cvae_zdim
        self.fdim = cvae_fdim
        self.cvae_sigma = cvae_sigma
        self.encoder_energymap = MLP(input_dim=emap_h*emap_w, output_dim=2*cvae_fdim, hidden_size=cvae_enc_emap_size)
        self.encoder_past = MLP(input_dim=past_length*2, output_dim=cvae_fdim, hidden_size=cvae_enc_past_size)
        self.encoder_futurebias = MLP(input_dim=2, output_dim=cvae_fdim, hidden_size=cvae_enc_bias_size)
        self.encoder_cvae = MLP(input_dim=3*cvae_fdim+cvae_fdim, output_dim=2*cvae_zdim, hidden_size=cvae_enc_latent_size)
        self.decoder = MLP(input_dim=3*cvae_fdim+cvae_zdim, output_dim=2, hidden_size=cvae_dec_size)

    def forward_energy(self, energymaps, k_obs, k_env, map_h, map_w, device=torch.device('cpu')):
        energy_map = torch.zeros((energymaps.shape[0], map_h, map_w)).to(device)
        for i, (e_ped, e_obs, e_veh) in enumerate(energymaps):
            energy_map[i, :, :] = e_ped + k_obs*e_obs + k_env*e_veh
        return energy_map

    def forward_cvae(self, energymap, past_traj, next_step = None, device=torch.device('cpu')):
        """ 
        The code is based on the implementation from https://github.com/realcrane/Human-Trajectory-Prediction-via-Neural-Social-Physics
        Paper name: Human Trajectory Prediction via Neural Social Physics
        Publication: ECCV2022
        """

        # provide destination iff training
        # assert model.training
        # encode
        encode_emap = self.encoder_energymap(energymap) # ped_num*32
        encode_past = self.encoder_past(past_traj) # ped_num*16
        x_encode = torch.cat((encode_past, encode_emap), dim=1) #ped_num*48

        if not self.training:
            z = torch.Tensor(past_traj.size(0), self.zdim)
            z.normal_(0, self.cvae_sigma)

        else:
            # during training, use the destination to produce generated_dest and use it again to predict final future points
            # CVAE code
            ns_features = self.encoder_futurebias(next_step) #pednum*16
            features = torch.cat((x_encode, ns_features), dim = 1) #ped_num*64
            latent =  self.encoder_cvae(features)#ped_num*32 mean+var

            mu = latent[:, 0:self.zdim] # ped_num*16
            logvar = latent[:, self.zdim:] # ped_num*16

            var = logvar.mul(0.5).exp_() #ped_num*16
            eps = torch.DoubleTensor(var.size()).normal_()#ped_num*16
            eps = eps.to(device)
            z = eps.mul(var).add_(mu) #ped_num*16

        z = z.double().to(device)
        decoder_input = torch.cat((x_encode, z), dim = 1) #ped_num*64
        generated_np = self.decoder(decoder_input) #ped_num*2

        if self.training:
            return generated_np, mu, logvar

        return generated_np


    def forward_lstm(self, input_lstm, hidden_states, cell_states):
        #input_lstm: peds*4
        # LSTM
        input_embedded = self.relu(self.input_embedding_layer(input_lstm)) #peds*lstm_embedding_size
        h_nodes, c_nodes = self.lstm_cell(input_embedded, (hidden_states, cell_states)) #h_nodes/c_nodes: peds*lstm_hidden_size
        outputs = self.lstm_output_layer(h_nodes) #peds*lstm_output_size

        return outputs, h_nodes, c_nodes

    def forward_coarse(self, past_trajs, velocity_space, device='cpu'):
        """
        generate coarse trajectory for each timestep
        """
        past_emb = self.encoder_pastTraj(past_trajs)

        candidate_embed = self.encoder_velspace(velocity_space)

        past_emb = past_emb.unsqueeze(1)
        for i in range(self.gcn_layer):
            candidate_vel_scores = self.coarse_attention[i](past_emb, candidate_embed, interaction=True) # N x N
            past_emb = past_emb + torch.matmul(candidate_vel_scores, candidate_embed)
        coarse_pred_vel = self.coarse_prediction(past_emb.squeeze(1))  # N 2
     
        return coarse_pred_vel