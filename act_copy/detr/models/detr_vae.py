# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from .transformer_no_conditions import build_transformer_no_conditions

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names,velocity_control = False, context_length=1, conditioning_enabled=True, latent_dim=256, decoder_no_state=False):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            conditioning_enabled: if False, encoder only sees actions (no qpos/qvel) for debugging KL collapse
            decoder_no_state: if True, decoder does not receive qpos/qvel conditioning (only latent z)
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.conditioning_enabled = conditioning_enabled
        self.decoder_no_state = decoder_no_state
        hidden_dim = transformer.d_model
        self.latent_dim = latent_dim # final size of latent z
        
        self.action_head = nn.Linear(hidden_dim, 8) #outputs action predictions
        self.is_pad_head = nn.Linear(hidden_dim, 1) #outputs is_pad predictions
        self.query_embed = nn.Embedding(num_queries, hidden_dim) #query embeddings for the decoder
        self.context_length = context_length
        self.time_embed = nn.Embedding(context_length, hidden_dim)
        if backbones is not None:
            
            self.input_projections = nn.ModuleList()
            for cam_id, cam_name in enumerate(self.camera_names):
                proj = nn.Conv2d(backbones[cam_id].num_channels, hidden_dim, kernel_size=1)
                self.input_projections.append(proj)
            self.backbones = nn.ModuleList(backbones)
            #self.input_proj_robot_state = nn.Linear(8, hidden_dim)
            self.input_proj_robot_state = nn.Linear(8, hidden_dim)
            if velocity_control:
                #self.input_proj_robot_state_vel = nn.Linear(8, hidden_dim)  # project concatenated [qpos, qvel]
                self.input_proj_robot_state_vel = nn.Linear(state_dim, hidden_dim)  # project concatenated [qpos, qvel]
        else:
            # input_dim = 7 + 7 # robot_state + env_state
            #self.input_proj_robot_state = nn.Linear(8, hidden_dim)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(8, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None
        
        # encoder extra parameters
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(8, hidden_dim) # project action to embedding
        #self.encoder_joint_proj = nn.Linear(8, hidden_dim)  # project qpos to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
        if velocity_control:
            self.encoder_joint_vel_proj = nn.Linear(state_dim, hidden_dim) # project action to embedding

        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        # Position table size: [CLS] + qpos + qvel (if velocity_control) + a_seq
        pos_table_len = 1 + 1 + (1 if velocity_control else 0) + num_queries
        self.register_buffer('pos_table',get_sinusoid_encoding_table(pos_table_len, hidden_dim))

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding for decoder input
        # learned position embedding for latent, proprio (and optionally velocity)
        additional_pos_size = 3*self.context_length if velocity_control else 2*self.context_length
        self.additional_pos_embed = nn.Embedding(additional_pos_size, hidden_dim)
        self.temporal_embed = nn.Embedding(self.context_length, hidden_dim)       
    
    """
    TRANSFORMER ENCODER (for VAE latent space) LOGIC:
    - Input to encoder: [CLS] token, qpos, (optionally qvel), action sequence
    - Obtain latent z from encoder output at CLS token
    """
    def forward(self, qpos, image, env_state, qvel=None, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs = qpos.shape[0]


        
        ### --------------------------------------------------------------
        ### ------------------------ ENCODER STEP ------------------------
        ### --------------------------------------------------------------
        ### Obtain latent z from action sequence (and optionally qpos/qvel)

        if is_training:            # project action sequence to embedding dim, and concat with a CLS token
            
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_short = qpos[:,-1,:]
            qvel_short = qvel[:,-1,:] if qvel is not None else None
            qpos_embed = self.encoder_joint_proj(qpos_short)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
           
            if qvel is not None:
                qvel_embed = self.encoder_joint_vel_proj(qvel_short)  # (bs, hidden_dim)
                qvel_embed = torch.unsqueeze(qvel_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            if qvel is not None:
                encoder_input = torch.cat([cls_embed, qpos_embed, qvel_embed, action_embed], axis=1) # (bs, seq+3, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+3, bs, hidden_dim)
                # do not mask cls token, qpos, and qvel
                cls_joint_is_pad = torch.full((bs, 3), False).to(qpos.device) # False: not a padding
            else:
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+2, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+2, bs, hidden_dim)
                # do not mask cls token and qpos
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+2 or seq+3)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
        
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        ### --------------------------------------------------------------
        ### ------------------------ DECODER STEP ------------------------
        ### --------------------------------------------------------------
        ### Obtain action predictions from latent z and image features (and optionally qpos/qvel)
        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                for i in range(self.context_length):
                    features, pos = self.backbones[cam_id](image[:, cam_id,i]) # HARDCODED
                    features = features[0] # take the last layer feature
                    pos = pos[0]
                    pos += self.temporal_embed.weight[i].unsqueeze(1).unsqueeze(2) # add temporal embedding
                    all_cam_features.append(self.input_projections[cam_id](features).unsqueeze(1)) 
                    all_cam_pos.append(pos.unsqueeze(1))
            # proprioception features
            if qvel is not None:
                # Concatenate qpos and qvel for richer proprioceptive input
                proprio_input_vel = self.input_proj_robot_state_vel(qvel)
            else:
                proprio_input_vel = None
            
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3) # concat on width dimension
            pos = torch.cat(all_cam_pos, axis=3)
            
            # Prepare decoder conditioning: optionally exclude qpos/qvel
            if self.decoder_no_state:
                # Only use latent z, not qpos/qvel
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, None, self.additional_pos_embed.weight, velocity_input=None, temporal_embed=self.temporal_embed.weight)[0]
            else:
                # Include qpos and optionally qvel in decoder conditioning
                proprio_input_pos = self.input_proj_robot_state(qpos)
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input_pos, self.additional_pos_embed.weight, velocity_input = proprio_input_vel, temporal_embed=self.temporal_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) 
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        # DECODER OUTPUTS
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 8
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=8, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 8
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):

    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)
    # Get conditioning_enabled flag if it exists, defaults to True
    conditioning_enabled = getattr(args, 'conditioning_enabled', True)

    # Get decoder_no_state flag if it exists, defaults to False
    decoder_no_state = getattr(args, 'decoder_no_state', False)
    
    # Choose transformer based on decoder_no_state flag
    if decoder_no_state:
        transformer = build_transformer_no_conditions(args)
    else:
        transformer = build_transformer(args)

    encoder = build_encoder(args)
    
    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=args.state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        velocity_control = args.velocity_control, 
        context_length = args.context_length,
        conditioning_enabled = conditioning_enabled,
        latent_dim = args.latent_dim,
        decoder_no_state = decoder_no_state,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 8 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

