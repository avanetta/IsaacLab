import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

#from act_copy.detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import torch
import IPython
import numpy as np
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.beta_warmup_epochs = args_override.get('beta_warmup_epochs', 0)  # 0 = no warmup (use full KL from start)


    def __call__(self, qpos, image, actions=None, is_pad=None, qvel=None, epoch = 0):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, qvel=qvel, actions=actions, is_pad=is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            #total_kld, dim_wise_kld, mean_kld = kl_divergence_with_free_bits(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            
            # Beta-warmup: linearly increase KL weight from 0 to kl_weight over warmup epochs
            if self.beta_warmup_epochs > 0:
                beta = min(1.0, epoch / self.beta_warmup_epochs)  # Linear warmup: 0 -> 1
            else:
                beta = 1.0  # No warmup, use full KL weight immediately
            
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight * beta
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state, qvel) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def kl_divergence_with_free_bits(mu, logvar, free_bits=0.01):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    klds = torch.clamp(klds, min=free_bits) # apply free bits threshold
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
