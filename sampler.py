import torch as th
from torch import nn
import numpy as np
import warnings
from time import time as get_time

from generator import *
from myFunctionals import complex_mul
from util import *

class mc_sampler:
    def __init__(self, net_ref, net_sampler):
        self.net_ref = net_ref
        self.net_sampler = net_sampler
        self.sample = None
        self.weight = None
        self.counter = -1
    def get_samples(self, nbatch, gamma=0.5,prob_weights = None):
        '''
        sample (2*nbatch) samples for recycling. Take out [:nbatch], [0.5nbatch,1.5nbatch],[nbatch:],[1.5batch:0.5nbatch]
        for 4 gradient descent
        Returns:
            sampled spin configuration of shape=(nbatch,nspin)
            associated weight of shape (nbatch)
        '''
        recycle=4
        extend_size = 2
        self.counter = (self.counter + 1) % recycle
        if self.counter % recycle == 0:
            w = self.net_sampler.subnet_l1importance() if prob_weights is None else prob_weights.clone()
            self.sample, self.weight = generate_fid_config(nbatch*extend_size, self.net_sampler, self.net_ref, gamma, w)
            #self.sample, self.weight = generate_ref_config(nbatch*extend_size, self.net_ref, gamma=0.1)
        half_batch = nbatch // 2
        lhalf_idx = th.arange(half_batch) + self.counter * half_batch
        rhalf_idx = th.arange(half_batch) + (self.counter+1)%recycle * half_batch
        output_sample = self.sample[th.cat([lhalf_idx,rhalf_idx])]
        output_weight = self.weight[th.cat([lhalf_idx,rhalf_idx])]
        return output_sample, output_weight         


def HamNet(ham, net_ensemble, source_basis, evaluate_source=False):## mixed nets
    '''
    Args:
        net_ensemble(module): the neural network used to evaluate
        source_basis(th.long): shape = (source_batch, spin)
        evaluate_source: If True, output will also include net_ensemble(source_basis)
    Returns:
        Hsita_Re, Hsita_Im(th.float): shape = (groups, source_batch, max_nbsize)
        (optional)sita_Re, sita_Im = (groups, source_batch)
    '''
    groups = net_ensemble.groups
    source_batchsize, nspin = source_basis.shape
    assert nspin == ham.nspin, "illegal lattice size"
    ## prepare the neighbor list and coupling strength
    ## basis_nbhood: (source_batch, max_nbsize, nspin), weight:(source_bath, max_nbsize)
    basis_nbhood, weight_nbhood = ham.get_neighbor_list_parallel(source_basis)
    max_nbsize = basis_nbhood.shape[-2]
    weight_nbhood = weight_nbhood.expand(groups, source_batchsize, max_nbsize)
    weight_nbhood_Re = weight_nbhood
    weight_nbhood_Im = th.zeros_like(weight_nbhood_Re)
    ## calculate Hnet(source_basis)
    if net_ensemble.groups == 1: ## reference net
        sita_nbhood_Re, sita_nbhood_Im = net_ensemble(basis_nbhood.reshape(-1, nspin))
    else:
        sita_nbhood_Re, sita_nbhood_Im = net_ensemble.unique_forward(basis_nbhood.reshape(-1, nspin))
    sita_nbhood_Re = sita_nbhood_Re.reshape(groups, source_batchsize, max_nbsize)
    sita_nbhood_Im = sita_nbhood_Im.reshape(sita_nbhood_Re.shape)
    Hsita_Re = (sita_nbhood_Re * weight_nbhood_Re).sum(-1) 
    Hsita_Im = (sita_nbhood_Im * weight_nbhood_Re).sum(-1) 
    if evaluate_source:
        return Hsita_Re, Hsita_Im, sita_nbhood_Re[:,:,0], sita_nbhood_Im[:,:,0]
    else:
        return Hsita_Re, Hsita_Im

def sample_weighted_variance_unified_importance(batchsize, ham, mc_sampler, net_ref, net_ensemble): 
    '''
    sample weighted variance divergence = \sum_i <psi_i|(H_E_i)^2|psi_i> 
    and expH_i = <psi_i|H|psi_i>/<psi_i|psi_i> for all i
    where psi_i = c_i*sita_i +\frac{c_i^2}{\sum_j c_j^2}  * (ref - \sum_j c_j*sita_j)
    WEIGHTS for all term: |ref_net|^2 + \sum_i c_i^2|net_ensemble_i|^2
    '''
    start_time = get_time()
    dim_group, dim_batch = 0, 1
    E_target = net_ensemble.target_bands.unsqueeze(-1)
    start_time = get_time()
    source_basis, source_weight = mc_sampler.get_samples(batchsize)
    print('sampling time:',get_time()-start_time)
    ## evaluate [Hsita_r,Hsita_i,sita_r,sita_i] at given configurations
    sita_evaluated = HamNet(ham, net_ensemble, source_basis, evaluate_source=True) ## all shapes=(ngroup, batchsize)
    ## evaluate [Href_r, Href_i, ref_r, ref_i] at given configurations
    ref_evaluated = HamNet(ham, net_ref, source_basis, evaluate_source=True) ## all shapes=(1, batchsize)
    ## evaluate [Hpsi_r, Hpsi_i, psi_r, psi_i] at given configurations
    Hpsi_r, Hpsi_i, psi_r, psi_i = net_ensemble.faithful_transform(
        sita_evaluated, ref_evaluated) ## all shapes=(nbands, batchsize)
    ## compute <psi_i|(H_E_i)^2|psi_i>
    HmEpsi_norm2 = (Hpsi_r - psi_r * E_target)**2 + (Hpsi_i - psi_i * E_target)**2
    
    ## compute <psi_i|H|psi_i>, no gradiant needed
    with th.no_grad():
        psiHpsi = ((psi_r * Hpsi_r + psi_i * Hpsi_i)  / source_weight.unsqueeze(0)).mean(-1)
        psipsi = ((psi_r * psi_r + psi_i * psi_i)  / source_weight.unsqueeze(0)).mean(-1)
        expH = psiHpsi / psipsi
        varH = (HmEpsi_norm2 / source_weight.unsqueeze(0)).mean(-1) / psipsi
    weighted_variance = (HmEpsi_norm2.sum(0) / source_weight).mean()
    return weighted_variance, expH, varH, psipsi

def sample_weighted_variance_unified_importance_exact(ham, net_ref, net_ensemble, backbone=False): 
    '''
    sample weighted variance divergence = \sum_i <psi_i|(H_E_i)^2|psi_i> 
    and expH_i = <psi_i|H|psi_i>/<psi_i|psi_i> for all i
    where psi_i = c_i*sita_i +\frac{c_i^2}{\sum_j c_j^2}  * (ref - \sum_j c_j*sita_j)
    WEIGHTS for all term: |ref_net|^2 + \sum_i c_i^2|net_ensemble_i|^2
    '''
    dim_group, dim_batch = 0, 1
    # shape(source_basis)=(batchsize,nspin);
    if backbone:
        if hasattr(net_ref,'reference_state'):
            source_basis = ham.get_backbone(net_ref.reference_state)
        else:
            raise NotImplementedError
    else:
        source_basis = ham.get_full_basis()
    ## evaluate [Hsita_r,Hsita_i,sita_r,sita_i] at given configurations
    sita_evaluated = HamNet(ham, net_ensemble, 
        source_basis, evaluate_source=True) ## all shapes=(ngroup, batchsize)
    ## evaluate [Href_r, Href_i, ref_r, ref_i] at given configurations
    ref_evaluated = HamNet(ham, net_ref, 
        source_basis, evaluate_source=True) ## all shapes=(1, batchsize) 
    ## evaluate [Hpso_r, Hpsi_i, psi_r, psi_i] at given configurations
    Hpsi_r, Hpsi_i, psi_r, psi_i = net_ensemble.faithful_transform(
        sita_evaluated, ref_evaluated) ## all shapes=(nbands, batchsize)
    ## compute <psi_i|(H_E_i)^2|psi_i>
    E_target = net_ensemble.target_bands.unsqueeze(-1)
    HmEpsi_norm2 = (Hpsi_r - psi_r * E_target)**2 + (Hpsi_i - psi_i * E_target)**2
    weighted_variance = HmEpsi_norm2.sum() 
    ## compute <psi_i|H|psi_i>, no gradiant needed
    with th.no_grad():
        psiHpsi = (psi_r * Hpsi_r + psi_i * Hpsi_i).sum(-1)
        psipsi = (psi_r * psi_r + psi_i * psi_i).sum(-1)
        expH = psiHpsi / psipsi
        varH = HmEpsi_norm2.sum(-1) / psipsi
    return weighted_variance, expH, varH, psipsi
