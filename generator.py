from time import time as get_time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import logging
from myFunctionals import *
from util import *
import warnings

# from pytorch_memlab import LineProfiler,profile_every,profile
def generate_fid_config(nbatch, net_ensemble, net_ref,gamma=0.5, prob_weights=None):  ## mixed nets
    """GENERATE configuration with respect to 
            |net_ref|^2 + \sum_i c_i^2|net_ensemble_i|^2
        or  |net_ref|^2 + \sum_i c_i^2/(\sum_j c_j^2)|net_ensemble_i|^2.
    Args:
        nbatch: size of the batch going to be generated 
        gamma(float):  soften the probability distribution
    Returns:
        samples(Tensor): Generated Configuration (nbatch, in_dim).
        samples_weight(Tensor): amplitude^2 of the generated configuration 
                                wrt the coupled adjacent groups of shape (nbatch) 
    """
    dim_group, dim_batch, dim_spin = 0, 1, 2
    sanity_check = True
    cellsize= net_ensemble.cellsize
    if prob_weights is None:
        prob_weights = net_ensemble.subnet_l1importance()
    ref_prob_weights = prob_weights[0]
    net_prob_weights = prob_weights[1:,None]
    # ref_prob_weights = net_prob_weights.mean()
    samples = th.zeros((nbatch, net_ensemble.nspin),device=net_prob_weights.device, dtype = th.long)
    with th.no_grad():
        ## sample N cells through N forward propagation.
        for current_idx in range(net_ensemble.ncell):
            #print('current_idx:',current_idx)
            #print('sampled cell:',samples[0,:current_idx * cellsize])
            preped_config = net_ensemble._preprocess(samples) #(1,nbatch, cellsize, ncell)
            start_idx = current_idx * cellsize
            ########## reference state ##########
            # start_time = get_time()
            if hasattr(net_ref, 'is_wrapped'): ## no need to include the reference net of this net_ref?
                wrapped_net = net_ref.net_ensemble
                wrapped_net_prob_weights = wrapped_net.subnet_l1importance().unsqueeze(-1)
                ref_amp_output = wrapped_net.amp_net(preped_config)
                ref_amp_adjusted = th.abs(ref_amp_output)**gamma
                ref_amp_normed = wrapped_net._postprocess(preped_config, ref_amp_adjusted)
                ref_amp2_normed = ref_amp_normed**2 ## (groups, nbatch, ncell, 2^cellsize)
                ref_amp2_current = ref_amp2_normed[:,:, current_idx] ## (groups,nbatch,2^cellsize)
                
                if current_idx > 0:
                    ref_amp2_prev = wrapped_net._select(
                        samples[:,:start_idx], ref_amp2_normed[:,:, :current_idx]
                        )
                    ref_logamp2_prev = th.log(th.clamp(ref_amp2_prev,min=1e-8)).sum(-1) ## (groups,nbatch)
                    ref_logamp2_prev = ref_logamp2_prev + th.log(wrapped_net_prob_weights[1:])
                    ref_amp2_prev = th.exp(ref_logamp2_prev) ## (groups,nbatch)
                    ### some amp2_prev is 0
                else:
                    ref_amp2_prev = wrapped_net_prob_weights[1:]
            else:
                ref_amp_output = net_ref.amp_net(preped_config)  # (1,nbatch, 2^cellsize, ncell)
                ref_amp_normed = net_ref._postprocess(preped_config, ref_amp_output, noise=False) # (1,nbatch, ncell, 2^cellsize)
                ref_amp2_normed = ref_amp_normed**2
                ref_amp2_current = ref_amp2_normed[:,:,current_idx,:] ## (1,nbatch, 2^cellsize)
                if current_idx > 0:
                    ref_amp2_prev = ref_prob_weights * net_ref._select(
                            samples[:,:start_idx], ref_amp2_normed[:,:,:current_idx]
                            ).prod(-1)  ## (1,nbatch)
                else:
                    ref_amp2_prev = ref_prob_weights * th.ones_like(ref_amp2_normed[0,:,0,0]).unsqueeze(0)
                #print('ref_amp2_prev',ref_amp2_prev[0,0])
                assert len(ref_amp2_prev.shape) == 2
                
            # print('reference sampling time:',get_time()-start_time)
            ########## predicted state ###########
            amp_output = net_ensemble.amp_net(preped_config)
            amp_adjusted = th.abs(amp_output)**gamma
            amp_normed = net_ensemble._postprocess(preped_config, amp_adjusted)
            amp2_normed = amp_normed**2 ## (groups, nbatch, ncell, 2^cellsize)
            amp2_current = amp2_normed[:,:, current_idx] ## (groups,nbatch,2^cellsize)
            if current_idx > 0:
                amp2_prev = net_ensemble._select(
                    samples[:,:start_idx], amp2_normed[:,:, :current_idx]
                    ) ## (groups, nbatch, current_idx)
                logamp2_prev = th.log(th.clamp(amp2_prev,min=1e-8)).sum(-1) ## (groups,nbatch)
                logamp2_prev = logamp2_prev + th.log(net_prob_weights)
                amp2_prev = th.exp(logamp2_prev) ## (groups,nbatch)
                ### some amp2_prev is 0
            else:
                amp2_prev = net_prob_weights.repeat(1,nbatch)
            assert len(amp2_prev.shape) == 2
            ######## combine reference state and predicted state
            amp2_combined = (
                amp2_current * amp2_prev.unsqueeze(-1)
                ).sum(0) + (
                ref_amp2_current * ref_amp2_prev.unsqueeze(-1)
                ).sum(0)    ## (nbatch,2^cellsize)
            amp2_combined_base = (amp2_prev.sum(0) + ref_amp2_prev.sum(0)).unsqueeze(-1) ## (nbatch,1)
#             print('amp2_current',amp2_current)
#             print('amp2_prev',amp2_prev)
#             print('ref_amp2_current',ref_amp2_current)
#             print('ref_amp2_prev',ref_amp2_prev)

            prob_current = amp2_combined / amp2_combined_base
            # start_time = get_time()
            if sanity_check:
                if amp2_combined_base.min()<=0:
                    anomaly = amp2_combined_base<=0
                    logging.info("%i anomaly detected in amp2_combined_base"%(anomaly.to(dtype=th.long).sum()))
                    logging.info('anomaly:{}'.format(amp2_combined_base[anomaly]) )
                    print("{} anomaly detected in amp2_combined_base".format(anomaly.to(dtype=th.long).sum()))
                    print('anomaly:{}'.format(amp2_combined_base[anomaly]) )
                    raise ValueError
                if amp2_prev.sum(0).min() <= 0:
                    warnings.warn("amp2_prev=",amp_prev.sum(0))
                #if ref_amp2_prev.min() <= 0:
                #    warnings.warn("ref_amp2_prev=",ref_amp2_prev)
                # logging.info("now_idx=%i,amp2_prev=%s"%(current_idx,amp2_prev))
                # logging.info("now_idx=%i,amp2_combined_base.min %f"%(current_idx, amp2_combined_base.min()))
                # logging.info("now_idx=%i,amp2_combined_base=%s"%(current_idx,amp2_combined_base.flatten()))
                # logging.info("now_idx=%i,prob_current_sum=%s"%(current_idx,prob_sum))
                prob_sum = prob_current.sum(-1)
                if prob_sum.min()<=0:
                    anomaly = (prob_sum<=0)
                    logging.info("%i anomaly detected in prob_current"%(anomaly.to(dtype=th.long).sum()))
                    logging.info('anomaly:{}'.format(prob_current[anomaly]) )
                    print("{} anomaly detected in prob_current".format(anomaly.to(dtype=th.long).sum()))
                    print('anomaly:{}'.format(prob_current[anomaly]) )
                    ## continue the error report tomorrow
            selected_idx = th.multinomial(prob_current,1).reshape(nbatch)
            selected_config = 1 - 2 * integer2bit(selected_idx, cellsize) # (nabtch, cellsize)
            samples[:,start_idx:start_idx + cellsize] = selected_config.type_as(samples)

        amp2_final = net_ensemble._select(samples, amp2_normed)  ## (groups, nbatch, ncell)
        amp2_final = ( amp2_final.prod(-1) 
                        * net_prob_weights).sum(dim_group)
        if hasattr(net_ref, 'is_wrapped'):
            ref_amp2_final = wrapped_net._select(samples, ref_amp2_normed)  ## (groups, nbatch, ncell)
            ref_amp2_final = ( ref_amp2_final.prod(-1) 
                            * wrapped_net_prob_weights[1:]).sum(dim_group)
            samples_weight = (amp2_final + ref_amp2_final) / (net_prob_weights.sum() + wrapped_net_prob_weights[1:].sum())
        else:
            ref_amp2_final =  net_ensemble._select(samples, ref_amp2_normed)[0]
            ref_amp2_final = ref_amp2_final.prod(-1) * ref_prob_weights
            ## normalize samples_weight
            samples_weight = (amp2_final + ref_amp2_final) / prob_weights.sum()
    return net_ensemble.recover(samples), samples_weight

def generate_ref_config(nbatch, net_ref, gamma=0.1):   
    """GENERATE configuration with respect to 
            |net_ref|^2  
    Args:
        nbatch: size of the batch going to be generated 
        gamma(float):  soften the probability distribution
    Returns:
        samples(Tensor): Generated Configuration (nbatch, in_dim).
        samples_weight(Tensor): amplitude^2 of the generated configuration 
                                wrt the coupled adjacent groups of shape (nbatch) 
    """
    dim_group, dim_batch, dim_spin = 0, 1, 2
    sanity_check = True
    cellsize= net_ref.cellsize
    samples = th.zeros((nbatch, net_ref.nspin),device=net_ref.expH.device, dtype = th.long)
    net_ref.reinit(epsilon=gamma)
    with th.no_grad():
        ## sample N cells through N forward propagation.
        for current_idx in range(net_ref.ncell):
            #print('current_idx:',current_idx)
            #print('sampled cell:',samples[0,:current_idx * cellsize])
            preped_config = net_ref._preprocess(samples) #(1,nbatch, cellsize, ncell)
            start_idx = current_idx * cellsize
            ########## reference state ##########
            # start_time = get_time()
            if hasattr(net_ref, 'is_wrapped'): ## no need to include the reference net of this net_ref?
                wrapped_net = net_ref.net_ensemble
                wrapped_net_prob_weights = wrapped_net.subnet_l1importance().unsqueeze(-1)
                ref_amp_output = wrapped_net.amp_net(preped_config)
                ref_amp_adjusted = th.abs(ref_amp_output)**gamma
                ref_amp_normed = wrapped_net._postprocess(preped_config, ref_amp_adjusted)
                ref_amp2_normed = ref_amp_normed**2 ## (groups, nbatch, ncell, 2^cellsize)
                ref_amp2_current = ref_amp2_normed[:,:, current_idx] ## (groups,nbatch,2^cellsize)
                
                if current_idx > 0:
                    ref_amp2_prev = wrapped_net._select(
                        samples[:,:start_idx], ref_amp2_normed[:,:, :current_idx]
                        )
                    ref_logamp2_prev = th.log(th.clamp(ref_amp2_prev,min=1e-8)).sum(-1) ## (groups,nbatch)
                    ref_logamp2_prev = ref_logamp2_prev + th.log(wrapped_net_prob_weights[1:])
                    ref_amp2_prev = th.exp(ref_logamp2_prev) ## (groups,nbatch)
                    ### some amp2_prev is 0
                else:
                    ref_amp2_prev = wrapped_net_prob_weights[1:]
            else:
                ref_amp_output = net_ref.amp_net(preped_config)  # (1,nbatch, 2^cellsize, ncell)
                ref_amp_normed = net_ref._postprocess(preped_config, ref_amp_output, noise=False) # (1,nbatch, ncell, 2^cellsize)
                ref_amp2_normed = ref_amp_normed**2
                ref_amp2_current = ref_amp2_normed[:,:,current_idx,:] ## (1,nbatch, 2^cellsize)
                if current_idx > 0:
                    ref_amp2_prev = net_ref._select(
                            samples[:,:start_idx], ref_amp2_normed[:,:,:current_idx]
                            ).prod(-1)  ## (1,nbatch)
                else:
                    ref_amp2_prev = th.ones_like(ref_amp2_normed[0,:,0,0]).unsqueeze(0)
                #print('ref_amp2_prev',ref_amp2_prev[0,0])
                assert len(ref_amp2_prev.shape) == 2
                
            amp2_combined = (
                ref_amp2_current * ref_amp2_prev.unsqueeze(-1)
                ).sum(0)    ## (nbatch,2^cellsize)
            amp2_combined_base = ref_amp2_prev.sum(0).unsqueeze(-1) ## (nbatch,1)
#             print('amp2_current',amp2_current)
#             print('amp2_prev',amp2_prev)
#             print('ref_amp2_current',ref_amp2_current)
#             print('ref_amp2_prev',ref_amp2_prev)

            prob_current = amp2_combined / amp2_combined_base
            if sanity_check:
                if amp2_combined_base.min()<=0:
                    anomaly = amp2_combined_base<=0
                    logging.info("%i anomaly detected in amp2_combined_base"%(anomaly.to(dtype=th.long).sum()))
                    logging.info('anomaly:{}'.format(amp2_combined_base[anomaly]) )
                    print("{} anomaly detected in amp2_combined_base".format(anomaly.to(dtype=th.long).sum()))
                    print('anomaly:{}'.format(amp2_combined_base[anomaly]) )
                    raise ValueError
                #if ref_amp2_prev.min() <= 0:
                #    warnings.warn("ref_amp2_prev=",ref_amp2_prev)
                # logging.info("now_idx=%i,amp2_prev=%s"%(current_idx,amp2_prev))
                # logging.info("now_idx=%i,amp2_combined_base.min %f"%(current_idx, amp2_combined_base.min()))
                # logging.info("now_idx=%i,amp2_combined_base=%s"%(current_idx,amp2_combined_base.flatten()))
                # logging.info("now_idx=%i,prob_current_sum=%s"%(current_idx,prob_sum))
                prob_sum = prob_current.sum(-1)
                if prob_sum.min()<=0:
                    anomaly = (prob_sum<=0)
                    logging.info("%i anomaly detected in prob_current"%(anomaly.to(dtype=th.long).sum()))
                    logging.info('anomaly:{}'.format(prob_current[anomaly]) )
                    print("{} anomaly detected in prob_current".format(anomaly.to(dtype=th.long).sum()))
                    print('anomaly:{}'.format(prob_current[anomaly]) )
                    ## continue the error report tomorrow
            selected_idx = th.multinomial(prob_current,1).reshape(nbatch)
            selected_config = 1 - 2 * integer2bit(selected_idx, cellsize) # (nabtch, cellsize)
            samples[:,start_idx:start_idx + cellsize] = selected_config.type_as(samples)

        if hasattr(net_ref, 'is_wrapped'):
            ref_amp2_final = wrapped_net._select(samples, ref_amp2_normed)  ## (groups, nbatch, ncell)
            ref_amp2_final = ( ref_amp2_final.prod(-1) 
                            * wrapped_net_prob_weights[1:]).sum(dim_group)
            samples_weight = (amp2_final + ref_amp2_final) / (net_prob_weights.sum() + wrapped_net_prob_weights[1:].sum())
        else:
            ref_amp2_final =  net_ref._select(samples, ref_amp2_normed)[0]
            ref_amp2_final = ref_amp2_final.prod(-1)
            ## normalize samples_weight
            samples_weight = ref_amp2_final
    net_ref.reinit(epsilon=0.0)
    return net_ref.recover(samples), samples_weight
def evaluate_generative_prob(samples, net_ensemble, net_ref,gamma=0.5, prob_weights=None):
    dim_group, dim_batch, dim_spin = 0, 1, 2
    if prob_weights is None:
        prob_weights = net_ensemble.subnet_l1importance()
    ref_prob_weights = prob_weights[0]
    net_prob_weights = prob_weights[1:,None]
    preped_config = net_ensemble._preprocess(samples) #(1,nbatch, cellsize, ncell)
    ## reference part
    ref_amp_output = net_ref.amp_net(preped_config)  # (1,nbatch, 2^cellsize, ncell)
    ref_amp_normed = net_ref._postprocess(preped_config, ref_amp_output, noise=False) # (1,nbatch, ncell, 2^cellsize)
    ref_amp2_normed = ref_amp_normed**2
    ## spectral net part
    amp_output = net_ensemble.amp_net(preped_config)
    amp_adjusted = th.abs(amp_output)**gamma
    amp_normed = net_ensemble._postprocess(preped_config, amp_adjusted)
    amp2_normed = amp_normed**2 ## (groups, nbatch, ncell, 2^cellsize)
    ## assemble
    amp2_final = net_ensemble._select(samples, amp2_normed)  ## (groups, nbatch, ncell)
    amp2_final = ( amp2_final.prod(-1) * net_prob_weights).sum(dim_group)
    ref_amp2_final = (net_ensemble._select(
                samples, ref_amp2_normed
                )[0]).prod(-1) * ref_prob_weights
    ## normalize samples_weight
    samples_weight = (amp2_final + ref_amp2_final) / prob_weights.sum()
    return samples_weight
def archived_generate_fid_config(nbatch, net_ensemble, net_ref):  ## separated nets
    """GENERATE configuration with respect to 
            |net_ref|^2 + \sum_i c_i^2|net_ensemble_i|^2
        or  |net_ref|^2 + \sum_i c_i^2/(\sum_j c_j^2)|net_ensemble_i|^2.
    Args:
        nbatch: size of the batch going to be generated 
    Returns:
        samples(Tensor): Generated Configuration (nbatch, in_dim).
        samples_weight(Tensor): amplitude^2 of the generated configuration 
                                wrt the coupled adjacent groups of shape (nbatch) 
    """
    dim_group, dim_batch, dim_spin = 0, 1, 2
    sanity_check = True
    device = net_ensemble.expH.device
    cellsize= net_ensemble.cellsize
    samples = th.zeros((nbatch, net_ensemble.nspin),device=device, dtype = th.long)
    with th.no_grad():
        ## sample N cells through N forward propagation.
        for current_idx in range(net_ensemble.ncell):
            start_idx = current_idx * cellsize
            # ## weights the network contribution by |net_ref|^2 + \sum_i c_i^2|net_ensemble_i|^2.
            # net_prob_weights = net_ensemble.groups_amp2[:,None]
            ## weights the network contribution by |net_ref|^2 + \sum_i c_i^2/(\sum_j c_j^2)|net_ensemble_i|^2.
            net_prob_weights = net_ensemble.groups_amp2_ratio[:,None]
            ########## reference state ##########
            if hasattr(net_ref, 'is_wrapped'):
                wrapped_net = net_ref.net_ensemble
                inputs = samples.unsqueeze(0).repeat(wrapped_net.groups,1,1)
                ref_amp_output = wrapped_net.amp_net(wrapped_net._preprocess(inputs))
                ref_amp_normed, ref_alive_mask = wrapped_net._postprocess(
                    inputs, ref_amp_output, generative = True)
                ref_amp2_normed = ref_amp_normed**2 ## (groups, nbatch, ncell, 2^cellsize)
                wrapped_amp2 = net_ref.groups_amp2.to(dtype=ref_amp2_normed.dtype,device=device)
                if current_idx > 0:
                    ref_amp2_prev = wrapped_net._select(
                        inputs[:,:,:start_idx], ref_amp2_normed[:,:, :current_idx]
                        )
                    ref_logamp2_prev = th.log(th.clamp(ref_amp2_prev,min=1e-8)).sum(-1) ## (groups,nbatch)
                    ref_logamp2_prev = ref_logamp2_prev + th.log(wrapped_amp2[:,None])
                    ref_amp2_prev = th.exp(ref_logamp2_prev) ## (groups,nbatch)
                    ### some amp2_prev is 0
                else:
                    ref_amp2_prev = wrapped_amp2[:,None].repeat(1,nbatch)
                ref_amp2_current = ref_amp2_normed[:,:, current_idx] ## (groups,nbatch,2^cellsize)
            else:
                inputs = samples.unsqueeze(0)  ## (1, nbatch, nspin)
                ref_amp_output = net_ref.amp_net(inputs, cellsize)  # (1,nbatch, 2^cellsize, ncell)
                ref_amp_normed, ref_alive_mask = net_ensemble._postprocess(
                    inputs, ref_amp_output, generative = True) # (1,nbatch, ncell, 2^cellsize)
                ref_amp2_normed = ref_amp_normed**2
                if current_idx > 0:
                    ref_amp2_prev = (net_ensemble._select(
                            inputs[:,:,:start_idx], ref_amp2_normed[:,:,:current_idx]
                            )[0]).prod(-1).unsqueeze(0)  ## (1,nbatch)
                    assert len(ref_amp2_prev.shape) == 2
                else:
                    ref_amp2_prev = th.ones_like(ref_amp2_normed[0,:,0,0]).unsqueeze(0)
                ref_amp2_current = ref_amp2_normed[0,:,current_idx,:].unsqueeze(0) ## (1,nbatch, 2^cellsize)
            ########## predicted state ###########
            inputs = samples.unsqueeze(0).repeat(net_ensemble.groups,1,1)
            amp_output = net_ensemble.amp_net(net_ensemble._preprocess(inputs))
            amp_normed, alive_mask = net_ensemble._postprocess(
                inputs, amp_output, generative = True)
            amp2_normed = amp_normed**2 ## (groups, nbatch, ncell, 2^cellsize)
            if current_idx > 0:
                amp2_prev = net_ensemble._select(
                    inputs[:,:,:start_idx], amp2_normed[:,:, :current_idx]
                    )
                logamp2_prev = th.log(th.clamp(amp2_prev,min=1e-8)).sum(-1) ## (groups,nbatch)
                logamp2_prev = logamp2_prev + th.log(net_prob_weights)
                amp2_prev = th.exp(logamp2_prev) ## (groups,nbatch)
                ### some amp2_prev is 0
            else:
                amp2_prev = net_prob_weights.repeat(1,nbatch)
            amp2_current = amp2_normed[:,:, current_idx] ## (groups,nbatch,2^cellsize)
            ######## combine reference state and predicted state
            amp2_combined = (
                amp2_current * amp2_prev.unsqueeze(-1)
                ).sum(0) + (
                ref_amp2_current * ref_amp2_prev.unsqueeze(-1)
                ).sum(0)    ## (nbatch,2^cellsize)
            amp2_combined_base = (amp2_prev.sum(0) + ref_amp2_prev.sum(0)).unsqueeze(-1) ## (nbatch,1)
            prob_current = amp2_combined / amp2_combined_base
            if sanity_check:
                if amp2_prev.sum(0).min() <= 0:
                    warnings.warn("amp2_prev=",amp_prev.sum(0))
                #if ref_amp2_prev.min() <= 0:
                #    warnings.warn("ref_amp2_prev=",ref_amp2_prev)
                # logging.info("now_idx=%i,amp2_prev=%s"%(current_idx,amp2_prev))
                # logging.info("now_idx=%i,amp2_combined_base.min %f"%(current_idx, amp2_combined_base.min()))
                # logging.info("now_idx=%i,amp2_combined_base=%s"%(current_idx,amp2_combined_base.flatten()))
                # logging.info("now_idx=%i,prob_current_sum=%s"%(current_idx,prob_sum))
                prob_sum = prob_current.sum(-1)
                for ii,pp in enumerate(prob_sum): 
                    if pp == 0:
                        logging.info("anomaly in prob_current=(%i,%s)"%(ii,prob_current[ii]))
                        logging.info("anomaly in amp2_current=(%i,%s)"%(ii,amp2_current[:,ii]))
            selected_idx = th.multinomial(prob_current,1).reshape(nbatch)
            selected_config = 1 - 2 * integer2bit(selected_idx, cellsize) # (nabtch, cellsize)
            samples[:,start_idx:start_idx + cellsize] = selected_config.type_as(samples)

        amp2_final = net_ensemble._select(
            samples.unsqueeze(0).repeat(net_ensemble.groups,1,1), amp2_normed
            )  ## (groups, nbatch, ncell)
        amp2_final = ( amp2_final.prod(dim_spin) 
                        * net_prob_weights).sum(dim_group)
        if hasattr(net_ref, 'is_wrapped'):
            ref_amp2_final = wrapped_net._select(
                samples.unsqueeze(0).repeat(wrapped_net.groups,1,1), ref_amp2_normed
                )  ## (groups, nbatch, ncell)
            ref_amp2_final = ( ref_amp2_final.prod(dim_spin) 
                            * wrapped_amp2[:,None]).sum(dim_group)
        else:
            ref_amp2_final = (net_ensemble._select(
                        samples.unsqueeze(0), ref_amp2_normed
                        )[0]).prod(-1)
        ## normalize samples_weight
        samples_weight = (amp2_final + ref_amp2_final) / (1.0+net_prob_weights.sum())
        return net_ensemble.recover(samples), samples_weight

def archived_generate_var_config(nbatch, net_ensemble, tol=0.05):
    """GENERATE configuration with respect to each group according to wave amplitude |net_ensemble_i|^2
    Args:
        nbatch: size of the batch going to be generated 
    Returns:
        samples(Tensor): Generated Configuration with shape (groups, nbatch, in_dim).
        samples_weight(Tensor): amplitude^2 of the generated configuration with shape (groups, nbatch)
    """
    tolerance = tol
    samples = th.zeros((net_ensemble.groups, nbatch, net_ensemble.nspin),
                device=net_ensemble.disorder.device, dtype = net_ensemble.disorder.dtype )
    cellsize= net_ensemble.cellsize
    with th.no_grad():
        # padding if the length less than receptive field size
        # n_pad = self.receptive_field - x.size(1)
        # if n_pad > 0:
        #     x = F.pad(x, (n_pad, 0), "constant", self.n_quantize // 2)
        for current_idx in range(net_ensemble.ncell):
            start_idx = current_idx * cellsize
            amp_output = net_ensemble.amp_net(net_ensemble._preprocess(samples))
            amp_normed, alive_mask = net_ensemble._postprocess(samples, amp_output, generative = True)
            ## (groups,nbatch,ncell, 2^cellsize)
            tolerable = alive_mask[:,:,current_idx]
            prob_current = amp_normed[:,:,current_idx, :]**2 ## (groups,nbatch, 2^cellsize)
            ## reweighting the distribution 
            prob_current[tolerable] = th.clamp(
                prob_current[tolerable], min=tolerance, max= 1-tolerance)  ## un-normalzied
            # prob_current = prob_current**(1-tol)
            prob_current = prob_current / prob_current.sum(-1).unsqueeze(-1)
            #########
            selected_idx = th.multinomial(
                prob_current.reshape(net_ensemble.groups*nbatch,-1)
                ,1).reshape(net_ensemble.groups,nbatch)
            selected_config = 1 - 2 * integer2bit(selected_idx, cellsize) # (groups, nabtch, cellsize)
            selected_prob = th.gather(
                prob_current, -1, selected_idx.unsqueeze(-1)
                                        ).squeeze(-1)
            samples[:,:,start_idx:start_idx + cellsize] = selected_config.type_as(samples)
            if current_idx == 0:
                prob_final =  selected_prob
            else:
                prob_final *= selected_prob                     
    return net_ensemble.recover(samples), prob_final
    
def archived_generate_otg_config(nbatch, net_ensemble, D=1):
    """GENERATE configuration with respect to coupled groups of given distance.
    Args:
        nbatch: size of the batch going to be generated 
        D: the distance between groups 
    Returns:
        samples(Tensor): Generated Configuration (groups, 2*nbatch, in_dim).
        samples_weight(Tensor): amplitude^2 of the generated configuration 
                                wrt the coupled adjacent groups of shape (groups-1, nbatch) 
    """
    dim_group, dim_batch, dim_spin = 0, 1, 2
    cellsize = net_ensemble.cellsize
    samples = th.zeros((net_ensemble.groups, nbatch, net_ensemble.nspin),
            device=net_ensemble.disorder.device, dtype = th.long )
    with th.no_grad():
        for current_idx in range(net_ensemble.ncell):
            start_idx = current_idx * cellsize
            inputs = th.cat([samples, th.roll(samples,shifts=D,dims=dim_group)], dim_batch)
            amp_output = net_ensemble.amp_net(net_ensemble._preprocess(inputs))
            # logging.info("current_idx:%i,amp_output:%s"%(current_idx, amp_output))
            amp_normed, alive_mask = net_ensemble._postprocess(inputs, amp_output, generative = True)
            amp2_normed = amp_normed**2  ## (groups,2*nbatch, ncell, 2^cellsize)
            amp2_normed_l = amp2_normed[:-D, :nbatch]  ## (groups-D, nbatch, ncell, 2^cellsize)
            amp2_normed_r = amp2_normed[D:,nbatch:]
            if current_idx > 0:
                amp2_prev_l = net_ensemble._select(
                    samples[:-D,:,:start_idx], amp2_normed_l[:,:,:current_idx]
                    )        ## (groups-D,nbatch,current_idx-1)
                logamp2_prev_l = th.log(th.clamp(amp2_prev_l,min=1e-8)).sum(-1) ## (groups,nbatch)
                amp2_prev_l = th.exp(logamp2_prev_l) ## (groups,nbatch)
                
                amp2_prev_r = net_ensemble._select(
                    samples[:-D,:,:start_idx], amp2_normed_r[:,:,:current_idx]
                    )        ## (groups-D,nbatch,current_idx-1)
                logamp2_prev_r = th.log(th.clamp(amp2_prev_r,min=1e-8)).sum(-1) ## (groups,nbatch)
                amp2_prev_r = th.exp(logamp2_prev_r) ## (groups,nbatch)
            else:
                amp2_prev_l = th.ones_like(amp2_normed_l[:,:,0,0])
                amp2_prev_r = th.ones_like(amp2_normed_r[:,:,0,0])
            amp2_prev_sum = amp2_prev_l + amp2_prev_r
            amp2_curr_l = amp2_normed_l[:,:, current_idx]   ## (groups-D,nbatch,2^cellsize)
            amp2_curr_r = amp2_normed_r[:,:, current_idx]   ## (groups-D,nbatch,2^cellsize)
            prob_current = (amp2_curr_l * amp2_prev_l.unsqueeze(-1) 
                        + amp2_curr_r * amp2_prev_r.unsqueeze(-1)) / amp2_prev_sum.unsqueeze(-1)
                        ## (groups-D,nbatch,2^cellsize)
            prob_current = prob_current.reshape((net_ensemble.groups-D)*nbatch, -1)
            if prob_current.min()<0:
                warnings.warn('current_idx={},min(prob_current)={}'.format(current_idx, prob_current.min()))
                prob_current = th.clamp(prob_current, min=0)
            if (prob_current.sum(-1)).min() <= 0:
                warnings.warn('some row in prob_current.sum(-1) = 0')
            # logging.info("current_idx=%i,prob_current:%s"%(current_idx,prob_current))
            selected_idx = th.multinomial(prob_current,1).reshape(net_ensemble.groups-D, nbatch)
            selected_config = 1 - 2 * integer2bit(selected_idx, cellsize) # (groups-D, nabtch, cellsize)
            samples[:-D,:,start_idx:start_idx + cellsize] = selected_config.type_as(samples)
            samples[-D:] = samples[:D]*1   ## regularize the auxilliary last dimension

        samples_final = th.cat([samples, th.roll(samples,shifts=D,dims=dim_group)], dim_batch)
        amp2_final_l = net_ensemble._select(
            samples[:-D], amp2_normed_l).prod(dim_spin)
        amp2_final_r = net_ensemble._select(
            samples[:-D], amp2_normed_r).prod(dim_spin)
        samples_weight = amp2_final_l + amp2_final_r
    return net_ensemble.recover(samples_final), samples_weight

def archived_generate_reotg_config(nbatch, net_ensemble, weighted = None):
    """GENERATE configuration with respect to sum of all spectral-wave-function.
    Args:
        nbatch: size of the batch going to be generated 
    Returns:
        samples(Tensor): Generated Configuration (nbatch, in_dim).
        samples_weight(Tensor): amplitude^2 of the generated configuration 
                                wrt the coupled adjacent groups of shape (nbatch) 
    """
    dim_group, dim_batch, dim_spin = 0, 1, 2
    device = net_ensemble.expH.device
    cellsize= net_ensemble.cellsize
    samples = th.zeros((nbatch, net_ensemble.nspin),device=device, dtype = th.long)
    if weighted is None:
        weight = th.ones_like(net_ensemble.groups_amp2[:,None]) / net_ensemble.groups 
    else:
        weight = weighted.unsqueeze(-1)
    with th.no_grad():
        for current_idx in range(net_ensemble.ncell):
            start_idx = current_idx * cellsize
            ########## predicted state ###########
            inputs = samples.unsqueeze(0).repeat(net_ensemble.groups,1,1)
            amp_output = net_ensemble.amp_net(net_ensemble._preprocess(inputs))
            amp_normed, alive_mask = net_ensemble._postprocess(
                inputs, amp_output, generative = True)
            
            amp2_normed = amp_normed**2 ## (groups, nbatch, ncell, 2^cellsize)
            if current_idx > 0:
                amp2_prev = net_ensemble._select(
                    inputs[:,:,:start_idx], amp2_normed[:,:, :current_idx]
                    )
                logamp2_prev = th.log(th.clamp(amp2_prev,min=1e-8)).sum(-1) ## (groups,nbatch)
                logamp2_prev = logamp2_prev + th.log(weight)
                amp2_prev = th.exp(logamp2_prev) ## (groups,nbatch)
                ### some amp2_prev is 0
            else:
                amp2_prev = weight.repeat(1,nbatch)
            amp2_current = amp2_normed[:,:, current_idx] ## (groups,nbatch,2^cellsize)
            ########
            amp2_combined = (amp2_current * amp2_prev.unsqueeze(-1)).sum(0)   ## (nbatch,2^cellsize)
            amp2_combined_base = (amp2_prev.sum(0)).unsqueeze(-1) ## (nbatch,1)
            prob_current = amp2_combined / amp2_combined_base
            ####### sanity check starts
            if amp2_prev.sum(0).min() <= 0:
                warnings.warn("amp2_prev=",amp_prev.sum(0))
            prob_sum = prob_current.sum(-1)
            for ii,pp in enumerate(prob_sum):
                if pp == 0:
                    logging.info("anomaly in prob_current=(%i,%s)"%(ii,prob_current[ii]))
                    logging.info("anomaly in amp2_current=(%i,%s)"%(ii,amp2_current[:,ii]))
            ####### sanity check ends
            selected_idx = th.multinomial(prob_current,1).reshape(nbatch)
            selected_config = 1 - 2 * integer2bit(selected_idx, cellsize) # (nabtch, cellsize)
            samples[:,start_idx:start_idx + cellsize] = selected_config.type_as(samples)

        amp2_final = net_ensemble._select(
            samples.unsqueeze(0).repeat(net_ensemble.groups,1,1), amp2_normed
            )  ## (groups, nbatch, ncell)
        amp2_final = ( amp2_final.prod(dim_spin) * weight ).sum(dim_group)
        samples_weight = amp2_final
        return net_ensemble.recover(samples), samples_weight
    
def archived_sample_overlap(nbatch, net_bra, net_ket):
    device = net_bra.expH.device
    ncell = net_bra.ncell
    cellsize= net_bra.cellsize
    nspin = net_bra.nspin
    ngroup_bra = net_bra.groups
    ngroup_ket = net_ket.groups
    sample_mat = th.zeros((ngroup_bra, ngroup_ket, nbatch, nspin),device=device, dtype = th.long)
    with th.no_grad():
        for current_idx in range(ncell):
            start_idx = current_idx * cellsize
            inputs_bra = sample_mat.reshape(ngroup_bra, -1, nspin)
            inputs_ket = sample_mat.transpose(0,1).reshape(ngroup_ket, -1, nspin)
            amp_output_bra = net_bra.amp_net(net_bra._preprocess(inputs_bra))
            amp_output_ket = net_ket.amp_net(net_ket._preprocess(inputs_ket))
            amp_normed_bra, alive_mask_bra = net_bra._postprocess(
                                    inputs_bra, amp_output_bra, generative = True)
            amp_normed_ket, alive_mask_ket = net_ket._postprocess(
                                    inputs_ket, amp_output_ket, generative = True)
            amp2_normed_bra = amp_normed_bra**2  ## (ngroup_bra, ngroup_ket*nbatch, ncell, 2**cellsize)
            amp2_normed_ket = amp_normed_ket**2
            if current_idx > 0:
                amp2_prev_bra = net_bra._select(
                    inputs_bra[:,:,:start_idx], amp2_normed_bra[:,:,:current_idx]
                    )        ## (ngroup_bra,ngroup_ket*nbatch,current_idx-1)
                amp2_prev_bra = th.exp(
                    th.log(th.clamp(amp2_prev_bra,min=1e-8)).sum(-1)
                ) ## (ngroup_bra,ngroup_ket*nbatch)
                
                amp2_prev_ket = net_ket._select(
                    inputs_ket[:,:,:start_idx], amp2_normed_ket[:,:,:current_idx]
                    )        ## (ngroup_ket,ngroup_bra*nbatch,current_idx-1)
                amp2_prev_ket = th.exp(
                    th.log(th.clamp(amp2_prev_ket,min=1e-8)).sum(-1)
                ) ## (ngroup_ket,ngroup_bra*nbatch)
            else:
                amp2_prev_bra = th.ones_like(amp_normed_bra[:,:,0,0])
                amp2_prev_ket = th.ones_like(amp_normed_ket[:,:,0,0])
            amp2_prev_bra_mat = amp2_prev_bra.reshape(ngroup_bra,ngroup_ket, nbatch)
            amp2_prev_ket_mat = amp2_prev_ket.reshape(ngroup_ket,ngroup_bra, nbatch).transpose(0,1)
            amp2_prev_sum_mat = amp2_prev_bra_mat + amp2_prev_ket_mat
            amp2_curr_bra_mat = amp2_normed_bra[:,:, current_idx].reshape(ngroup_bra,ngroup_ket, nbatch,-1)
            amp2_curr_ket_mat = amp2_normed_ket[:,:, current_idx].reshape(ngroup_ket,ngroup_bra, nbatch,-1).transpose(0,1)
            prob_current = (amp2_curr_bra_mat * amp2_prev_bra_mat.unsqueeze(-1) 
                        + amp2_curr_ket_mat * amp2_prev_ket_mat.unsqueeze(-1)) / amp2_prev_sum_mat.unsqueeze(-1)
                        ## (ngroup_bra,ngroup_ket, nbatch, 2^cellsize)
            prob_current = prob_current.reshape(ngroup_bra * ngroup_ket * nbatch, -1)
            # if prob_current.min()<0:
            #     warnings.warn('current_idx={},min(prob_current)={}'.format(current_idx, prob_current.min()))
            #     prob_current = th.clamp(prob_current, min=0)
            # if (prob_current.sum(-1)).min() <= 0:
            #     warnings.warn('some row in prob_current.sum(-1) = 0')
            # logging.info("current_idx=%i,prob_current:%s"%(current_idx,prob_current))
            selected_idx = th.multinomial(prob_current,1).reshape(ngroup_bra,ngroup_ket, nbatch)
            selected_config = 1 - 2 * integer2bit(selected_idx, cellsize) # (ngroup_bra,ngroup_ket, nbatch, cellsize)
            sample_mat[:,:,:,start_idx:start_idx + cellsize] = selected_config.type_as(sample_mat)

        amp2_final_bra = net_bra._select(
            sample_mat.reshape(ngroup_bra, -1, nspin), amp2_normed_bra).prod(-1)## (ngroup_bra,ngroup_ket*nbatch)
        amp2_final_ket = net_ket._select(
            sample_mat.transpose(0,1).reshape(ngroup_ket, -1, nspin), amp2_normed_ket).prod(-1)
        samples_weight = amp2_final_bra.reshape(ngroup_bra,ngroup_ket,nbatch) \
            + amp2_final_ket.reshape(ngroup_ket,ngroup_bra,nbatch).transpose(0,1) # (ngroup_bra,ngroup_ket,nbatch)
    return net_bra.recover(sample_mat), samples_weight