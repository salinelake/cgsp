
import numpy as np
from time import time as get_time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import logging
from myModules import *
from myFunctionals import *
from util import *
import warnings

class spectral_base(nn.Module):
    def __init__(self, sz_sum=None, nspin=10, pbc = True, cellsize=2, groups=1, nbands=1, disorder=None, target_bands = None):
        super(spectral_base, self).__init__()
        self.groups = groups
        self.nbands = nbands
        self.nspin = nspin
        ## sz conservation
        self.sz_sum=sz_sum
        if sz_sum is not None:
            assert (nspin+sz_sum) % 2 == 0
            self.s_up_sum = (self.nspin+self.sz_sum) / 2
        else:
            self.s_up_sum = None
        if disorder is not None:
            if th.is_tensor(disorder):
                self.register_buffer('disorder', disorder)
            else:
                self.register_buffer('disorder', th.tensor(disorder))
        ####################### Spectral properties   ######################
        if th.is_tensor(target_bands):
            self.register_buffer('target_bands', target_bands)
        else:
            self.register_buffer('target_bands', th.zeros(self.nbands))
        self.register_buffer('expH', th.zeros(self.nbands))
        self.register_buffer('stdH', th.ones(self.nbands))
        self.register_buffer('norm', th.zeros(self.nbands))
 
        ############ new setup of linear transformation
        self.ref_coef = nn.Parameter(th.ones(self.nbands,1)/self.nbands)
        _x = th.ones(self.nbands) / self.nbands**0.5 / 2
        ## a prior knowledge about the spectrum
        # _x[:self.nbands//2]*=0.1   ## suppress lower-half spectrum 
        # _x[self.nbands//2:]*=0.1   ## suppress upper-half spectrum 

        self.trans_amp = nn.Parameter(_x)
        self.trans_coef = nn.Parameter(th.randn(self.nbands,self.groups))
        
        ####################### lattice partition ######################
        self.cellsize = cellsize
        assert nspin % cellsize == 0,'number of spin is not divided by cell size'        
        self.ncell = nspin // cellsize ## the extra dimension is the fixed seed
        self.n_preproc = 1
        ## enforce periodic boundary condition
        self.pbc = pbc
        ### reorder cell-wise
        # self.reorder_idx = [i for i in range(self.ncell)]
        # self.recover_idx = [i for i in range(self.ncell)]
        # if self.pbc:
        #     self.reorder_idx = []
        #     for i in range(self.ncell//2):
        #         self.reorder_idx += [i,self.ncell-i-1]
        #     if self.ncell % 2 == 1:
        #         self.reorder_idx.append(self.ncell//2)
        #     for i, p in enumerate(self.reorder_idx):
        #         self.recover_idx[p] = i
        ## reorder spin-wise
        self.reorder_idx = [i for i in range(self.nspin)]
        self.recover_idx = [i for i in range(self.nspin)]
        if self.pbc:
            self.reorder_idx = []
            for i in range(self.nspin//2):
                self.reorder_idx += [i,self.nspin-i-1]
            if self.nspin % 2 == 1:
                self.reorder_idx.append(self.nspin//2)
            for i, p in enumerate(self.reorder_idx):
                self.recover_idx[p] = i
        ####################### Neural Networks Definition ######################
        self.amp_net = None
        self.phase_net = None
        
    def linear_transform(self):
        normed_coef = self.trans_coef / self.trans_coef.norm(dim=1).unsqueeze(-1) * self.trans_amp.unsqueeze(-1)  # (nbands,ngroup)
        trans = th.cat([self.ref_coef,normed_coef],-1)
        shift = - trans.sum(0)
        shift[0] = shift[0] + 1
        shift_weight = trans**2 / (trans**2).sum(0)
        result = trans + shift*shift_weight
        return result
    
    def faithful_transform(self, sita_list, ref_list):
        '''
        Gather spectral net(containing 'ngroups' subnets) outputs and reference net outputs. 
        Transform them into faithful CGSP representation of the reference state according to
        the linear transform matrix provided by spectral net
        Args: 
            sita_list: a list of same-shape tensor coming from spectral net, shape=(groups, batchsize)
            ref_list: a list of same-shape tensor coming from reference net, shape = (1,batchsize)
            net_ensemble: the spectral net
        Returns:
            psi_list: a list of same-shape tensor, shape = (nbands, batchsize)
        '''
        if type(sita_list) == list or type(sita_list) == tuple:
            if len(sita_list) != len(ref_list):
                raise ValueError("lists size not equivalent")
            psi_list = []    
            for sita, ref in zip(sita_list, ref_list):
                psi_list.append(
                    th.matmul(self.linear_transform(), th.cat([ref, sita],0))
                )
            return psi_list
        else:
            return th.matmul(self.linear_transform(), th.cat([ref_list, sita_list],0))
     
    def subnet_l2importance(self):
        M = self.linear_transform().detach()
        return (M**2).sum(0)
    
    def subnet_l1importance(self):
        M = self.linear_transform().detach()
        importance = th.abs(M).sum(0)
        importance[0] *= 0.1
        # importance = th.ones_like(importance)
        return importance 
    
    def data_parallel(self):
        self.amp_net = nn.DataParallel(self.amp_net, dim=1)
        try:
            self.phase_net = nn.DataParallel(self.phase_net, dim=1)
        except:
            pass

    def reassign_config(self, inputs):
        '''
        standardize spin configuration, outputs 3D tensor of type th.long.
        Args: 
            inputs(th.tensor or np.array, float or int): spin configurations, shape = (nbatch, nspin)
        Returns:
            configuration(th.long): spin configurations, shape = (nbatch, nspin)
        '''
        configuration= inputs.clone() if th.is_tensor(inputs) else th.tensor(inputs)
        if len(configuration.shape) == 2:
            configuration = configuration.to(device = self.expH.device, dtype=th.long)
        else:
            raise ValueError("dim of basis_input should be 2")
        return configuration
    
    def reorder(self, inputs):
        '''
        Args:
            inputs：spin configuration of shape (*,batch, nspin)
        Returns:
            reordered spin configuration of shape (*,batch,nspin)
        '''
        if inputs.shape[-1] != self.nspin:
            raise ValueError("illegal configuration shape")
        ## reorder cell-wisely
        # s_rearranged = inputs.reshape(-1, self.ncell, self.cellsize)[:, self.reorder_idx,:]
        ## reorder spin-wisely
        s_rearranged = inputs.reshape(-1, self.nspin)[:, self.reorder_idx]
        return s_rearranged.reshape(inputs.shape)
    
    def recover(self, inputs):
        '''
        Args:
            inputs：reordered spin configuration of shape (*,batch, nspin)
        Returns:
            natural ordered spin configuration (*,batch,nspin)
        '''
        if inputs.shape[-1] != self.nspin:
            raise ValueError("illegal configuration shape")
        ## reorder cell-wisely
        #s_rearranged = inputs.reshape(-1, self.ncell, self.cellsize)[:, self.recover_idx,:]
        ## reorder spin-wisely
        s_rearranged = inputs.reshape(-1, self.nspin)[:, self.recover_idx]
        return s_rearranged.reshape(inputs.shape)
    
    def _select(self, inputs, x):
        '''select desired spin component
        !! During generation, nspin is not necessarily self.nspin !!
        Args:
            inputs (Tensor or np.array): tensor with the shape (batch, nspin).
            x(Tensor) : unnormalized wave-function with shape (groups, batch, ncell, 2^cellsize)
        Returns:
            x_selected(Tensor): tensor with shape (groups, batch,  ncell)
        '''
        cellsize = self.cellsize
        s = self.reassign_config(inputs)
        s = s.unsqueeze(0).repeat(x.shape[0],1,1)
        groups, nbatch, nspin = s.shape
        select_spin = s.reshape(groups, nbatch, nspin//cellsize, cellsize)
        select_idx = bit2integer( (1-select_spin)/2, cellsize)
        return th.gather(x,-1,select_idx.unsqueeze(-1)).squeeze(-1)
    
    def _preprocess(self, inputs, for_phase=False):
        '''
        For amp_net: Add extra virtual spin on top, take away physical spin at bottom. 
        For phase_net: enforce PBC
        Args:
            configuration (Tensor or np.array): tensor with the shape (batch, nspin).
        Returns:
            output(th.tensor, float type): tailored configuration, shape=(1,batch, cellsize, ncell).  
            ## first dimension added for DataParallel
        '''
        s = self.reassign_config(inputs)
        s = s.to(dtype=self.expH.dtype)
        output = s.reshape(-1, self.ncell, self.cellsize).transpose(-1,-2) # (nbatch,cellsize,ncell)
        if for_phase == True:  ## pbc can be directly handled here without worrying generative function        
            if self.pbc:
                return th.cat([
                    output[:,:,-1].unsqueeze(-1), output, output[:,:,0].unsqueeze(-1)
                    ], -1).unsqueeze(0)
            else:
                return output.unsqueeze(0)
        else:
            extra = th.zeros_like(output[:,:,0]).unsqueeze(-1)
            output = th.cat([extra, output[:,:,:-1]],-1)
            return output.unsqueeze(0)
    
    def _CSmask(self, inputs):
        '''
        get mask for enforcing charge conservation
        Args:
            inputs (Tensor or np.array): preprocessed configuration, tensor with the shape (groups, batch, ncell, cellsize).
        Returns:
            alive_mask(Tensor): tensor with shape (groups, batch, ncell, 2^cellsize)
        '''
        s = inputs 
        groups, nbatch, ncell, cellsize = s.shape
        assert (self.sz_sum is not None)
        s_up_cumsum = ((s>0.1).sum(-1)).cumsum(-1)  #(groups, nbatch, ncell)
        s_dn_cumsum = ((s<-0.1).sum(-1)).cumsum(-1)  #(groups, nbatch, ncell)
        ###parallel implementation
        cell_spinconf = 1 - 2 * integer2bit(th.arange(2**cellsize),cellsize)
        cell_spinconf = cell_spinconf.to(dtype=th.long, device=s.device)
        cell_countup = (cell_spinconf>0).sum(-1) #(2**cellsize)
        cell_countdn = (cell_spinconf<0).sum(-1)
        up_mute_mask = (s_up_cumsum[:,:,:,None] + cell_countup[None,None,None,:]) > self.s_up_sum  # (groups, batch, ncell,2^cellsize)
        up_mute_mask = up_mute_mask.cumsum(-2) > 0
        dn_mute_mask = (s_dn_cumsum[:,:,:,None] + cell_countdn[None,None,None,:]) > (self.nspin-self.s_up_sum)  
        dn_mute_mask = dn_mute_mask.cumsum(-2) > 0
        mute_mask = up_mute_mask | dn_mute_mask
        alive_mask = ~mute_mask
        anomaly = ((mute_mask.sum(-1)) >= 2**cellsize).to(dtype=th.long)
        if anomaly.sum() > 0:
            for idx_g in range(groups):
                for idx_b in range(nbatch):
                    if anomaly[idx_g,idx_b].sum()>0:
                        logging.debug("mute_mask anomaly detected: group-{}, batch-{}".format(idx_g,idx_b))
                        logging.debug("abnormal mask = %s" % (mute_mask[idx_g,idx_b]))
                        logging.debug("Config = %s" % (s[idx_g,idx_b]))
        return alive_mask

    def _postprocess(self, inputs, amplitude):
        pass
    
    def forward(self):
        pass
    
    def unique_forward(self, basis_input):
        """cheaper FORWARD CALCULATION with th.unique
        Args:
            basis_input (Tensor or np.array): tensor with the shape (batch, in_dim).
        Returns:
            output(Tensor): wave function psi(configuration) with the shape (groups, batch) .
        """
        nbatch = basis_input.shape[0]
        unique_input, inverse_idx, counts = th.unique(basis_input, sorted=False, return_inverse=True,return_counts=True, dim=0)
        unique_output_r, unique_output_i = self.forward(unique_input)  ## (group, unique_batch)
        output_r = th.gather(unique_output_r,
                             dim=1,
                             index = inverse_idx.expand(self.groups,nbatch))
        output_i = th.gather(unique_output_i,
                             dim=1,
                             index = inverse_idx.expand(self.groups,nbatch))
        return output_r, output_i
    
    def faithful_forward(self, basis_input, net_ref):
        """ faithful forward: include reference state
        Args:
            basis_input (Tensor or np.array): tensor with the shape (batch, in_dim).
            net_ref:(th.module): reference state
        Returns:
            output(Tensor): wave function psi(configuration) with the shape (nbands, batch) .
        """
        sita_r, sita_i = self.unique_forward(basis_input)
        ref_r, ref_i = net_ref(basis_input)
        return self.faithful_transform([sita_r,sita_i],[ref_r, ref_i])    

class unentangled_net(spectral_base):  ## TODO: phase net
    def __init__(self, sz_sum=None, nspin=10, pbc = True, cellsize=2, groups=1, stype='neel'):
        super(unentangled_net, self).__init__(sz_sum, nspin, pbc, cellsize, groups)
        if stype == 'neel':
            self.register_buffer('reference_state', th.tensor([-1,1]*(nspin//2)).type(th.long))
            epsilon = 0.0
            unentangled_amp = th.zeros(nspin,2)
            unentangled_amp[0::2,0] += epsilon
            unentangled_amp[1::2,0] += (1 - epsilon**2)**0.5
            unentangled_amp[:,1] = (1 - unentangled_amp[:,0]**2)**0.5
            unentangled_amp = self.reorder(unentangled_amp.transpose(0,1)).transpose(0,1)
            # unentangled_phase = th.zeros(nspin,2)
        elif stype == 'wall':
            self.register_buffer('reference_state', th.tensor([-1]*(nspin//2)+[1]*(nspin//2)).type(th.long))
            epsilon = 0.0
            unentangled_amp = th.zeros(nspin,2)
            unentangled_amp[:(nspin//2),0] += epsilon
            unentangled_amp[(nspin//2):,0] += (1 - epsilon**2)**0.5
            unentangled_amp[:,1] = (1 - unentangled_amp[:,0]**2)**0.5
            unentangled_amp = self.reorder(unentangled_amp.transpose(0,1)).transpose(0,1)
            # unentangled_phase = th.zeros(nspin,2)
        else:
            raise NotImplementedError
        self.amp_net = amp_net_unentangled(unentangled_amp)
        # self.phase_net = phase_net_unentangled(nspin, self.ncell, cellsize, unentangled_phase)
        
    def reinit(self,epsilon):
        unentangled_amp = th.zeros(self.nspin,2)
        unentangled_amp[0::2,0] += epsilon
        unentangled_amp[1::2,0] += (1 - epsilon**2)**0.5
        unentangled_amp[:,1] = (1 - unentangled_amp[:,0]**2)**0.5
        unentangled_amp = self.reorder(unentangled_amp.transpose(0,1)).transpose(0,1)
        self.amp_net = amp_net_unentangled(unentangled_amp).to(device=self.reference_state.device)
        
    def _postprocess(self, inputs, amplitude, noise=False):
        '''
        Enforce charge conservation and patch anomaly, then normalize probability
        Args:
            inputs (Tensor or np.array): preprocessed configuration, tensor with the shape (1, batch, cellsize, ncell).
            amplitude(Tensor) : unnormalized amplitude with shape (groups, batch, 2^cellsize, ncell)
        Returns:
            amp_normed(Tensor): tensor with shape (groups, batch, ncell, 2^cellsize)
      
        '''
        amplitude = amplitude.transpose(-1,-2)   # (groups, batch, ncell,2^cellsize)  ## in-place update to save memory 
        groups, nbatch, ncell, _ = amplitude.shape
        if self.sz_sum is None:   ## no need to worry about charge conservation!
            alive_mask = th.ones_like(amplitude,dtype=th.bool)
        else: ## exert charge conservation
            alive_mask = self._CSmask(inputs.transpose(-1,-2)).repeat(groups,1,1,1)
        amplitude = amplitude * alive_mask.to(dtype=amplitude.dtype)
        amp2 = (amplitude.detach()**2).sum(-1)   #(groups, nbatch, ncell)
        needs_regular = (amp2 <= 0).to(dtype=th.long)
        amplitude += (alive_mask.to(dtype=th.long) * needs_regular.unsqueeze(-1)).type_as(amplitude)
        amplitude = amplitude / ((amplitude**2).sum(-1).unsqueeze(-1))**0.5   ## normalized amplitude
        if noise:
            epsilon = 0
            amplitude += epsilon * alive_mask.type_as(amplitude)
            amplitude = amplitude / ((amplitude**2).sum(-1).unsqueeze(-1))**0.5   ## normalized amplitude
        return amplitude
    
    def forward(self, basis_input):
        """FORWARD CALCULATION.
        Args:
            basis_input (Tensor or np.array): tensor with the shape ((groups,) batch, in_dim).
        Returns:
            output(Tensor): wave function psi(configuration) with the shape (groups, batch) .
        """
        ## amp part
        configuration = self.reorder(self.reassign_config(basis_input))
        preped_config = self._preprocess(configuration)
        amp = self.amp_net(preped_config)  ## unsqueeze to get 4D input(required by DataParallel)
        amp = self._select(configuration, self._postprocess(preped_config, amp))
        amp = amp.prod(-1)
        psi_r = amp 
        psi_i = th.zeros_like(psi_r)
        return psi_r , psi_i    


class spectral_net(spectral_base):
    def __init__(self, sz_sum=None, nspin=10, pbc = True, cellsize=2, groups=1, nbands=1, disorder=None, 
                n_resch=64, n_phase_hidden = 64, n_skipch=2,
                dilation_depth=3, dilation_repeat=1, kernel_size=2,
                target_bands = None):
        super(spectral_net, self).__init__(sz_sum, nspin, pbc, cellsize, groups, nbands, disorder, target_bands)
        ## Hyperparameters for amp-net
        self.n_resch = n_resch        ## residue channel
        self.n_skipch = 2**cellsize      ## skip layer output channel =  quantum number
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.dilation_repeat = dilation_repeat
        self.dilations = [2 ** i for i in range(self.dilation_depth)] * self.dilation_repeat
        self.receptive_field = 2+(self.kernel_size - 1) * sum(self.dilations) + 1
        assert self.receptive_field >= self.ncell
        ## Define neural network
        self.amp_net = amp_net_mixed(ncell=self.ncell, cellsize=self.cellsize, groups=groups, n_preproc=self.n_preproc, 
                n_resch=n_resch, n_skipch=self.n_skipch, dilations=self.dilations, kernel_size=kernel_size)
        self.phase_net = phase_cnn_mixed(groups=groups, ncell=self.ncell+2*int(self.pbc), 
                cellsize = self.cellsize, hidden_channel = n_phase_hidden)

    def _postprocess(self, inputs, amplitude):
        '''
        Enforce charge conservation and patch anomaly, then normalize probability
        Args:
            inputs (Tensor or np.array): preprocessed configuration, tensor with the shape (1, batch, cellsize, ncell).
            amplitude(Tensor) : unnormalized amplitude with shape (groups, batch, 2^cellsize, ncell)
        Returns:
            amp_normed(Tensor): tensor with shape (groups, batch, ncell, 2^cellsize)
        '''
        amplitude = amplitude.transpose(-1,-2)   # (groups, batch, ncell,2^cellsize)  ## in-place update to save memory 
        groups, nbatch, ncell, _ = amplitude.shape
        
        if self.sz_sum is None:   ## no need to worry about charge conservation!
            alive_mask = th.ones_like(amplitude,dtype=th.bool)
        else: ## exert charge conservation
            alive_mask = self._CSmask(inputs.transpose(-1,-2)).repeat(groups,1,1,1)
        amplitude = amplitude * alive_mask.to(dtype=amplitude.dtype)
        ############################## sanity check ######################################
        amp2 = (amplitude.detach()**2).sum(-1)   #(groups, nbatch, ncell)
        anm_flag = (amp2 <= 0)
        if amp2.min()<= 0:
            anomaly = (anm_flag.type(th.long).sum(-1)) > 0  
            logging.debug("{} anomaly detected".format(anomaly.type(th.long).sum()))
            logging.debug("abnormal amplitude= %s" % (amplitude[anomaly]))
            # logging.debug("unmasked amplitude= %s" % (amplitude_copy[anomaly]))
            logging.debug("Config = %s" % (inputs.transpose(-1,-2).repeat(groups,1,1,1)[anomaly]))
            ## patch the anomaly
            needs_regular = anm_flag.to(dtype=th.long)
            amplitude += (alive_mask.to(dtype=th.long) * needs_regular.unsqueeze(-1)).type_as(amplitude)
        ##################################################################################
        amplitude = amplitude / ((amplitude**2).sum(-1).unsqueeze(-1))**0.5   ## normalized amplitude
        return amplitude
    
    def forward(self, basis_input):
        """FORWARD CALCULATION.
        Args:
            basis_input (Tensor or np.array): tensor with the shape (batch, in_dim).
        Returns:
            output(Tensor): wave function psi(configuration) with the shape (groups, batch) .
        """
        ## phase part
        # phase = self.phase_net(self._preprocess(self.reassign_config(basis_input), for_phase=True))
        ## amp part
        configuration = self.reorder(self.reassign_config(basis_input))
        preped_config = self._preprocess(configuration)
        amp = self.amp_net(preped_config)
        amp = self._select(configuration, self._postprocess(preped_config, amp))
        amp = amp.prod(-1)
        ## real wave function
        psi_r = amp
        psi_i = th.zeros_like(amp)
        ## complex wave function
        # psi_r = amp * th.cos(phase)
        # psi_i = amp * th.sin(phase)
        return psi_r , psi_i    

class ensemble2ref_wrapper(nn.Module):  ## TODO
    def __init__(self, net_ensemble, net_ref, active_group):
        super(ensemble2ref_wrapper, self).__init__()
        self.net_ensemble = net_ensemble
        self.net_ref = net_ref
        self.reference_state = net_ref.reference_state
        self.groups = 1
        self.active_group = active_group
        self.expH = net_ensemble.expH[active_group]
        self.stdH = net_ensemble.stdH[active_group]
        self.norm = net_ensemble.norm[active_group]
        self.is_wrapped = True
    
    def forward(self, basis_input):
        '''
        Returns wave function psi(basis_input) with the shape (batch).
        '''
        with th.no_grad():
            psi_r,psi_i = self.net_ensemble.faithful_forward(basis_input, self.net_ref)
            return psi_r[self.active_group].unsqueeze(0), psi_i[self.active_group].unsqueeze(0)
