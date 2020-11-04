import os
import numpy as np
import torch as th
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import anytree
from anytree import AnyNode, LevelOrderIter, RenderTree
from anytree.exporter import DictExporter, JsonExporter
from anytree.importer import DictImporter, JsonImporter
from anytree.search import findall as findnodes

import pickle
import logging
from time import time as get_time

from spectral_net import *
from sampler import *
from myFunctionals import complex_mul
from hilbert import *
from util import *

class spectral_projector:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.nspin = config.nspin
        hz = config.hz.clone() if th.is_tensor(config.hz) else th.tensor(config.hz)
        model_paras = {
            'dim': config.model_dim,
            'device' : config.device,
            'dtype' : config.dtype,
            'Lx' : config.nspin,
            'Jx' : config.Jx,
            'Jz' : config.Jz,
            'hz' : hz.to(device= self.device, dtype =self.dtype),
            'conserve':'Sz',
            'sz_conserve': config.total_sz,
            'pbc_x': config.pbc_x,
        }
        if config.model_name == 'xxz1d':
            self.hilbert = heisenberg(model_paras)
        else:
            raise NotImplementedError
        self.reassign = reassign_device_dtype(self.device, self.dtype)
        
    def create_net(self, groups, nbands, target_bands = None):
        config = self.config
        dilation_depth = int(np.ceil(np.log(self.nspin//config.cell_size-1) / np.log(2)))
        dilation_depth = np.maximum(dilation_depth, 2)
        if target_bands is not None:
            if nbands != target_bands.shape[0]:
                raise ValueError('nbands={} != #target_bands={}'.format(nbands, target_bands.shape[0]))

        net_ensemble = self.reassign(
                spectral_net(
                            sz_sum=config.total_sz, 
                            nspin=self.nspin, 
                            pbc = config.pbc_x,
                            cellsize=config.cell_size, 
                            groups = groups, 
                            nbands = nbands,
                            disorder=config.hz,
                            n_resch = config.hidden,
                            n_phase_hidden = config.phase_hidden,
                            dilation_depth = dilation_depth, 
                            dilation_repeat = 1, 
                            kernel_size=2, 
                            target_bands = target_bands
                            )
        )
        if self.config.num_device > 1:
            net_ensemble.data_parallel()
        return net_ensemble
    
    def extremal_state(self, option = 'min'): # an old version; needs rewriting
        comment = self.config.experiment_name + 'net' + option
        writer = SummaryWriter(comment=comment)
        ConfigDict = self.config.train_config
        net = self.create_net(groups=1)
        E_history = []
        target = -1 if option=='min' else 1
        ############### Training
        batchsize = ConfigDict['batchsize_begin']
        if self.config.sampling_mode == 'MC':
            pass
        else:
            basis_input_full = self.hilbert.get_full_basis().unsqueeze(0)
        net.set_entanglement(False)
        lr = 0.02
        optimizer_adam = th.optim.Adam(net.parameters(), lr = lr, betas = (0.9,0.999))

        for epoch in range(ConfigDict['epoches']):
            if epoch == 4000:
                lr = ConfigDict['lr_begin'] 
                net.set_entanglement(True)
                optimizer_adam = lr_scheduler(optimizer_adam, lr)
            if self.config.sampling_mode == 'MC':
                expH, varH = self.mc_energy_expectation(batchsize, net, target,tol=0.01)
            else:
                psi_r, psi_i = net(basis_input_full)
                Hpsi_r, Hpsi_i = self.Hnet(net, basis_input_full)
                HmEpsi_norm2 = (Hpsi_r - psi_r * target)**2 + (Hpsi_i - psi_i * target)**2
                varH = HmEpsi_norm2.sum()
                with th.no_grad():
                    expH = (psi_r * Hpsi_r + psi_i * Hpsi_i).sum()
            E_history.append(th2np(expH))
            loss = varH
            optimizer_adam.zero_grad()
            loss.backward()
            optimizer_adam.step()
            if epoch % ConfigDict['freq_display'] == 0 :
                print("{} energy level = {} ".format(option,expH))
                writer.add_scalar('var_extm', varH, epoch)
                writer.add_scalar('expH_extm', expH, epoch)
                writer.add_scalar('entanglement_switch',net.amp_net.entanglement_switch,epoch)
                writer.add_scalar('learning_rate', lr, epoch)
                writer.add_scalar('batchsize', batchsize, epoch)
                logging.info("=========== Solving %s states, epoch : %i/%i ==========="%
                             (option, epoch, ConfigDict['epoches']))
                logging.info("expH: %f; varH: %f" % (th2np(expH), th2np(varH)))
                if self.nspin < 16 and self.config.sampling_mode == 'MC' and epoch % 1000==0:
                    test_psi_r, test_psi_i = net(basis_input_full)
                    test_Hpsi_r, test_Hpsi_i = self.Hnet(net, basis_input_full)
                    test_HmEpsi_norm2 = (test_Hpsi_r - test_psi_r * target)**2 + (test_Hpsi_i - test_psi_i * target)**2
                    test_psi_var = test_HmEpsi_norm2.sum()
                    test_expH = (test_psi_r * test_Hpsi_r + test_psi_i * test_Hpsi_i).sum()
                    writer.add_scalar('true_var_extm', test_psi_var, epoch)
                    writer.add_scalar('true_expH_extm', test_expH, epoch)
            if epoch % ConfigDict['freq_test'] == 0:
                lr = np.maximum( lr * ConfigDict['lr_scale_fac'], ConfigDict['lr_end'])
                optimizer_adam = lr_scheduler(optimizer_adam, lr)
                batchsize = np.minimum(int(batchsize * 1.4), ConfigDict['batchsize_end'])
        writer.close()
        return np.array(E_history[-50:]).mean()
    
    def mc_project(self,net_ensemble, net_ref, ref_label='0'):
        comment = self.config.experiment_name + 'net' + ref_label
        writer = SummaryWriter(comment=comment)
        ConfigDict = self.config.train_config
        bandwidth = (net_ensemble.target_bands[-1] -net_ensemble.target_bands[0])/(net_ensemble.target_bands.shape[0]-1)
        if hasattr(net_ref, 'is_wrapped'):
            ref_norm = net_ref.norm
        else:
            ref_norm = 1
        scale_factor =  (bandwidth/2*ref_norm)**2
        sampler = mc_sampler(net_ref, net_ensemble)
        ################ Training ######################
        batchsize = ConfigDict['batchsize_begin']
        lr = ConfigDict['lr_begin']
        optimizer_adam = th.optim.Adam([
            {'params': net_ensemble.amp_net.parameters()},
            {'params': net_ensemble.phase_net.parameters()},
            {'params': net_ensemble.ref_coef},
            {'params': net_ensemble.trans_amp, 'lr': 3e-5},
            {'params': net_ensemble.trans_coef},
            ], lr = lr, betas=(0.95, 0.99))
        #optimizer_adam = th.optim.Adam(net_ensemble.parameters(), lr = lr, betas=(0.95, 0.99))
        pretrain_epoch = 5000
        for epoch in range(ConfigDict['epoches']):
            start_time = get_time()
            optimizer_adam.zero_grad()
            if epoch < pretrain_epoch:
                weighted_variance_pre, expH, varH, norm2 = sample_weighted_variance_unified_importance_exact(
                    self.hilbert, net_ref, net_ensemble, backbone=True)
                loss_pre = weighted_variance_pre / scale_factor
                # loss_pre.backward()
                (loss_pre* (1 - epoch/pretrain_epoch)).backward()
            weighted_variance, expH, varH, norm2 = sample_weighted_variance_unified_importance(
                batchsize, self.hilbert, sampler, net_ref, net_ensemble)
            loss = weighted_variance / scale_factor   
            (loss*np.minimum(epoch/pretrain_epoch,1.0)).backward()
            optimizer_adam.step()
            loss_avg = th2np(loss) if epoch==0 else (0.99 * loss_avg + 0.01 * th2np(loss))
            # statistics & logging
            if epoch % ConfigDict['freq_display'] == 0:
                if epoch > 1000:
                    net_ensemble.expH = 0.99 * net_ensemble.expH + 0.01 * expH
                    net_ensemble.stdH = 0.99 * net_ensemble.stdH + 0.01 * varH**0.5
                    net_ensemble.norm = 0.99 * net_ensemble.norm + 0.01 * norm2**0.5
                logging.info("======== Reference label:%s, ref_bands=(exp=%f) ========"
                        %(ref_label,net_ref.expH ))
                logging.info("Target bands: %s"%(net_ensemble.target_bands))
                logging.info("Epoch : %i/%i,  num_data:%i,lr: %f,  elapsed_time: %f"
                             %(epoch, ConfigDict['epoches'], batchsize, lr, get_time() - start_time))
                logging.info("weighted_variance: %s" % (loss))
                logging.info('refnorm : %s, amp2_sum/refnorm2 : %s' % (ref_norm, norm2.sum()/ref_norm**2))
                logging.info('l1importance : %s' % (net_ensemble.subnet_l1importance()/ref_norm))
                # logging.info("l2importance:%s" % ( net_ensemble.subnet_l2importance()))
                logging.info("stdH:%s" % ( (varH/scale_factor)**0.5))
                logging.info("(expH - E)/sigma: %s" % (
                    (net_ensemble.expH - net_ensemble.target_bands)/ scale_factor**0.5))
                logging.info("norm/refnorm: %s" % (norm2**0.5/ref_norm))
                writer.add_scalar('batchsize', batchsize, epoch)
                writer.add_scalar('learning_rate', lr, epoch)
                writer.add_scalar('l2importance',net_ensemble.subnet_l2importance()[1:].sum()/ref_norm**2, epoch)
                writer.add_scalar('l1importance',net_ensemble.subnet_l1importance()[1:].sum()/ref_norm, epoch)
                writer.add_scalar('amp2_sum', norm2.sum()/ref_norm**2, epoch)
                writer.add_scalar('sum_weighted_var_loss', weighted_variance / scale_factor  , epoch)
                if epoch < pretrain_epoch:
                    writer.add_scalar('pre_weighted_var_loss', weighted_variance_pre / scale_factor , epoch)
                writer.add_scalar('sum_weighted_var_loss_avg', loss_avg, epoch)
            ## save and do convergence test
            if epoch > ConfigDict['start_epoch'] and epoch % ConfigDict['freq_test'] == 0:
                self.save(net_ensemble, ref_label)
                print('net saved,epoch=',epoch)
        return

    def exact_project(self, net_ensemble, net_ref, ref_label='0'): 
        comment = self.config.experiment_name + 'net' + ref_label
        writer = SummaryWriter(comment=comment)
        ConfigDict = self.config.train_config
        bandwidth = (net_ensemble.target_bands[-1]-net_ensemble.target_bands[0])/(net_ensemble.target_bands.shape[0]-1)
        scale_factor = (bandwidth/2)**2
        ############  Parameters  ############
        lr = ConfigDict['lr_begin']
        optimizer_adam = th.optim.Adam(net_ensemble.parameters(), lr = lr, betas=(0.9, 0.999))
        for epoch in range(ConfigDict['epoches']):
            start_time = get_time()
            weighted_variance, expH, varH, norm2 = sample_weighted_variance_unified_importance_exact(
                self.hilbert, net_ref, net_ensemble)
            net_ensemble.expH = expH
            loss = weighted_variance / scale_factor
            optimizer_adam.zero_grad()
            loss.backward()
            optimizer_adam.step()
            ## statistics & logging
            with th.no_grad():
                ## display statistics
                if epoch % ConfigDict['freq_display'] == 0:
                    writer.add_scalar('learning_rate', lr, epoch)
                    # writer.add_scalar('amp2_sum',net_ensemble.groups_amp2_sum, epoch)
                    writer.add_scalar('amp2_sum',norm2.sum(), epoch)
                    writer.add_scalar('sum_weighted_var_loss', loss , epoch)
                    writer.add_scalar('l2importance',net_ensemble.subnet_l2importance()[1:].sum(), epoch)
                    logging.info("======== Reference label:%s, ref_bands=%f Target bands: %s========"
                            %(ref_label, net_ref.expH, net_ensemble.target_bands))
                    logging.info("epoch : %i/%i,  elapsed_time: %f"
                            %(epoch, ConfigDict['epoches'], get_time() - start_time))
                    logging.info("loss: %f " %(loss))
                    logging.info("std: %s" % ((varH/scale_factor)**0.5))
                    logging.info("norm: %s" % (norm2**0.5))
                    logging.info('amp2_sum : %s' % (norm2.sum()))
                    logging.info('l1importance : %s' % (net_ensemble.subnet_l1importance()))
                    #logging.info('l2importance : %s' % (net_ensemble.subnet_l2importance()))
                    #logging.info("amplitude: %s" % (net_ensemble.groups_amplitude))
                ## convergence test
                if epoch > 0 and epoch % ConfigDict['freq_test'] == 0:
                    self.save(net_ensemble, ref_label)
                    lr = np.maximum( lr * ConfigDict['lr_scale_fac'], ConfigDict['lr_end'])
                    optimizer_adam = lr_scheduler(optimizer_adam, lr)
        writer.close()
        return
    
    def save(self, net_ensemble, ref_label):
        directory = './nn_data/{}/{}.net'.format(self.config.experiment_name, ref_label)
        th.save(net_ensemble.state_dict(), directory)
        logging.info('=======================================================')
        logging.info('net{} has been saved..'.format(ref_label))
        logging.info('=======================================================')

        
class spectral_tree(spectral_projector):
    def __init__(self, config):
        super(spectral_tree, self).__init__(config)
        self.hierarchy = config.hierarchy
        self.hierarchy_groups = config.hierarchy_groups
        self.nbands = config.nbands
        
        self.init_state = self.reassign(
            unentangled_net(config.total_sz, self.nspin, 
                       pbc=self.config.pbc_x,
                       cellsize=config.cell_size,
                       groups=1, 
                       stype = 'wall')#'neel')
            )  

        self.root = AnyNode(id='0')
    
    def id2int(self, net_id):
        return [int(i) for i in net_id.split('-')]
    
    def get_node(self, net_id):
        for node in LevelOrderIter(self.root):
            if node.id == net_id:
                mynode = node
                break
        return mynode
    
    def generate_bands(self, ub, lb, num_bands): 
        '''
        Generate equally partitioned bands between ub and lb.  ## no points at the two ends
        '''
        bandwidth = (ub - lb) / num_bands
        bands = th.arange(num_bands,dtype=self.dtype,device=self.device) * bandwidth + lb + bandwidth/2
        return bands
    
    def generate_children_from(self, node): 
        '''
        Returns the index of the subnet of which the amplitude is larger than threshold
        '''
        child_id_list = []
        amp = node.net.norm.detach()
        for i in range(node.net.groups):
            if amp[i] > 0.1: #/ amp.sum() > 0.01:
                child_id = node.id + '-' + str(i)
                child_node = AnyNode(id = child_id, 
                                     parent = node, 
                                     )
                child_id_list.append(child_id)
        return child_id_list
    
    def init_tree(self):
        root = self.root
        target_bands = self.generate_bands(
            self.config.energy_ub, self.config.energy_lb, self.nbands[0])
        root.net = self.create_net(groups = self.hierarchy_groups[0], 
                                   nbands = self.nbands[0],
                                   target_bands = target_bands)
        self.load_model('0')
        if self.config.sampling_mode == 'full':
            self.exact_project(root.net, self.init_state, root.id)
        else:
            self.mc_project(root.net, self.init_state, root.id)
        return 
    
    def train_node(self, net_id):
        mynode = self.get_node(net_id)
        parent_node = mynode.parent
        net_id_int = [int(i) for i in net_id.split('-')]
        active_parent_band = net_id_int[-1]
        self.load_model(parent_node.id)
        ## setup subnet of parent_node as the initial state surrogate
        if parent_node.depth == 0:
            parent_ref = self.init_state
        else:
            NotImplementedError('needs to implement recursive net wrapper')
        net_ref = ensemble2ref_wrapper(net_ensemble = parent_node.net, 
                                       net_ref = parent_ref, 
                                       active_group = active_parent_band)
        ## determine the target_bands
        band_center = parent_node.net.expH[active_parent_band]
        std = parent_node.net.stdH[active_parent_band]
        target_bands = self.generate_bands(band_center + 3*std,
                                           band_center - 3*std,
                                           self.nbands[mynode.depth])
        mynode.net = self.create_net(self.hierarchy_groups[mynode.depth], 
                                     self.nbands[mynode.depth],
                                     target_bands = target_bands)
        mynode.net.trans_amp.data *= net_ref.norm  ## use some preknowledge about the overall amplitude
        ## if model already exists, load
        self.load_model(net_id)
        if self.config.sampling_mode == 'full':
            self.exact_project(mynode.net, net_ref, net_id)
        else:
            self.mc_project(mynode.net, net_ref, net_id)
        return
  
    def save_tree(self, root, directory=None):
        data_path = (directory or '.') + '/nn_data/' + self.config.experiment_name + '/spectral_tree.dict'
        attriter = lambda attrs: [(k, v) for k, v in attrs if k != 'net']
        ## save as pickled dict
        dict_exporter = DictExporter(attriter=attriter)
        treedict = dict_exporter.export(root)
        with open(data_path, 'wb') as outfile:
            pickle.dump(treedict, outfile)
            
    def load_tree(self, directory=None):
        data_path = (directory or '.') + '/nn_data/' + self.config.experiment_name + '/spectral_tree.dict'
        if os.path.exists(data_path):  
            importer = DictImporter()
            with open(data_path,'rb') as file:
                treedict = pickle.load(file)
                self.root = importer.import_(treedict)
        print(RenderTree(self.root).by_attr('id'))
  
    def load_model(self, net_id, directory=None):
        net_path = (directory or '.') + '/nn_data/' + self.config.experiment_name + '/{}.net'.format(net_id)
        if os.path.exists(net_path):
            for node in LevelOrderIter(self.root):
                if node.id == net_id:
                    node.net = self.create_net(self.hierarchy_groups[node.depth], self.nbands[node.depth])
                    node.net.load_state_dict(th.load(net_path, map_location = self.device))
                    logging.info('=======================================================')
                    logging.info('Loaded model from %s'%(net_path) )
                    logging.info('=======================================================')
                    break
        
    
    def save_model(self, net_id):
        directory = './nn_data/' + self.config.experiment_name + '/{}.net'.format(net_id)
        for node in LevelOrderIter(self.root):
            if node.id == net_id:
                th.save(node.net.state_dict(), directory)
                break
        logging.info('=======================================================')
        logging.info('Saved model to %s'%(directory) )
        logging.info('=======================================================')
        print('Saved model to %s'%(directory) )

    def load_all_model(self, leaf_only = False, directory = None):
        net_path = (directory or '.') + '/nn_data/' + self.config.experiment_name + '/{}.net'
        if leaf_only:
            for leaf in self.root.leaves:
                leaf.net = self.create_net(self.hierarchy_groups[leaf.depth], self.nbands[node.depth])
                leaf.net.load_state_dict(th.load(net_path.format(leaf.id), map_location = self.device))
        else:
            for node in LevelOrderIter(self.root):
                node.net = self.create_net(self.hierarchy_groups[node.depth], self.nbands[node.depth])
                node.net.load_state_dict(th.load(net_path.format(node.id), map_location = self.device))
        logging.info('=======================================================')
        logging.info('Loaded all model from %s'%(directory) )
        logging.info('=======================================================')
    
    def edgeWavefunc(self, basis_input, t, depth=0):
        total_psi_r, total_psi_i = 0, 0
        for node in LevelOrderIter(self.root):
            children_list = [self.id2int(child.id)[-1] for child in node.children if child.depth<=depth]
            leaves_list = [i for i in range(node.net.nbands) if i not in children_list]
            expH = node.net.expH
            ## assign ref_net
            if node.depth == 0:
                net_ref = self.init_state
            elif node.depth == 1:
                net_id_int = [int(i) for i in node.id.split('-')]
                net_ref = ensemble2ref_wrapper(node.parent.net, self.init_state, net_id_int[-1])
            else:
                NotImplementedError('needs to implement recursive net wrapper')
            with th.no_grad():
                psi_r, psi_i = node.net.faithful_forward(basis_input, net_ref) ## (nbands, nbatch)
                psi_r, psi_i = complex_mul(psi_r,psi_i,th.cos(-expH*t)[:,None],th.sin(-expH*t)[:,None])
            total_psi_r += psi_r[leaves_list].sum(0)
            total_psi_i += psi_i[leaves_list].sum(0)
        return total_psi_r, total_psi_i 
    
    def get_observation(self, depth=0, t=0,batchsize=4000,exact=False): 
        if exact:
            basis_input = self.hilbert.get_full_basis()
            nspin = basis_input.shape[-1]
            np_basis_input = th2np(basis_input)        
            psi_r, psi_i = self.edgeWavefunc(basis_input,t, depth)
            np_psi = th2np(psi_r) + 1j * th2np(psi_i)
            entropy = vonNeumannEntropy(np_basis_input, np_psi)
            np_psi2 = th2np(psi_r ** 2 + psi_i ** 2)
            szsz_avg = np.zeros(nspin//2)
            if self.config.pbc_x:
                for l in range(1,nspin//2+1):
                    szsz_avg[l-1] = (np_psi2 * 
                        (np_basis_input * np.roll(np_basis_input,shift=l,axis=1)).mean(-1)
                        ).sum() / np_psi2.sum()
            else:
                for l in range(1,nspin//2+1):
                    szsz_avg[l-1] = (np_psi2 * 
                        (np_basis_input[:,:-l] * np_basis_input[:,l:]).mean(-1)
                        ).sum() / np_psi2.sum()
            sdw_avg = np.matmul(np_psi2, np_basis_input) / np_psi2.sum()
            stagger_avg = sdw_avg[1::2].sum() - sdw_avg[0::2].sum()
            return entropy, szsz_avg, stagger_avg, sdw_avg
        else:
            basis_input, mcWeight = generate_fid_config(batchsize,self.root.net,self.init_state,gamma=0.5)
            nspin = basis_input.shape[-1]
            psi_r, psi_i = self.edgeWavefunc(basis_input,t, depth)
            psi2 = psi_r**2 + psi_i**2
            mirror_szsz_avg = th.zeros(nspin//2,dtype=psi2.dtype,device=psi2.device)
            szsz_avg = th.zeros(nspin//2,dtype=psi2.dtype,device=psi2.device)
            if self.config.pbc_x:
                for l in range(1,nspin//2+1):
                    szsz_observable = (basis_input * basis_input.roll(l,dims=1)).type_as(psi2).mean(-1)
                    szsz_avg[l-1] = ( szsz_observable * psi2 / mcWeight).mean() / (psi2/mcWeight).mean()
            else:
                for l in range(1,nspin//2+1):
                    szsz_observable = (basis_input[:,:-1] * basis_input[:,1:]).type_as(psi2).mean(-1)
                    szsz_avg[l-1] = ( szsz_observable * psi2 / mcWeight).mean() / (psi2/mcWeight).mean()
            
            ## only for wall initial state
            mirror_szsz_observable = (basis_input[:,:nspin//2] *  basis_input[:,nspin//2:].flip(1))
            mirror_szsz_avg = (mirror_szsz_observable * psi2.unsqueeze(-1) / mcWeight.unsqueeze(-1)
                              ).mean(0) / (psi2/mcWeight).mean()
            
            sdw_avg = (basis_input * psi2.unsqueeze(-1) / mcWeight.unsqueeze(-1) 
                      ).mean(0) / (psi2/ mcWeight).mean()  ##(nspin)
            stagger_avg = sdw_avg[1::2].sum() - sdw_avg[0::2].sum()
            entropy=0
            return entropy, th2np(szsz_avg), th2np(mirror_szsz_avg), th2np(stagger_avg), th2np(sdw_avg)

    
    
    
    
    
    
    
    
    