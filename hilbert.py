import numpy as np
import torch as th
from sympy.utilities.iterables import multiset_permutations
from math import factorial
from util import combination, integer2bit, bit2integer, th2np

class heisenberg:
    def __init__(self,model_params):
        self.dim = model_params.get('dim')
        self.device = model_params.get('device')
        self.dtype = model_params.get('dtype')
        self.Lx = model_params.get('Lx')
        self.Jx = model_params.get('Jx')
        self.Jz = model_params.get('Jz')
        if self.dim == 1:
            self.nspin = self.Lx
        elif self.dim == 2:
            self.Ly = model_params.get('Ly')
            self.Jy = model_params.get('Jy')
            self.nspin = self.Lx * self.Ly
        else:
            raise NotImplementedError
        try:
            hz = model_params.get('hz')
            self.hz = hz if th.is_tensor(hz) else th.tensor(hz)
            self.hz = self.hz.to(device=self.device,dtype=self.dtype)
        except:
            self.hz = th.zeros(self.nspin, device = self.device, dtype = self.dtype)
            
        ### conservation
        if model_params.get('conserve') is None:
            self.sz_conserve = None
            self.dim_hilbert = 2**self.nspin
        elif model_params.get('conserve') == 'Sz':
            self.sz_conserve = model_params.get('sz_conserve')
            if (self.nspin + self.sz_conserve) % 2 == 0:
                self.num_spinup = (self.nspin + self.sz_conserve) // 2
                self.num_spindown = self.nspin - self.num_spinup
            else:
                raise ValueError('sz_conserve imcompatible with number of spins')
            self.dim_hilbert = int(factorial(self.nspin) / factorial(self.num_spinup)/ factorial(self.num_spindown))
        else:
            raise NotImplementedError
        
        ### boundary condition
        self.pbc_x = model_params.get('pbc_x')
        if self.dim == 2:
            self.pbc_y = model_params.get('pbc_y')
        
    def get_full_basis(self):
        assert (self.nspin <= 20), "basis too large, use rand_basis instead"
        ref_state = np.concatenate([np.ones(self.num_spinup), -np.ones(self.num_spindown)]).astype(int)
        np_output = np.array(list(multiset_permutations(ref_state)))
        return th.tensor(np_output,dtype=self.dtype,device = self.device)
    
    def step(self, state): ## pbc only
        exchange = state*th.roll(state,shifts=-1,dims=1)
        allows_exchange= (1-exchange) // 2
        exchange_idx = th.multinomial(allows_exchange.type(th.float),1).flatten()
        new_state = state.clone()
        nbatch, nspin = new_state.shape
        new_state[th.arange(nbatch),exchange_idx] *= -1
        new_state[th.arange(nbatch), (exchange_idx+1)%nspin] *= -1
        return new_state
    
    def get_backbone(self, reference_state): ## 1D only, pbc only
        '''
        get configuration near the reference state for pretraining
        '''
        s_nbhood, w_nbhood = self.get_neighbor_list_parallel(reference_state.reshape(1,self.nspin))
        nearest = s_nbhood[0][th.abs(w_nbhood[0])>1e-8].clone()
        #s_nbhood, w_nbhood = self.get_neighbor_list_parallel(nearest)
        #second_nearest = s_nbhood[0][th.abs(w_nbhood[0])>1e-8].clone()
        reps = self.nspin**2 // 8
        n_walker = 4000 // reps 
        backbone = [nearest.type_as(reference_state),reference_state.reshape(1,self.nspin).repeat(n_walker, 1)]
        for i in range(reps):
            backbone.append(self.step(backbone[-1]))
        backbone = th.cat(backbone,0)
        return th.unique(backbone,sorted=False,dim=0)
    
    def get_full_H(self):
        full_basis = self.get_full_basis()
        nbatch = full_basis.shape[0]
        basis_label = bit2integer((full_basis+1)//2,self.nspin).flatten()
        label2idx = th.zeros(basis_label.max()+1,dtype=th.long,device = self.device)
        label2idx[basis_label] = th.arange(nbatch,device = self.device)        
        full_H = th.zeros(nbatch, nbatch,device = self.device)
        s_nbhood, w_nbhood = self.get_neighbor_list_parallel(full_basis)
        nnbs = s_nbhood.shape[1]
        s_nbhood = s_nbhood.reshape(-1,self.nspin)
        w_nbhood = w_nbhood.reshape(-1)
        
        left_idx = th.repeat_interleave(th.arange(nbatch,device = self.device),nnbs)
        right_label = bit2integer( (s_nbhood+1)//2, self.nspin ).flatten()
        right_idx = label2idx[right_label]
        
        adjacency_mask = (th.abs(w_nbhood) > 1e-8)
        full_H[left_idx[adjacency_mask],right_idx[adjacency_mask]] += w_nbhood[adjacency_mask]
        return full_H, label2idx
    
    def get_exact_observation(self, T, reference_state):
        full_H, label2idx = self.get_full_H()
        e,v = th.symeig(full_H,eigenvectors=True)
        v = v.transpose(0,1)
        full_basis = self.get_full_basis()
        
        init_state = reference_state.to(device = full_basis.device)
        init_label = bit2integer((init_state.reshape(-1,self.nspin)+1)//2,self.nspin).flatten()
        init_idx = label2idx[init_label]
        a = v[:, init_idx].flatten() # (nbatch)
        print('norm(a)',(a**2).sum())
        time = th.arange(101,dtype=th.float) / 100.0 * T
        sdw_avg = th.zeros(time.shape[0], self.nspin).to(device=full_basis.device)
        szsz_avg = th.zeros(time.shape[0], self.nspin//2).to(device=full_basis.device)
        # szsz_avg = th.zeros_like(time)
        stagger_avg, entropy =  th.zeros_like(time),th.zeros_like(time)
        for i, t in enumerate(time):
            psi_r = (a * th.cos(-e*t)).unsqueeze(-1) * v
            psi_i = (a * th.sin(-e*t)).unsqueeze(-1) * v
            
            psi2 = (psi_r.sum(0))**2+(psi_i.sum(0))**2
            if self.pbc_x:
                for l in range(1,self.nspin//2+1):
                    szsz_observable= (full_basis * full_basis.roll(l,dims=1)).type_as(psi2).mean(-1)
                    szsz_avg[i,l-1] = ( szsz_observable * psi2 ).sum()
                # szsz_observable = (full_basis * full_basis.roll(1,dims=1)).type_as(psi2).mean(-1)
                # szsz_avg[i] = ( szsz_observable * psi2 ).sum()
            else:
                raise NotImplementedError
                # szsz_observable = (full_basis[:,:-1] * full_basis[:,1:]).type_as(psi2).mean(-1)
            sdw = (full_basis * psi2.unsqueeze(-1)).sum(0)  ##(nspin)
            sdw_avg[i] =  sdw   
            stagger_avg[i] = sdw[1::2].sum() - sdw[0::2].sum()
        return th2np(time), th2np(entropy), th2np(szsz_avg), th2np(stagger_avg), th2np(sdw_avg)

    def get_neighbor_list_parallel(self, configuration):
        '''
        Args:
        configuration: tensor of shape (nbatch, nspin)
        Returns:
        s_nbhood: tensor of shape (nbatch, nneighbors, nspin)
        w_nbhood: tensor of shape (nbatch, nneighbors)
        '''
        if len(configuration.shape) != 2:
            raise ValueError("illegal configuration shape")
        s = configuration.clone() if th.is_tensor(configuration) else th.tensor(configuration)
        s = s.to(dtype = self.dtype, device= self.device)
        nbatch, nspin = s.shape
        if self.dim == 1:
            nneighbor = nspin + int(self.pbc_x)
            s_nbhood = s.unsqueeze(1).repeat(1,nneighbor,1)
            flag_kept = th.ones_like(s_nbhood[:,:,0])
            z = 1 - int(self.pbc_x)
            for i in range(1,nneighbor):
                s_nbhood[:,i,i-2+z] = s[:,i-1+z].clone()
                s_nbhood[:,i,i-1+z] = s[:,i-2+z].clone()
                flag_kept[:,i] = (1.0 - (s[:,i-2+z] * s[:,i-1+z]))/2.0
            nearest_neighbor_weight = 1.0 / 2.0 * self.Jx
            w_nbhood = flag_kept * nearest_neighbor_weight
            w_nbhood[:, 0] = (self.hz[None,:] * s).sum(1) + ((s[:,:-1]*s[:,1:]).sum(1) + s[:,0]*s[:,-1]) * self.Jz / 4.0  ## edge weight of self-loop
            return s_nbhood, w_nbhood
        else:
            raise NotImplementedError

            
class hilbert_tfi1d:
    '''
        !Not tested!
        Transverse-field ising model: H = -Jz*sz*sz - hx*sx - hz*sz  (s_i is the pauli matrix with eigenvalue +-1)
    '''
    ## 
    def __init__(self, nspin, hx, hz, Jz):
        self.name = 'tfi1d'
        self.nspin = nspin
        self.hx = hx
        self.hz = hz
        self.Jz = Jz
        self.dtype = hz.dtype
        self.device = hz.device
        self.dim_hilbert = 2**nspin

    def get_full_basis(self):
        '''
        Returns:
            full sz-basis(Tensor, th.int64): sz-basis in the (-1,1) representation, tensor with shape (2**nspin, nspin)
        '''
        assert (self.nspin < 20), "basis too large, use rand_basis instead"
        basis = integer2bit(th.arange(self.dim_hilbert), self.nspin).to(dtype=self.dtype, device=self.device)
        return 2 * basis - 1   ##(0,1) representation to (-1,1) representation

    def rand_basis(self, size = 50, replace = False):
        '''
        Args:
            size: number of random basis generated
            replace: samples are drawn with replacement
        Returns:
            random sz-basis(Tensor, th.int64): sz-basis in the (-1,1) representation, tensor with shape (size, nspin)
        '''
        if replace==False and size>self.dim_hilbert:
            raise ValueError("required basis size is larger than the total number of hilbert basis")
        basis_binary = th.multinomial(th.ones(self.dim_hilbert),size,replace)
        basis = integer2bit(basis_binary, self.nspin).to(dtype=self.dtype, device=self.device)
        return 2 * basis - 1
    ###########  opreration on sz basis ##########
    def step(self,states):
        ## TODO one step forward under random walk rule
        pass
    
    def edge_weight(self, s1, s2): # the batched edge weight between sites s1 & s2 
        '''
        Args:
            s1;s2( th.tensor(th.int)): batched basis of size (nbatch, nspin)
        Returns:
            edge weight( th.tensor): sz-basis in the (-1,1) representation, tensor with shape (size, nspin)
        '''
        ds = (s1 != s2).to(dtype=th.int64)
        ## adjacent sites contribution
        idx_aj = (ds >0).sum(1) == 1
        w_aj = - self.hx 
        ## same sites contribution
        idx_same = (ds.sum(1) == 0)
        w_same = - (self.hz[None,:]*s1).sum(1) - (s1[:,:-1]*s1[:,1:]).sum(1) * self.Jz
        return idx_aj.type_as(w_same) * w_aj + idx_same.type_as(w_same) * w_same


    def get_neighbor_list_parallel(self, configuration):
        '''
        Args:
            configuration(th.tensor(th.int)): batched basis of size (nbatch, nspin)
        Returns:
            neighbor_list, edge weight(np.array or th.tensor): sz-basis in the (-1,1) representation, tensor with shape (size, nspin)
        '''
        if len(configuration.shape) != 2 or configuration.shape[1]!=self.nspin:
            raise ValueError("illegal configuration shape")
        assert th.is_tensor(configuration) and th.is_tensor(self.hz)
        s = configuration
        nbatch, nnbs = s.shape[0], self.nspin+1

        s_nbhood = s.unsqueeze(-1).repeat(1,1,nnbs)
        w_nbhood = - self.hx * th.ones((nbatch,nnbs),device=s.device, dtype=self.hz.dtype)
        for i in range(self.nspin):
            s_nbhood[:,i,i+1] *= -1
        w_nbhood[:,0] = self.edge_weight(s,s)
        return s_nbhood, w_nbhood

    
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import pickle
    L = 16
    Jt = 6
    model_paras = {
            'dim': 1,
            'device' : 'cpu',
            'dtype' : th.float,
            'Lx' : L,
            'Jx' : 1/L,
            'Jz' : -1/L,
            'hz' : th.zeros(L),
            'conserve':'Sz',
            'sz_conserve': 0,
            'pbc_x': True,
        }
    ham = heisenberg(model_paras)
    neel = th.tensor([-1,1]*(L//2))
    wall = th.tensor([-1]*(L//2)+[1]*(L//2))
    time, entropy, szsz_avg, stagger_avg, sdw_avg = ham.get_exact_observation(Jt*L, wall)
    with open('./benchmark/wallJz-1L{}.npy'.format(L),"wb") as file:
        pickle.dump(time/L, file)
        pickle.dump(entropy, file)
        pickle.dump(szsz_avg, file)
        pickle.dump(stagger_avg, file)
        pickle.dump(sdw_avg, file)
    for sdw in sdw_avg.transpose():
        plt.plot(time/L, sdw)
    plt.savefig('./benchmark/wallJz-1L{}sdw.png'.format(L),dpi=150)#.format(L))
    plt.close()
    plt.figure()
    for idx,szsz in enumerate(szsz_avg.transpose()):
        plt.plot(time/L, szsz,label='id={}'.format(idx)) 
    plt.legend()
    plt.savefig('./benchmark/wallJz-1L{}szsz.png'.format(L),dpi=150)