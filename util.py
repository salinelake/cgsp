import torch as th
import numpy as np
from math import factorial

def th2np(x, digits = None):
    ## convert torch tensor to numpy array
    np_x = x.cpu().detach().numpy()
    result = np_x if digits == None else np.round(np_x, digits)
    return result

def sorted_eig(m):
    E_otged, W_otged = np.linalg.eig(m)
    E_otged = E_otged.real
    sort_perm = E_otged.argsort()
    E_otged.sort()     # <-- This sorts the list in place.
    W_otged = W_otged[:, sort_perm]
    return E_otged, W_otged

def integer2bit(integer, num_bits=4):
    """Turn integer tensor to binary representation.
        Args:
            integer : torch.Tensor, tensor with integers
            num_bits : Number of bits to specify the precision. Default: 8.
        Returns:
            Tensor: Binary tensor. Adds last dimension to original tensor for
            bits.
    """
    result = [((integer.unsqueeze(-1)) >> (num_bits-1-i)
                        )%2 for i in range(num_bits)
                        ]
    return th.cat(result,-1)

def bit2integer(bits, num_bits):
    exponent_mask =  num_bits - 1.0 - th.arange(num_bits,dtype=bits.dtype, device=bits.device)
    exponent_mask = 2**exponent_mask[None,None,None,:]
    select_idx = (bits * exponent_mask).sum(-1)
    return select_idx.type(th.long)

def vonNeumannEntropy(basis_input, vector_state):
        '''
        Calculate the von-neuman entropy between equal bipartite from a vectorized eigenstate
        Args:
            basis_input: numpy 2d-array (L, self.num_spin)
                            will be converted to binary representation in the !reverse! order 
            np_psi: complex numpy vector (L)
        Returns:
            np.float
        '''
        assert len(basis_input.shape) == 2
        assert basis_input.shape[0] == vector_state.shape[0]
        num_spin = basis_input.shape[1]
        sys_spin = int(num_spin / 2)
        env_spin = int(num_spin - sys_spin)
        sys_dim = 2 ** sys_spin
        env_dim = 2 ** env_spin
        tensor_state = np.zeros((sys_dim,env_dim),dtype=np.complex128)
        for idx, basis in enumerate(basis_input):
            sys_conf_bin = ((basis[0::2]+1)/2).astype(int)
            env_conf_bin = ((basis[1::2]+1)/2).astype(int)
            sys_conf_int = (sys_conf_bin * 2**np.arange(sys_spin)).sum()
            env_conf_int = (env_conf_bin * 2**np.arange(env_spin)).sum()
            tensor_state[int(sys_conf_int), int(env_conf_int)] = vector_state[idx]
        singular_value = np.linalg.svd(tensor_state, compute_uv=False)
        plogp = - singular_value**2 * np.log(singular_value**2) / np.log(2)
        return plogp.sum()

def batched_vec_kron(x):
    vec_dimension = x.shape[-1]
    power = x.shape[-2]
    y = x.reshape(-1, power, vec_dimension)
    result = y[:,0,:].clone()
    for i in range(1,power):
        result = th.matmul(
            result.unsqueeze(-1), 
            y[:,i,:].unsqueeze(-2)
            ).reshape(-1, vec_dimension**(i+1))
    output_shape = x.shape[:-2] + th.ones(vec_dimension**power).shape
    return result.reshape(output_shape)

def logcosh(x):
    return th.log(th.cosh(x))

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))

def deg_fn(position):
        original_shape = position.shape
        if th.is_tensor(position):
            p = position
        else:
            p = th.tensor(position)
        p = p.reshape(-1,original_shape[-1])
        deg = (p[:,:-1] != p[:,1:]).sum(-1)
        return deg.reshape(original_shape[:-1]).to(dtype = th.float)

def logdeg_fn(position):
        deg = deg_fn(position)
        return th.log(deg)

def reassign_device_dtype(device, dtype):
    def reassign(torch_obj):
        return torch_obj.to(device = device, dtype = dtype)
    return reassign


def lr_scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def adam_scheduler(optimizer, lr, betas=(0.9,0.95)):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = betas
    return optimizer

def myAdam(net, lr, betas=(0.9,0.9)):
    optimizer = th.optim.Adam(
                [{'params': fc.parameters()} for fc in net.fc_lys]
                +[{'params': net.log_rescale_r, 'lr':lr * 10},
                  {'params': net.log_rescale_i, 'lr':lr * 10}]
                , lr = lr, betas=betas )
    return optimizer

def sum_inverse_sqr(N, idx):
    ans = 0
    for i in range(N):
        if i != idx:
            ans += 1/(2*np.abs(i-idx))**2
    return ans

def combination(N,m):
    assert m<=N, "N<m"
    return factorial(N)/ factorial(m) / factorial(N-m)
