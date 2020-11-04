import torch.nn.functional as F
from torch.nn.functional import max_pool2d, dropout, dropout2d
import torch as th
import math

def complex_mul(xr,xi, yr,yi):
    zr = xr * yr - xi * yi
    zi = xr * yi + xi * yr
    return zr, zi

def gelu(x):
    return 0.5 * x * (1 + th.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))

def complex_exp(input_r, input_i):
    return th.exp(input_r) * th.cos(input_i), th.exp(input_r) * th.sin(input_i)

def complex_tanh(input_r, input_i):
    denom = th.exp(2*input_r) + th.exp(-2*input_r) + 2 * th.cos(2*input_i)
    nom_r = th.exp(2*input_r) - th.exp(-2*input_r)
    nom_i = 2 * th.sin(2 * input_i)
    return nom_r/denom, nom_i/denom

def complex_sinh(input_r, input_i):
    ep_r, ep_i = complex_exp(input_r, input_i)
    em_r, em_i = complex_exp(-input_r, -input_i)
    return (ep_r - em_r)/2, (ep_i - em_i)/2

def complex_cosh(input_r, input_i):
    ep_r, ep_i = complex_exp(input_r, input_i)
    em_r, em_i = complex_exp(-input_r, -input_i)
    return (ep_r + em_r)/2, (ep_i + em_i)/2

def complex_logcosh(input_r, input_i):
    pass

def complex_Tanhshrink(input_r, input_i):
    return input_r - th.tanh(input_r), input_i - th.tanh(input_i)

def complex_naive_relu(input_r,input_i):
    return th.relu(input_r), th.relu(input_i)

def complex_naive_tanh(input_r,input_i):
    return th.tanh(input_r), th.tanh(input_i)

def conv1d_local(x, weight, bias = None, stride =1 , padding = 0, dilation=1, groups = 1):
    '''
    Args:
        x(Tensor): Float tensor with shape (groups, batch, in_channel, height).
        weight(Tensor): Float tensor with shape (groups, out_height, out_channel, in_channel, kernal_height)
        bias(Tensor): Float tensor with shape (groups, out_channel, out_height)
    Returns:
        out(Tensor): Float tensor with shape (groups, batch, out_channel, outH) 
    '''
    if x.dim() !=4:
        raise NotImplementedError("Input Error: Only 4D input Tensors supported (got {}D)".format(x.dim()))
    if weight.dim() != 5:
        # groups x outH x outC x inC x kH
        raise NotImplementedError("Input Error: Only 5D weight Tensors supported (got {}D)".format(weight.dim()))
    nbatch = x.shape[1]
    inH = x.shape[-1]
    _, outH, outC, inC, kH = weight.shape

    ## cols =>(groups*batch, in_channel*k_height, out_H) , th.nn.Unfold is only implemented for 4D tensor. 
    cols = F.unfold(
        x.reshape(-1, inC, inH,1),
        kernel_size=(kH,1), 
        dilation=(dilation,1), padding=(padding,0), stride=(stride,1))
    ## cols ==> (batch, groups, outH, 1, in_channel*k_height)
    cols = cols.reshape(groups, nbatch, inC*kH, outH, 1).permute(1,0,3,4,2)
    ## out => (batch, groups, outH, 1, out_channel)
    out = th.matmul(
        cols, 
        weight.view(groups, outH, outC, inC * kH).transpose(2,3), ## weight => (groups, outH, in_channel * k_height, out_channel)
        )   
    out = out.reshape(nbatch, groups, outH, outC)
    out = out.permute(1, 0, 3, 2)  ## out => (groups, batch, out_channel, outH)
    if bias is not None:
        out = out + bias.unsqueeze(1)
    return out