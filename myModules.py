import numpy as np
import math
from torch import nn
from torch.nn import Module, Parameter, init, Sequential
from torch.nn import Conv1d, Conv2d, Linear
from myFunctionals import *
from util import batched_vec_kron

class SimulatedComplexLinear(Module):
    '''
    input : (groups, batch, in_features)
    output: (groups, batch, out_features)
    '''
    __constants__ = ['bias', 'in_features', 'out_features', 'groups']
    def __init__(self, in_features, out_features, groups = 1, bias=True):
        super(SimulatedComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.groups = groups
        self.weight = Parameter(th.Tensor(groups, out_features, in_features))
        if bias:
            self.bias = Parameter(th.Tensor(groups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("dim(input) has to be 3") 
        if x.shape[0] != self.groups or x.shape[2] != self.in_features:
            raise ValueError("Input shape must be (groups, batch, in_features)") 
        output = th.bmm(x, self.weight.transpose(1,2))
        if self.bias is not None:
            output += self.bias[:,None,:]
        return output
        # output_r = output[:,:,:self.out_features]
        # output_i = output[:,:,self.out_features:]
        # return output_r, output_i

    def extra_repr(self):
        return 'in_features={}, out_features={}, groups={}, bias={}'.format(
            self.in_features, self.out_features, self.groups, self.bias is not None
        )

class CausalConv1d(Module):
    """1D DILATED CAUSAL CONVOLUTION."""
    def __init__(self, in_channels, out_channels, 
                 kernel_size, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.dilation = dilation
        self.groups = groups
        self.padding = padding = (kernel_size - 1) * dilation
        self.conv = Conv1d(in_channels*groups, out_channels*groups, kernel_size,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Float tensor variable with the shape  (groups, batch, in_channel, height).
        Returns:
            Tensor: Float tensor variable with the shape (groups, batch, out_channel, height)
        """
        if x.dim() != 4:
            raise NotImplementedError("The total dimension of input should be 4, but get {}".format(x.dim()))
        groups, nbatch, in_ch, in_height = x.shape
        if groups == self.groups:
            x = x.transpose(0,1).reshape(nbatch, groups*in_ch,in_height)
        else:
            raise ValueError("input shape is illegal")
        x = self.conv(x)
        x = x.reshape(nbatch,groups,self.out_channels,-1).transpose(0,1)
        if self.padding != 0:
            x = x[:, :, :, :-self.padding]
        assert x.shape[-1] == in_height
        return x
    
class NonCausalConv1d(Module):
    """1D DILATED CAUSAL CONVOLUTION."""
    def __init__(self, in_channels, out_channels, 
                dilation=1, groups=1, bias=True):
        super(NonCausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 3
        self.stride = 1
        self.dilation = dilation
        self.groups = groups
        self.conv = Conv1d(in_channels*groups, out_channels*groups, self.kernel_size,
                              padding=self.dilation, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Float tensor variable with the shape  (groups, batch, in_channel, height).
        Returns:
            Tensor: Float tensor variable with the shape (groups, batch, out_channel, height)
        """
        if x.dim() != 4:
            raise NotImplementedError("The total dimension of input should be 4, but get {}".format(x.dim()))
        groups, nbatch, in_ch, in_height = x.shape
        if groups == self.groups:
            x = x.transpose(0,1).reshape(nbatch, groups*in_ch,in_height)
        else:
            raise ValueError("input shape is illegal")
        x = self.conv(x)
        x = x.reshape(nbatch,groups,self.out_channels,in_height).transpose(0,1)
        return x

##======================================###
    
class amp_net_mixed(Module):
    def __init__(self, ncell=10, cellsize=1, groups=1, n_preproc=2, n_resch=64, n_skipch=2,
                 dilations = None, kernel_size=2):
        super(amp_net_mixed, self).__init__()
        self.ncell = ncell
        self.ngroup = groups
        self.n_preproc = n_preproc
        self.n_resch = n_resch        ## residue channel
        self.n_skipch = n_skipch
        self.kernel_size = kernel_size
        self.dilations = dilations
        ## first layer
        self.causal_in = CausalConv1d(in_channels=n_preproc*cellsize, out_channels=n_resch, 
                 kernel_size=kernel_size, dilation=1,  bias=True)
        ## convolution layers
        self.dil_act = nn.ModuleList()
        self.skip_1x1 = nn.ModuleList()

        for d in self.dilations:
            self.dil_act += [CausalConv1d(n_resch, n_resch, kernel_size,
                                                    dilation=d, bias=True)]
            self.skip_1x1 += [CausalConv1d(n_resch, self.ngroup * n_skipch, kernel_size=1,
                                                    dilation=d, bias=True)]
        #self.skip_2x2 = CausalConv1d(n_resch, self.ngroup * n_skipch, kernel_size=2,
        #                                        dilation=1, bias = True )
        #self.local_linear = SimulatedComplexLinear(self.ngroup*n_skipch, self.ngroup*n_skipch, groups = ncell, bias = True)
        
        self.skip_conv = CausalConv1d(n_resch*len(self.dilations), n_resch*len(self.dilations), kernel_size=1,
                                                dilation=1, bias = True)
        self.local_linear = SimulatedComplexLinear(n_resch*len(self.dilations), self.ngroup*n_skipch*4, groups = ncell, bias = True)
        self.final_linear = SimulatedComplexLinear(n_skipch*4, n_skipch, groups = ncell*self.ngroup, bias = True)

    def seq_forward(self, x):
        """FORWARD CALCULATION - amplitude part. no skip connection. uncomment self.skip_2x2 to enable.
        Args:
            configuration (Tensor): tensor with the shape (1,batch, channel, ncell).
            the configuration is already reordered if pbc is true
        Returns:
            output(Tensor): wave function amplitude with the shape (groups, batch, n_skipth, ncell) .
        """
        lrelu = nn.LeakyReLU(0.1)
        tanhsh = nn.Tanhshrink()
        x = self.causal_in(x)
        x = tanhsh(x)
        for i, dil_ly in enumerate(self.dil_act):
            x = dil_ly(x)
            x = lrelu(x)
        x = self.skip_2x2(x) / self.n_resch  #(1, batch, self.ngroup * nskipth, ncell)

        x = x.permute(0,-1,1,2).reshape(self.ncell,-1, self.ngroup * self.n_skipch)
        x = self.local_linear(x)
        x = x.reshape(self.ncell, -1, self.ngroup, self.n_skipch).permute(2,1,3,0)
        return x
    def skipadd_forward(self, x):
        """FORWARD CALCULATION - amplitude part. skip connection through adding
        Args:
            configuration (Tensor): tensor with the shape (1,batch, channel, ncell).
            the configuration is already reordered if pbc is true
        Returns:
            output(Tensor): wave function amplitude with the shape (groups, batch, n_skipth, ncell) .
        """
        lrelu = nn.LeakyReLU(0.1)
        tanhsh = nn.Tanhshrink()
        x = self.causal_in(x)
        x = tanhsh(x)
        skip = 0
        for i, dil_ly in enumerate(self.dil_act):
            x = dil_ly(x)
            x = lrelu(x)
            skip += self.skip_1x1[i](x)  / self.n_resch #(1, batch, self.ngroup * nskipth, ncell)
        skip = lrelu(skip)
        skip = skip.permute(0,-1,1,2).reshape(self.ncell,-1, self.ngroup * self.n_skipch)
        skip = self.local_linear(skip)
        skip = skip.reshape(self.ncell, -1, self.ngroup, self.n_skipch).permute(2,1,3,0)
        return skip
    def forward(self, x):
        """FORWARD CALCULATION - amplitude part. direct skip connection
        Args:
            configuration (Tensor): tensor with the shape (1,batch, channel, ncell).
            the configuration is already reordered if pbc is true
        Returns:
            output(Tensor): wave function amplitude with the shape (groups, batch, n_skipth, ncell) .
        """
        nbatch = x.shape[1]
        lrelu = nn.LeakyReLU(0.1)
        tanhsh = nn.Tanhshrink()
        x = self.causal_in(x)
        x = tanhsh(x)
        skip = []
        for i, dil_ly in enumerate(self.dil_act):
            x = dil_ly(x)
            x = lrelu(x) # #(1, batch, n_resch, ncell)
            skip.append(x.clone())
        skip = th.cat(skip, 2) #(1, batch, n_resch*layers, ncell)
        skip = self.skip_conv(skip) #(1, batch, n_resch, ncell)
        skip= lrelu(skip)
        skip = skip.squeeze(0).permute(-1,0,1) #(ncell,batch, n_resch)
        skip = self.local_linear(skip) #(ncell,batch, ngroup*n_skipch*2)
        skip = lrelu(skip)
        
        skip = skip.reshape(self.ncell, nbatch, self.ngroup, -1).permute(0,2,1,3).reshape(self.ncell*self.ngroup,nbatch, -1)
        skip = self.final_linear(skip) # (ncell*ngroup,batch, n_skipch)
        skip = skip.reshape(self.ncell, self.ngroup,-1,self.n_skipch).permute(1,2,3,0)
        return skip

class amp_net_unentangled(Module):
    def __init__(self, amp_reordered):
        super(amp_net_unentangled, self).__init__()
        self.register_buffer('amp', amp_reordered)  ## (nspin,2)
    def forward(self, inputs):
        '''
        Args:
            inputs(Tensor): Tensor with shape (1,batch, cellsize, ncell)
        Returns:
            amp_kroned(Tensor): tensor with shape (groups, nbatch, 2^cellsize, ncell)
        '''
        groups, nbatch, cellsize, ncell = inputs.shape
        nspin = self.amp.shape[0]
        amp_expanded = self.amp.expand(groups, nbatch, nspin, 2).reshape(
            groups, nbatch, ncell, cellsize, 2)
        amp_kroned = batched_vec_kron(amp_expanded) # (groups, nbatch, ncell, 2^cellsize)
        return amp_kroned.transpose(-1,-2)

class phase_cnn_mixed(Module):
    '''
    inputs of phase net is always in its natural order regradless of pbc, because phase_net is independent of generating process
    !Not tested!
    '''
    def __init__(self, groups, ncell, cellsize, hidden_channel=64):
        super(phase_cnn_mixed, self).__init__()
        ## phase net
        self.groups = groups
        self.ncell = ncell   ## the real ncell + 2 * pbc
        self.cellsize = cellsize
        self.hd_ch = hidden_channel
        kernal_size = 3
        conv_layers = int(np.ceil(np.log(ncell) / np.log(kernal_size)))
        conv_channels = [cellsize] + [hidden_channel] * conv_layers
        self.conv_list = nn.ModuleList([
            NonCausalConv1d(in_channels=conv_channels[i], out_channels=conv_channels[i+1], 
                dilation=kernal_size**i, groups=1, bias=True) for i in range(conv_layers)
        ])
        self.linear = nn.Linear(ncell*hidden_channel, self.groups)
    def forward(self,x):
        """FORWARD CALCULATION - phase part.
        Args:
            configuration (Tensor): tensor with the shape (1,batch, channel=cellsize, height=ncell).
        Returns:
            output(Tensor): wave function phase with the shape (groups, batch) .
        """
        _, nbatch, cellsize, ncell = x.shape
        tanhsh = nn.Tanhshrink()
        lrelu = nn.LeakyReLU(0.1)
        for conv in self.conv_list:
            x = conv(x)
            x = lrelu(x) ## (1, nbatch, hidden_channel, ncell)
        x = self.linear(x.reshape(nbatch, -1)) ## (nbatch, groups)
        # x = (th.tanh(x)+1)/2.0 * np.pi
        # x = (x - th.sin(x))/2
        return x.transpose(0,1)
