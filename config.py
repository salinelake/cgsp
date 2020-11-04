import numpy as np
import torch as th
class deepqd_config:
    def __init__(self, nnets=30, energy_ub=1, energy_lb=-1):
        #================system config=============#
        self.model_name = 'xxz1d' 
        self.nspin = None
        self.model_dim = 1
        self.Jx = None  ##  XXZ model
        self.Jy = None  ##  XXZ model
        self.Jz = None
        self.hz = None
        self.pbc_x = True
        self.total_sz = 0
        # self.neel_peak = np.tile(np.array([-1,1]), self.num_spin//2)
        ##================ energy bands config =============#
        self.energy_ub =  None
        self.energy_lb =  None
        #================neural net config=============#
        self.net_type = 'ensemble'   # 'sequential'
        # self.device = 'cpu'
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.num_device = 1
        self.cell_size = 1
        self.dtype = th.float32
        self.hidden = 128
        self.phase_hidden = 128
        #================ workflow config =============#
        self.vanilla_mode = False
        self.solve_energy_bounds = False
        self.sampling_mode = 'MC'  ##  'MC' or 'full'
        self.load_model = False
        self.model_path = None
        self.train = True
        self.observe_SDW = False
        self.hierarchy = 2
        self.hierarchy_groups = [10,9]
        self.nbands  = None

        #================training and testing config =============# 
        self.train_config = {
            "epoches":1000000,
            "start_epoch":0,
            # batchsize
            "batchsize_begin":128,
            "batchsize_end":512,
            ## optimizer
            #"pretrain_lr":3e-2,
            "lr_begin":0.01,
            "lr_scale_fac":0.96,
            "lr_end":3e-4,
            ## display and save
            "freq_display" : 10,
            "freq_save": 1000,
            "freq_test": 1000,
        }