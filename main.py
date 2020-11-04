import os
import logging
from datetime import datetime as dt
import pickle
from config import deepqd_config
from cgsp import *


def main(config_dict_path):
    ### Initialization
    args = deepqd_config()
    with open(config_dict_path,'rb') as f:
        args.__dict__ = pickle.load(f)
    ## overwrite device and number of device
    if th.cuda.is_available():
        args.device = 'cuda' if th.cuda.is_available() else 'cpu'
        args.num_device = th.cuda.device_count()
    else:
        args.device = 'cpu'
    log_dir = './log/' + args.experiment_name
    data_dir = './nn_data/' + args.experiment_name
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if args.solve_energy_bounds:
        logging.basicConfig(filename=log_dir + '/extreme.log',
                filemode='w',level=logging.DEBUG, format='%(levelname)-6s %(message)s')
        for k,v in args.__dict__.items():
            logging.info(" %s ->  %s"%(k,v))
        deepqd = spectral_tree(args)
        deepqd.extremal_state()
    else:
        net_id = args.train_node
        logging.basicConfig(filename=log_dir + '/train{}.log'.format(net_id),
            filemode='w',level=logging.DEBUG, format='%(levelname)-6s %(message)s')
        for k,v in args.__dict__.items():
            logging.info(" %s ->  %s"%(k,v))
        deepqd = spectral_tree(args)
        
        if args.train_node == '0':
            deepqd.init_tree()
        else:
            deepqd.load_tree()
            deepqd.train_node(net_id)

if __name__ == "__main__":
    config = deepqd_config()

    l = 32
    config.nspin = l
    config.cell_size = 4
    config.Jx = 1 / l;   config.Jy = 1 / l;   config.Jz = -1 / l
    config.energy_ub=0.45;   config.energy_lb=-0.25  #anti-ferromagnetic
    config.total_sz = 0
    config.hz = np.zeros(l)
    ## workflow
    config.solve_energy_bounds = False
    config.sampling_mode = 'MC'
    train_layer = 0
    ## training
    config.hierarchy_groups = [32]  ## M
    config.nbands = [32]            ## N
    config.hidden = 128
    config.phase_hidden = config.hidden
    config.train_config["batchsize_begin"] = 4000
    config.train_config["batchsize_end"] = 4000
    config.train_config["lr_begin"] = 1e-3
    suffix = 'wall-test'
    config.experiment_name = 'l{}M{}N{}{}B{}'.format(
        l, 
        '-'.join([str(x) for x in config.hierarchy_groups]),
        '-'.join([str(x) for x in config.nbands]),
        config.sampling_mode, config.train_config["batchsize_begin"]) + suffix

    ###########################
    log_dir = './log/'
    data_dir = './nn_data/'
    config_dict_dir = './config/'
    if not os.path.exists(config_dict_dir):
        os.mkdir(config_dict_dir)
        print("Directory " , config_dict_dir ,  " Created ")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        print("Directory " , log_dir ,  " Created ")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        print("Directory " , data_dir ,  " Created ")
    ###########################
    if train_layer == 0:
        config.train_node = '0'
        config_dict_path = config_dict_dir  + dt.now().strftime(
                "%y-%m-%d-%H-%M-%S") + config.experiment_name + 'NN0'
        with open(config_dict_path,'wb') as f:
            pickle.dump( config.__dict__, f)
        main(config_dict_path)
    elif train_layer == 1:
        config.device = 'cuda' if th.cuda.is_available() else 'cpu'
        config.num_device = th.cuda.device_count()
        deepqd = spectral_tree(config)
        deepqd.load_model('0')
        child_id_list = deepqd.generate_children_from(deepqd.root)
        deepqd.save_tree(deepqd.root)
        for node in child_id_list:
            config.train_node = node
            config_dict_path = config_dict_dir + '/' + dt.now().strftime(
                "%y-%m-%d-%H-%M-%S") + config.experiment_name + node
            with open(config_dict_path,'wb') as f:
                pickle.dump( config.__dict__, f)
            main(config_dict_path)
    else:
        raise NotImplementedError
    