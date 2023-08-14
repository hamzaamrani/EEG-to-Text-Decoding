import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config(case):
    if case == 'train_decoding': 
        # args config for training EEG-To-Text decoder
        parser = argparse.ArgumentParser(description='Specify config args for training EEG-To-Text decoder')
        
        parser.add_argument('-m', '--model_name', help='choose from {BrainTranslator, BrainTranslatorNaive}', default = "BrainTranslator" ,required=True)
        parser.add_argument('-t', '--task_name', help='choose from {task1,task1_task2, task1_task2_task3,task1_task2_taskNRv2}', default = "task1", required=True)
        
        parser.add_argument('-1step', '--one_step', dest='skip_step_one', action='store_true')
        parser.add_argument('-2step', '--two_step', dest='skip_step_one', action='store_false')

        parser.add_argument('-pre', '--pretrained', dest='use_random_init', action='store_false')
        parser.add_argument('-rand', '--rand_init', dest='use_random_init', action='store_true')
        
        parser.add_argument('-load1', '--load_step1_checkpoint', dest='load_step1_checkpoint', action='store_true')
        parser.add_argument('-no-load1', '--not_load_step1_checkpoint', dest='load_step1_checkpoint', action='store_false')

        parser.add_argument('-ne1', '--num_epoch_step1', type = int, help='num_epoch_step1', default = 20, required=True)
        parser.add_argument('-ne2', '--num_epoch_step2', type = int, help='num_epoch_step2', default = 30, required=True)
        parser.add_argument('-lr1', '--learning_rate_step1', type = float, help='learning_rate_step1', default = 0.00005, required=True)
        parser.add_argument('-lr2', '--learning_rate_step2', type = float, help='learning_rate_step2', default = 0.0000005, required=True)
        parser.add_argument('-b', '--batch_size', type = int, help='batch_size', default = 32, required=True)
        
        parser.add_argument('-s', '--save_path', help='checkpoint save path', default = './checkpoints/decoding', required=True)
        parser.add_argument('-subj', '--subjects', help='use all subjects or specify a particular one', default = 'ALL', required=False)
        parser.add_argument('-eeg', '--eeg_type', help='choose from {GD, FFD, TRT}', default = 'GD', required=False)
        parser.add_argument('-band', '--eeg_bands', nargs='+', help='specify freqency bands', default = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] , required=False)
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', default = 'cuda:0')
        
        args = vars(parser.parse_args())
        
    elif case == 'eval_decoding':
        # args config for evaluating EEG-To-Text decoder
        parser = argparse.ArgumentParser(description='Specify config args for evaluate EEG-To-Text decoder')
        parser.add_argument('-checkpoint', '--checkpoint_path', help='specify model checkpoint' ,required=True)
        parser.add_argument('-conf', '--config_path', help='specify training config json' ,required=True)
        parser.add_argument('-cuda', '--cuda', help='specify cuda device name, e.g. cuda:0, cuda:1, etc', default = 'cuda:0')
        args = vars(parser.parse_args())


    return args