import argparse
import torch
from trainer import train

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using {} device'.format(device))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CIFAR10',type=str, help=" CIFAR10 CIFAR100 ImageNet MiniImagenet SVHN.")
    parser.add_argument('--data_path', default='/root/workspace/datasets', type=str, help="Path to data directory.")
    parser.add_argument('--noise_rate', default=0.0, type=float, help="Noise rate.")
    parser.add_argument('--noise_mode', default='sym', type=str, help="Noise mode.")
    parser.add_argument('--num_workers', default=0, type=int, help="Num of workers.")
    parser.add_argument('--outputs_dim', default=10, type=int, help="Dimension of outputs.")
    parser.add_argument('--inputs_dim', default=32, type=int, help="Size of inputs' features.")
    parser.add_argument('--inputs_channel', default=3, type=int, help="Num of channels of inputs.")
    parser.add_argument('--depth', default=16, type=int, help="Depth of net.")
    parser.add_argument('--network', default='vgg16_bn', type =str)
    parser.add_argument('--mlp_ratio',default=4.0,type=float)
    parser.add_argument('--patch_size',default=16,type=int)
    parser.add_argument('--train_batch_size', default=128, type=int, help="Train Batch size")
    parser.add_argument('--test_batch_size', default=128, type=int, help="Test Batch size")
    parser.add_argument('--epoch_range', default=300, type=int, help="Num of Epoch")
    parser.add_argument('--device', default='cuda', type=str, help="CUDA or CPU")
    parser.add_argument('--trainlog_path',default='')
    parser.add_argument('--testlog_path',default='')       
    # wrn, densenet
    parser.add_argument('--widen_factor', default=1, type=int)
    parser.add_argument('--dropRate', default=0.1, type=float)
    parser.add_argument('--growthRate', default=2, type=int)
    parser.add_argument('--compressionRate', default=2, type=int)
    # pyramid
    parser.add_argument('--alpha', default=48, type=int)
    parser.add_argument('--block_type', default='basic', type=str)
    parser.add_argument('--base_channels', default=16, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    # other argument
    parser.add_argument('--resume', '-r', action='store_true')         # resume from checkpoint
    parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing.")
    # minimizer argument
    parser.add_argument('--minimizer', default='SMSAM', type=str, help="SMSAM/ASAM/SAM.")
    parser.add_argument('--rho', default=0.01, type=float, help="Rho for SAM/SMSAM/ASAM.")
    parser.add_argument('--eta', default=0.0, type=float, help="Eta for ASAM.")
    # optimizer argument
    parser.add_argument('--optimizer', default='SGD', type=str)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float, help="Weight decay factor.")
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--stat_decay', default=0.9, type=float)           # for SGD_mod
    parser.add_argument('--damping', default=1e-3, type=float)
    parser.add_argument('--batch_averaged', default=True, type=bool)        # for SGD_mod
    parser.add_argument('--TCov', default=100, type=int)         # for ESGD_mod
    parser.add_argument('--TInv', default=100, type=int)        # for ESGD_mod


    """ CIFAR-10 """
    for idx in range(3):
        dataset = 'CIFAR10'
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'wrn'
        depth = 28
        widen_factor = 6
        minimizer = ['SMSAM','ASAM', 'SAM']
        rho = [3e-2, 1e-2, 1e-1]
        momentum = [0.9,0.9,0.9]
        train_batch = [128,128,128]
        test_batch = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/accuracy/'+network.lower()+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/accuracy/'+network.lower()+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[idx]),
            '--network', network,
            '--dataset', dataset,
            '--depth', depth,
            '--widen_factor',widen_factor,
            '--minimizer', minimizer[idx],
            '--rho', str(rho[idx]),
            '--train_batch_size',str(train_batch[idx]),
            '--test_batch_size',str(test_batch[idx]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['SMSAM','ASAM', 'SAM'], \
                f"Invalid minimizer type. Please select ASAM or SAM or SMSAM"
        train(args_train)
    for idx in range(3):
        dataset = 'CIFAR10'
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'vit_small'
        minimizer = ['SMSAM','ASAM', 'SAM']
        rho = [1e-2, 1e-1, 1e-2]
        momentum = [0.9,0.9,0.9]
        train_batch = [128,128,128]
        test_batch = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/accuracy/'+network.lower()+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/accuracy/'+network.lower()+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[idx]),
            '--network', network,
            '--dataset', dataset,
            '--minimizer', minimizer[idx],
            '--rho', str(rho[idx]),
            '--train_batch_size',str(train_batch[idx]),
            '--test_batch_size',str(test_batch[idx]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['SMSAM','ASAM', 'SAM'], \
                f"Invalid minimizer type. Please select ASAM or SAM or SMSAM"
        train(args_train)

    """ CIFAR-100 """
    for idx in range(3):
        dataset = 'CIFAR100'
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'wrn'
        depth = 28
        widen_factor = 6
        minimizer = ['SMSAM','ASAM', 'SAM']
        rho = [3e-2, 1e-2, 1e-1]
        momentum = [0.9,0.9,0.9]
        train_batch = [128,128,128]
        test_batch = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/accuracy/'+network.lower()+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/accuracy/'+network.lower()+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[idx]),
            '--network', network,
            '--dataset', dataset,
            '--depth', depth,
            '--widen_factor',widen_factor,
            '--minimizer', minimizer[idx],
            '--rho', str(rho[idx]),
            '--train_batch_size',str(train_batch[idx]),
            '--test_batch_size',str(test_batch[idx]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['SMSAM','ASAM', 'SAM'], \
                f"Invalid minimizer type. Please select ASAM or SAM or SMSAM"
        train(args_train)
    for idx in range(3):
        dataset = 'CIFAR100'
        learning_rate = [1e-1,1e-1,1e-1]
        network = 'vit_small'
        minimizer = ['SMSAM','ASAM', 'SAM']
        rho = [1e-2, 1e-1, 1e-2]
        momentum = [0.9,0.9,0.9]
        train_batch = [128,128,128]
        test_batch = [128,128,128]
        trainlog_path = 'logs/'+dataset.lower()+'/accuracy/'+network.lower()+'/train'
        testlog_path = 'logs/'+dataset.lower()+'/accuracy/'+network.lower()+'/test'
        args_train = parser.parse_known_args(args=[
            '--learning_rate', str(learning_rate[idx]),
            '--network', network,
            '--dataset', dataset,
            '--minimizer', minimizer[idx],
            '--rho', str(rho[idx]),
            '--train_batch_size',str(train_batch[idx]),
            '--test_batch_size',str(test_batch[idx]),
            '--trainlog_path', trainlog_path,
            '--testlog_path', testlog_path])[0]
        assert args_train.minimizer in ['SMSAM','ASAM', 'SAM'], \
                f"Invalid minimizer type. Please select ASAM or SAM or SMSAM"
        train(args_train)
        