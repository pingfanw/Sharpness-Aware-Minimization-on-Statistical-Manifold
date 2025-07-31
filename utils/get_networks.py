from utils.network_utils_cifar import get_network_cifar
from utils.network_utils_mnist import get_network_mnist
from utils.network_utils_imagenet import get_network_imagenet
from utils.network_utils_miniimagenet import get_network_miniimagenet
from utils.network_utils_svhn import get_network_svhn

def get_network(args):
        nc = { 
            'mnist': 10,
            'fashionmnist': 10,
            'svhn': 10,
            'cifar10': 10,
            'cifar100': 100,
            'imagenet': 1000,
            'miniimagenet': 100
        }
        idm = {
            'mnist': 28,
            'fashionmnist': 28,
            'svhn': 32,
            'cifar10': 32,
            'cifar100': 32,
            'imagenet': 224,
            'miniimagenet': 84
        }
        icn = {
            'mnist': 1,
            'fashionmnist': 1,
            'svhn': 3,
            'cifar10': 3,
            'cifar100': 3,
            'imagenet': 3,
            'miniimagenet': 3
        }
        args.outputs_dim = nc[args.dataset.lower()]
        args.inputs_channel = icn[args.dataset.lower()]
        args.inputs_dim = idm[args.dataset.lower()]
        if args.dataset.lower() == 'mnist' or args.dataset.lower() == 'fashionmnist':
            if args.network == 'wrn':
                net = get_network_mnist(args.network,
                              depth=args.depth,
                              num_classes=args.outputs_dim,
                              growthRate=args.growthRate,
                              compressionRate=args.compressionRate,
                              widen_factor=args.widen_factor,
                              dropRate=args.dropRate).to(args.device)
            else:
                net = get_network_mnist(args.network).to(args.device)
        elif args.dataset.lower() == 'cifar10' or args.dataset.lower() == 'cifar100':
            if args.network == 'preresnet':
                net = get_network_cifar(args.network,
                              depth=args.depth,
                              num_classes=args.outputs_dim).to(args.device)
            elif args.network == 'pyramidnet':
                net = get_network_cifar(args.network,
                              depth = args.depth,
                              alpha = 48,
                              input_shape = (1, args.inputs_channel, args.inputs_dim, args.inputs_dim),
                              num_classes = args.outputs_dim,
                              base_channels = 16,
                              block_type = 'bottleneck'
                              ).to(args.device)
            elif args.network.lower() in ['vit_small','vit_base', 'vit_large', 'vit_huge']:
                net = get_network_cifar(args.network,
                                        mlp_ratio=args.mlp_ratio,
                                        input_size=args.inputs_dim,
                                        patch_size=args.patch_size,
                                        in_channels=args.inputs_channel,
                                        num_classes=args.outputs_dim).to(args.device)
            else:
                net = get_network_cifar(args.network,
                              depth=args.depth,
                              num_classes=args.outputs_dim,
                              growthRate=args.growthRate,
                              compressionRate=args.compressionRate,
                              widen_factor=args.widen_factor,
                              dropRate=args.dropRate).to(args.device)
        elif args.dataset.lower() == 'svhn':
            if args.network.lower() == 'preresnet':
                net = get_network_svhn(args.network,depth=args.depth,num_classes=args.outputs_dim).to(args.device)
            elif args.network.lower() == 'pyramidnet':
                net = get_network_svhn(args.network,
                                       depth = args.depth,
                                       alpha = 48,
                                       input_shape = (1,args.inputs_channel,args.inputs_dim,args.inputs_dim),
                                       num_classes = args.outputs_dim,
                                       base_channels = 16,
                                       block_type = 'bottleneck').to(args.device)
            elif args.network.lower() in ['vit_small','vit_base', 'vit_large', 'vit_huge']:
                net = get_network_svhn(args.network,
                                       mlp_ratio=args.mlp_ratio,
                                       input_size=args.inputs_dim,
                                       patch_size=args.patch_size,
                                       in_channels=args.inputs_channel,
                                       num_classes=args.outputs_dim).to(args.device)
            else:
                net = get_network_svhn(args.network,
                                       depth=args.depth,
                                       num_classes=args.outputs_dim,
                                       growthRate=args.growthRate,
                                       compressionRate=args.compressionRate,
                                       widen_factor=args.widen_factor,
                                       dropRate=args.dropRate).to(args.device)
        elif args.dataset.lower() == 'imagenet':
            if args.network == 'pyramidnet':
                net = get_network_imagenet(args.network,
                                depth=args.depth,
                                alpha=48,
                                input_shape=(1, args.inputs_channel, args.inputs_dim, args.inputs_dim),
                                num_classes=args.outputs_dim,
                                base_channels=16,
                                block_type='bottleneck').to(args.device)
            elif args.network.lower() in ['vit_small','vit_base', 'vit_large', 'vit_huge']:
                net = get_network_imagenet(args.network,
                                        mlp_ratio=args.mlp_ratio,
                                        input_size=args.inputs_dim,
                                        patch_size=args.patch_size,
                                        in_channels=args.inputs_channel,
                                        num_classes=args.outputs_dim).to(args.device)
            else:
                net = get_network_imagenet(args.network,
                              depth=args.depth,
                              num_classes=args.outputs_dim,
                              growthRate=args.growthRate,
                              compressionRate=args.compressionRate,
                              widen_factor=args.widen_factor,
                              dropRate=args.dropRate).to(args.device)
        elif args.dataset.lower() == 'miniimagenet':
            if args.network == 'pyramidnet':
                net = get_network_miniimagenet(args.network,
                                depth=args.depth,
                                alpha=48,
                                input_shape=(1, args.inputs_channel, args.inputs_dim, args.inputs_dim),
                                num_classes=args.outputs_dim,
                                base_channels=16,
                                block_type='bottleneck').to(args.device)
            elif args.network.lower() in ['vit_small','vit_base', 'vit_large', 'vit_huge']:
                net = get_network_miniimagenet(args.network,
                                        mlp_ratio=args.mlp_ratio,
                                        input_size=args.inputs_dim,
                                        patch_size=args.patch_size,
                                        in_channels=args.inputs_channel,
                                        num_classes=args.outputs_dim).to(args.device)
            else:
                net = get_network_miniimagenet(args.network,
                              depth=args.depth,
                              num_classes=args.outputs_dim,
                              growthRate=args.growthRate,
                              compressionRate=args.compressionRate,
                              widen_factor=args.widen_factor,
                              dropRate=args.dropRate).to(args.device)
        return net