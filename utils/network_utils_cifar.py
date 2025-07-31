from models.cifar import (alexnet, densenet, resnet,
                          vgg16_bn, vgg19_bn, wrn, preresnet, mlp, pyramidnet, vit_small, vit_base, vit_large, vit_huge)

def get_network_cifar(network, **kwargs):
    networks = {
        'alexnet': alexnet,
        'densenet': densenet,
        'resnet': resnet,
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'wrn': wrn,
        'preresnet': preresnet,
        'mlp': mlp,
        'pyramidnet': pyramidnet,
        'vit_small': vit_small,
        'vit_base': vit_base,
        'vit_large': vit_large,
        'vit_huge': vit_huge
    }
    # print(networks[network])
    return networks[network](**kwargs)

