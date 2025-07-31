from models.imagenet import (densenet, resnet, wrn, pyramidnet, vit_small, vit_base, vit_large, vit_huge)

def get_network_imagenet(network, **kwargs):

    networks = {
        'densenet': densenet,
        'resnet': resnet,
        'wrn': wrn,
        'pyramidnet': pyramidnet,
        'vit_small': vit_small,
        'vit_base': vit_base,
        'vit_large': vit_large,
        'vit_huge': vit_huge
    }
    # print(networks[network])
    return networks[network](**kwargs)

