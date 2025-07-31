from models.mnist import (wrn,mlp)

def get_network_mnist(network, **kwargs):

    networks = {
        'wrn': wrn,
        'mlp': mlp
    }
    return networks[network](**kwargs)

