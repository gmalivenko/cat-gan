import yaml
import argparse
from attrdict import AttrDict

from matplotlib import pyplot as plt

import torch
from torch.autograd import Variable

from models.generator import Generator


def test(params):
    G = Generator(params.network.generator)

    if params.restore.G:
        G.load_state_dict(torch.load(params.restore.G))

    gen_input = \
        Variable(torch.FloatTensor(
            1, params.network.generator.z_size,
            1, 1
        ).normal_(0, 1))

    torch_cat = G(gen_input)
    np_cat = torch_cat.data.numpy()[0] / 2.0 + 0.5
    np_cat = np_cat.transpose((1, 2, 0))

    plt.imshow(np_cat)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GAN testing script'
    )
    parser.add_argument('--conf', '-c', required=True,
                        help='a path to the configuration file')
    args = parser.parse_args()

    with open(args.conf, 'r') as stream:
        try:
            args = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    test(AttrDict(args))
