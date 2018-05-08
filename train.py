import yaml
import argparse
import numpy as np
from attrdict import AttrDict

import torch
import torch.utils.data as data_utils
from torch.autograd import Variable

from models.generator import Generator
from models.discriminator import discriminator

from dataset import CatsDataset


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    return


def train(params):
    if params.visdom.use_visdom:
        import visdom
        vis = visdom.Visdom(server=params.visdom.server, port=params.visdom.port)
        generated = vis.images(
            np.ones((params.train.batch_size, 3, params.visdom.image_size, params.visdom.image_size)),
            opts=dict(title='Generated')
        )

        gt = vis.images(
            np.ones((params.train.batch_size, 3, params.visdom.image_size, params.visdom.image_size)),
            opts=dict(title='Original')
        )

        # initialize visdom loss plots
        g_loss = vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1,)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='G Training Loss',
            )
        )
        d_loss = vis.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1,)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='D Training Loss',
            )
        )

    G = Generator(params.network.generator)
    D = discriminator()
    G.apply(weights_init)
    D.apply(weights_init)
    
    if params.restore.G:
        G.load_state_dict(torch.load(params.restore.G))

    if params.restore.D:
        D.load_state_dict(torch.load(params.restore.D))

    d_steps = 1
    g_steps = 1

    criterion = torch.nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=params.train.d_learning_rate)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=params.train.g_learning_rate)

    dataset = CatsDataset(params.dataset)
    train_loader = data_utils.DataLoader(
        dataset, batch_size=params.train.batch_size, shuffle=True,
        num_workers=0)

    t_ones = Variable(torch.ones(params.train.batch_size))
    t_zeros = Variable(torch.zeros(params.train.batch_size))

    if params.train.use_cuda:
        t_ones = t_ones.cuda()
        t_zeros = t_zeros.cuda()
        G = G.cuda()
        D = D.cuda()

    for epoch in range(params.train.epochs):
        for p in D.parameters():
            p.requires_grad = True

        for d_index in range(d_steps):
            # 1. Train D on real+fake
            D.zero_grad()

            #  1A: Train D on real
            d_real_data = Variable(next(iter(train_loader)))

            if params.train.use_cuda:
                d_real_data = d_real_data.cuda()

            d_real_decision = D(d_real_data)
            d_real_error = criterion(d_real_decision, t_ones)  # ones = true
            d_real_error.backward()
            # compute/store gradients, but don't change params
            d_optimizer.step()

            #  1B: Train D on fake
            d_gen_input = \
                Variable(torch.FloatTensor(
                    params.train.batch_size, params.network.generator.z_size,
                    1, 1
                ).normal_(0, 1))

            if params.train.use_cuda:
                d_gen_input = d_gen_input.cuda()

            d_fake_input = G(d_gen_input)

            d_fake_decision = D(d_fake_input)
            d_fake_error = criterion(d_fake_decision, t_zeros)  # zeros = fake
            d_fake_error.backward()

            loss = d_fake_error + d_real_error
            np_loss = loss.cpu().data.numpy()

            # Only optimizes D's parameters;
            # changes based on stored gradients from backward()
            d_optimizer.step()

        for p in D.parameters():
            p.requires_grad = False

        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            G.zero_grad()

            gen_input = \
                Variable(torch.FloatTensor(
                    params.train.batch_size, params.network.generator.z_size,
                    1, 1
                ).normal_(0, 1))

            if params.train.use_cuda:
                gen_input = gen_input.cuda()

            dg_fake_decision = D(G(gen_input))
            # we want to fool, so pretend it's all genuine
            g_error = criterion(dg_fake_decision, t_ones)

            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters

        if epoch % 10 == 0 and params.visdom.use_visdom:
            vis.line(
                X=np.array([epoch]),
                Y=np.array([np_loss]),
                win=d_loss,
                update='append'
            )

            vis.images(
                255 * (d_fake_input.cpu().data.numpy() / 2.0 + 0.5),
                win=generated
            )

            vis.images(
                255 * (d_real_data.cpu().data.numpy() / 2.0 + 0.5),
                win=gt
            )

            vis.line(
                X=np.array([epoch]),
                Y=np.array([g_error.cpu().data.numpy()]),
                win=g_loss,
                update='append'
            )

        print('Epoch: {0}, D: {1}, G: {2}'.format(
            epoch, d_fake_error.data[0], g_error.data[0]))

        if epoch % params.save.every == 0:
            torch.save(D.state_dict(), params.save.D)
            torch.save(G.state_dict(), params.save.G)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GAN training script'
    )
    parser.add_argument('--conf', '-c', required=True,
                        help='a path to the configuration file')
    args = parser.parse_args()

    with open(args.conf, 'r') as stream:
        try:
            args = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    train(AttrDict(args))
