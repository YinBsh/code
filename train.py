import argparse
import os
import numpy as np
import math
import prob as prob
import torch.nn as nn
import torch
from utils import uniformpoint,GO,envselect_pair,is_dominate
import copy
import random
import matplotlib.pyplot as plt
from datetime import datetime

seed=1000
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

out_path='result/{:%Y%m%d_%H%M%S}/'.format(datetime.now())
if not os.path.exists(out_path):
    os.makedirs(out_path)

n_epochs=5000
num_user = 16
num_var=4
POP_SIZE = 100
M = 2
t1 = 20
t2 = 20
pc = 1
pm = 1
D=num_user*num_var

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(D,D),
            nn.LeakyReLU(0.2),
            nn.Linear(D, D),
            nn.LeakyReLU(0.2),
            nn.Linear(D, D),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.l1=nn.Linear(D, D)
        self.l2=nn.Linear(D, D)
        self.l3=nn.Linear(2*D, 1)
        self.relu = nn.LeakyReLU(0.2)
        self.sig=nn.Sigmoid()


    def forward(self, nd,d):
        x1=self.relu(self.l1(nd))
        x2=self.relu(self.l2(d))
        return self.sig(self.l3(torch.cat([x1,x2],1)))


# Initialize generator and discriminator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=4e-4)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=4e-4)
torch.autograd.set_detect_anomaly(True)

plt.ion()


Z,N = uniformpoint(POP_SIZE,M)

low = np.zeros((1, D))
up = np.ones((1, D))
pop = np.tile(low, (N, 1)) + (np.tile(up, (N, 1)) - np.tile(low, (N, 1))) * np.random.rand(N, D)

N2=N
pop_pair=np.tile(low, (N2, 1)) + (np.tile(up, (N2, 1)) - np.tile(low, (N2, 1))) * np.random.rand(N2, D)
popfun_pair = prob.Func(pop_pair)



loss = torch.nn.BCELoss()

for epoch in range(n_epochs):
    print('epoch:{}'.format(epoch))

    if torch.rand(1).item() > 0.5:
        z = torch.randn(N, D, device=device) * torch.tensor(pop.std(0), device=device).repeat([N, 1]) \
            + torch.tensor(pop.mean(0), device=device).repeat([N, 1])

        off = generator(z.float().detach())
        off = off.cpu().detach().numpy()
    else:
        matingpool = random.sample(range(N), N)
        off = GO(pop[matingpool, :], t1, t2, pc, pm)


    mixpop = copy.deepcopy(np.vstack((pop, off)))
    pop,popfun, pop_dis,pop_pair,popfun_pair,pair1,pair2 = envselect_pair(pop_pair,popfun_pair,mixpop, N, Z, M, D)


    for i in range(10):
        D_out1 = discriminator(torch.Tensor(pair1).to(device).detach(), torch.Tensor(pair2).to(device).detach())
        loss_D1 = 10 * loss(D_out1, torch.ones(D_out1.shape[0], device=device).unsqueeze(1))
        D_out2 = discriminator(torch.Tensor(pair2).to(device).detach(), torch.Tensor(pair1).to(device).detach())
        loss_D2 = 10 * loss(D_out2, torch.zeros(D_out2.shape[0], device=device).unsqueeze(1))

        optimizer_D.zero_grad()
        (loss_D1 + loss_D2).backward()
        optimizer_D.step()

    for i in range(10):
        N1 = pop.shape[0]
        z = torch.randn(N1, D, device=device) * torch.tensor(pop.std(0), device=device).repeat([N1, 1]) \
            + torch.tensor(pop.mean(0), device=device).repeat([N1, 1])
        off = generator(z.float().detach())
        D_out3 = discriminator(off, torch.Tensor(pop).to(device))
        loss_G = 1 * loss(D_out3, torch.ones(D_out3.shape[0], device=device).unsqueeze(1))
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()


    if (epoch+1) % 200 == 0 or epoch==0:
        popfun = prob.Func(pop)
        label = is_dominate(popfun)
        nondom_set = popfun[label == 1, :]
        plt.cla()
        plt.grid()
        plt.xlabel('total time')
        plt.ylabel('total energy')
        plt.title('population={}'.format(epoch+1))
        for p in range(nondom_set.shape[0]):
            plt.plot(nondom_set[p, 0].item(), nondom_set[p, 1].item(), 'r*')

        plt.savefig(out_path+'{}.png'.format(epoch + 1))
        plt.pause(0.01)
        np.save(out_path + 'nondom_set_{}.npy'.format(epoch+1), nondom_set)
        np.save(out_path + 'nondom_pop_{}.npy'.format(epoch+1), pop[label == 1, :])

plt.ioff()
plt.show()
