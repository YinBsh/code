from scipy.special import comb
from itertools import combinations
import numpy as np
import copy
import math
import prob as prob
import torch

def NDsort(mixpop,N,M):
    nsort = N
    N,M = mixpop.shape[0],mixpop.shape[1]
    Loc1=np.lexsort(mixpop[:,::-1].T)
    mixpop2=mixpop[Loc1]
    Loc2=Loc1.argsort()
    frontno=np.ones(N)*(np.inf)

    maxfno=0
    while (np.sum(frontno < np.inf) < min(nsort,N)):
        maxfno=maxfno+1
        for i in range(N):
            if (frontno[i] == np.inf):
                dominated = 0
                for j in range(i):
                    if (frontno[j] == maxfno):
                        m=0
                        flag=0
                        while (m<M and mixpop2[i,m]>=mixpop2[j,m]):
                            if(mixpop2[i,m]==mixpop2[j,m]):
                                flag=flag+1
                            m=m+1
                        if (m>=M and flag < M):
                            dominated = 1
                            break
                if dominated == 0:
                    frontno[i] = maxfno
    frontno=frontno[Loc2]
    return frontno,maxfno


def GO(pop, t1, t2, pc, pm):
    pop1 = copy.deepcopy(pop[0:int(pop.shape[0] / 2), :])
    pop2 = copy.deepcopy(pop[(int(pop.shape[0] / 2)):(int(pop.shape[0] / 2) * 2), :])
    N, D = pop1.shape[0], pop1.shape[1]

    beta = np.zeros((N, D))
    mu = np.random.random_sample([N, D])
    beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (t1 + 1))
    beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (t1 + 1))
    beta = beta * ((-1) ** (np.random.randint(2, size=(N, D))))

    beta[np.tile(np.random.random_sample([N, 1]) > pc, (1, D))] = 1
    off = np.vstack(((pop1 + pop2) / 2 + beta * (pop1 - pop2) / 2, (pop1 + pop2) / 2 - beta * (pop1 - pop2) / 2))

    low = np.zeros((2 * N, D))
    up = np.ones((2 * N, D))
    site = np.random.random_sample([2 * N, D]) < pm / D

    mu = np.random.random_sample([2 * N, D])
    temp = site & (mu <= 0.5)
    off[off < low] = low[off < low]
    off[off > up] = up[off > up]
    off[temp] = off[temp] + (up[temp] - low[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (
                (1 - (off[temp] - low[temp]) / (up[temp] - low[temp])) ** (t2 + 1))) ** (1 / (t2 + 1)) - 1)
    temp = site & (mu > 0.5)
    off[temp] = off[temp] + (up[temp] - low[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (
                (1 - (up[temp] - off[temp]) / (up[temp] - low[temp])) ** (t2 + 1))) ** (1 / (t2 + 1)))

    return off


def uniformpoint(N,M):
    H1=1
    while (comb(H1+M-1,M-1)<=N):
        H1=H1+1
    H1=H1-1
    W=np.array(list(combinations(range(H1+M-1),M-1)))-np.tile(np.array(list(range(M-1))),(int(comb(H1+M-1,M-1)),1))
    W=(np.hstack((W,H1+np.zeros((W.shape[0],1))))-np.hstack((np.zeros((W.shape[0],1)),W)))/H1
    if H1<M:
        H2=0
        while(comb(H1+M-1,M-1)+comb(H2+M-1,M-1) <= N):
            H2=H2+1
        H2=H2-1
        if H2>0:
            W2=np.array(list(combinations(range(H2+M-1),M-1)))-np.tile(np.array(list(range(M-1))),(int(comb(H2+M-1,M-1)),1))
            W2=(np.hstack((W2,H2+np.zeros((W2.shape[0],1))))-np.hstack((np.zeros((W2.shape[0],1)),W2)))/H2
            W2=W2/2+1/(2*M)
            W=np.vstack((W,W2))
    W[W<1e-6]=1e-6
    N=W.shape[0]
    return W,N


def pdist(x, y):
    x0 = x.shape[0]
    y0 = y.shape[0]
    xmy = np.dot(x, y.T)
    xm = np.array(np.sqrt(np.sum(x ** 2, 1))).reshape(x0, 1)
    ym = np.array(np.sqrt(np.sum(y ** 2, 1))).reshape(1, y0)
    xmmym = np.dot(xm, ym)
    cos = xmy / (xmmym)
    return cos


def lastselection(popfun1, popfun2, K, Z, Zmin):
    popfun = copy.deepcopy(np.vstack((popfun1, popfun2))) - np.tile(Zmin, (popfun1.shape[0] + popfun2.shape[0], 1))
    N, M = popfun.shape[0], popfun.shape[1]
    N1 = popfun1.shape[0]
    N2 = popfun2.shape[0]
    NZ = Z.shape[0]

    extreme = np.zeros(M)
    w = np.zeros((M, M)) + 1e-6 + np.eye(M)
    for i in range(M):
        extreme[i] = np.argmin(np.max(popfun / (np.tile(w[i, :], (N, 1))), 1))

    extreme = extreme.astype(int)
    temp = np.linalg.pinv(np.mat(popfun[extreme, :]))
    hyprtplane = np.array(np.dot(temp, np.ones((M, 1))))
    a = 1 / (hyprtplane)
    if np.sum(a == math.nan) != 0:
        a = np.max(popfun, 0)
    np.array(a).reshape(M, 1)
    a = a.T
    popfun = popfun / (np.tile(a, (N, 1)))

    cos = pdist(popfun, Z)
    distance = np.tile(np.array(np.sqrt(np.sum(popfun ** 2, 1))).reshape(N, 1), (1, NZ)) * np.sqrt(1 - cos ** 2)
    d = np.min(distance.T, 0)
    pi = np.argmin(distance.T, 0)

    rho = np.zeros(NZ)
    for i in range(NZ):
        rho[i] = np.sum(pi[:N1] == i)

    choose = np.zeros(N2)
    choose = choose.astype(bool)
    zchoose = np.ones(NZ)
    zchoose = zchoose.astype(bool)
    while np.sum(choose) < K:
        temp = np.ravel(np.array(np.where(zchoose == True)))
        jmin = np.ravel(np.array(np.where(rho[temp] == np.min(rho[temp]))))
        j = temp[jmin[np.random.randint(jmin.shape[0])]]
        I = np.ravel(np.array(np.where(pi[N1:] == j)))
        I = I[choose[I] == False]
        if (I.shape[0] != 0):
            if (rho[j] == 0):
                s = np.argmin(d[N1 + I])
            else:
                s = np.random.randint(I.shape[0])
            choose[I[s]] = True
            rho[j] = rho[j] + 1
        else:
            zchoose[j] = False
    return choose

def is_dominate(fun_value):
    label=torch.zeros(fun_value.shape[0])
    epi=0
    for i in range(fun_value.shape[0]):
        x1=fun_value[:,0]<(fun_value[i,0]-epi)
        x2=fun_value[:,1]<(fun_value[i,1]-epi)
        if (x1*x2).sum().item()==0 and fun_value[i,0]<36000:
            label[i] = 1
        else:
            label[i] = 0

    return label

def envselect_pair( pop_pair, popfun_pair, mixpop, N, Z, M, D):

    mixpopfun = prob.Func(mixpop)

    x_th=36000

    if (mixpopfun[:, 0] < x_th).sum() < N:
        pop=copy.deepcopy(mixpop[mixpopfun[:, 0].argsort()[:N]])
        popfun=copy.deepcopy(mixpopfun[mixpopfun[:, 0].argsort()[:N]])
        pop_dis=copy.deepcopy(mixpop[mixpopfun[:, 0].argsort()[N:]])
        popfun_dis = copy.deepcopy(mixpopfun[mixpopfun[:, 0].argsort()[N:]])
        Zmin = np.array(np.min(popfun, 0)).reshape(1, M)

    else:
        mixpopfun = mixpopfun[np.where((mixpopfun[:, 0] < x_th))]
        mixpop = mixpop[np.where((mixpopfun[:, 0] < x_th))]

        Zmin = np.array(np.min(mixpopfun, 0)).reshape(1, M)
        frontno, maxfno = NDsort(mixpopfun, N, M)
        Next = frontno < maxfno

        Last = np.ravel(np.array(np.where(frontno == maxfno)))
        choose = lastselection(mixpopfun[Next, :], mixpopfun[Last, :], N - np.sum(Next), Z, Zmin)
        Next[Last[choose]] = True

        pop = copy.deepcopy(mixpop[Next, :])
        popfun = copy.deepcopy(mixpopfun[Next, :])
        pop_dis = copy.deepcopy(mixpop[~ Next, :])
        popfun_dis = copy.deepcopy(mixpopfun[~ Next, :])


    mix_pair = np.vstack((pop_dis, pop_pair))
    mixfun_pair = np.vstack((popfun_dis, popfun_pair))
    index = (np.zeros(mix_pair.shape[0]) == np.ones(mix_pair.shape[0]))

    th = 6400
    no = 6
    Zmin1 = Zmin[0] - 1

    pair1 = np.array([])
    pair2 = np.array([])
    for i in range(N):
        candidate = np.array([])
        can_list = []
        fun1 = popfun[i]
        for j in range(mix_pair.shape[0]):
            fun2 = mixfun_pair[j]
            if (fun1[0] <= fun2[0]) and (fun1[1] <= fun2[1]) and (
                    ((fun2[0] - fun1[0]) ** 2 + (fun2[1] - fun1[1]) ** 2) <= th):
                can_list.append(j)
                candidate = np.concatenate([candidate, fun2], 0)

        if len(can_list) > no:
            candidate = candidate.reshape(-1, M)
            can_dis = np.abs((candidate[:, 1] - Zmin1[1]) / (fun1[1] - Zmin1[1]) - (candidate[:, 0] - Zmin1[0]) / (
                        fun1[0] - Zmin1[0])) \
                      / np.sqrt((1 / (fun1[1] - Zmin1[1])) ** 2 + (1 / (fun1[0] - Zmin1[0])) ** 2)
            can_sort = can_dis.argsort()[:no]
            can_list = np.array(can_list)[can_sort]

        for item in can_list:
            pair1 = np.concatenate([pair1, pop[i]], 0)
            pair2 = np.concatenate([pair2, mix_pair[item]], 0)
            index[item] = True

    pair1 = pair1.reshape(-1, D)
    pair2 = pair2.reshape(-1, D)

    pop_pair1 = mix_pair[index, :]
    popfun_pair1 = mixfun_pair[index, :]

    return pop, popfun, pop_dis, pop_pair1, popfun_pair1, pair1, pair2
