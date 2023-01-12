import numpy as np

num_user = 16
num_slot=50
num_var=4
batch_size = 16
flops = batch_size*1e3*np.array([118.78, 122.880, 461.82, 293.377, 207.61, 222.26, 222.26, 249.28, 139.04,
                  127.97, 127.97, 510.42, 859.21, 659.22, 797.6, 797.6, 155.52, 1228.8,12.8])

size = 8 * batch_size*np.array([16 * 16 * 16, 16 * 16 * 16, 8 * 8 * 24, 8 * 8 * 24, 4 * 4 * 40, 4 * 4 * 40,
                 4 * 4 * 40, 2 * 2 * 80, 2 * 2 * 80, 2 * 2 * 80, 2 * 2 * 80, 2 * 2 * 112,
                 2 * 2 * 112, 1 * 1 * 160, 1 * 1 * 160, 1 * 1 * 160, 960, 1280,10])

params=8 * np.array([464,464,3440,4440,10328,20992,20992,32080,34760,31992,31992,214424,
                386120,429224,797360,797360,155520,1230080,12810])
num_layer=flops.shape[0]
g=np.load('g.npy')
g1=g[-num_slot:,:num_user]

Band=3e6
N0=10**(-17)
p_u = 0.05
p_d = 0.5
fpc=1
f_max = np.array([0.8,1.,0.8,1.2,0.8,1.,1.2,0.8,1.,1.2,1.,0.8,1.,1.2,1.2,0.8])*1e9*fpc
alpha=2e-28
F_max=6e9*2
local_ep=3
bn=np.array([200,150,250,150,200,150,150,250,150,200,150,250,150,250,200,150])*local_ep

a=1e-10

def Func(Var):
    Vars = np.copy(Var)
    Vars = Vars.reshape(-1, num_user, num_var)
    Vars = Vars[:, np.newaxis, :, :].repeat(num_slot, axis=1)
    S_s = Vars[:, :, :, 0]
    S_s = np.array(np.ceil(S_s * (num_layer - 1) + a), dtype='uint8')
    S_s = np.concatenate([S_s[:, :, :, np.newaxis], np.tile(num_layer - 1, [Vars.shape[0], num_slot, num_user, 1])], -1).min(-1)
    S_e = Vars[:, :, :, 1]
    S_e = np.array(np.floor(S_e * (num_layer - S_s)), dtype='uint8') + S_s
    S_e = np.concatenate([S_e[:, :, :, np.newaxis], np.tile(num_layer - 1, [Vars.shape[0], num_slot, num_user, 1])], -1).min(-1)
    f = Vars[:, :, :, 2]
    f = (S_e != S_s) * (f + a) * F_max / (((S_e != S_s) * (f + a)).sum(2)[:, :, np.newaxis] + a)
    f[np.isnan(f)] = 0
    B_params = Vars[:, :, :, 3] + a
    B_params = (B_params) * Band / (B_params).sum(2)[:, :, np.newaxis]
    B_split = B_params

    layer_split1 = np.zeros([Vars.shape[0], num_slot, num_user, num_layer])
    layer_split2 = np.zeros([Vars.shape[0], num_slot, num_user, num_layer])
    layer_split3 = np.zeros([Vars.shape[0], num_slot, num_user, num_layer])
    for i in range(Vars.shape[0]):
        for j in range(num_slot):
            for k in range(num_user):
                layer_split1[i, j, k, :S_s[i, j, k]] = 1
                layer_split2[i, j, k, S_s[i, j, k]:S_e[i, j, k]] = 1
                layer_split3[i, j, k, S_e[i, j, k]:] = 1

    T_parU = ((layer_split1 + layer_split3) * params).sum(-1) / (
            B_params * np.log2(1 + p_u * g1[np.newaxis, :, :] / B_params / N0) )
    T_parD = ((layer_split1 + layer_split3) * params).sum(-1) / (
            B_params * np.log2(1 + p_d * g1[np.newaxis, :, :] / B_params / N0) )

    T_sum_aa = (layer_split1 * flops).sum(-1) * 2 / f_max[np.newaxis, np.newaxis, :]
    T_sum_cc = (layer_split3 * flops).sum(-1) * 2 / f_max[np.newaxis, np.newaxis, :]
    T_UF = size[S_s-1] / (B_split * np.log2(1 + p_u * g1[np.newaxis, :, :] / ( B_split) / N0) ) * (S_e != S_s)
    T_UF[np.isnan(T_UF)] = 0
    T_UB = size[S_e-1] / (B_split * np.log2(1 + p_u * g1[np.newaxis, :, :] / ( B_split) / N0) ) * ( S_e != S_s)
    T_UB[np.isnan(T_UB)] = 0
    T_DF = size[S_e-1] / (B_split * np.log2(1 + p_d * g1[np.newaxis, :, :] / ( B_split) / N0) ) * (S_e != S_s)
    T_DF[np.isnan(T_DF)] = 0
    T_DB = size[S_s - 1] / (B_split * np.log2(1 + p_d * g1[np.newaxis, :, :] / ( B_split) / N0) ) * ( S_e != S_s)
    T_DB[np.isnan(T_DB)] = 0
    T_EF = (layer_split2 * flops).sum(-1) / (f) * (S_e != S_s)
    T_EF[np.isnan(T_EF)] = 0
    T_EB = T_EF

    T_s1 = np.concatenate([(T_UB + T_EB + T_DB)[:, :, :, np.newaxis], T_sum_cc[:, :, :, np.newaxis]], -1).max(-1)
    T_s2 = np.concatenate([(T_UB + T_EB + T_DB)[:, :, :, np.newaxis], T_sum_aa[:, :, :, np.newaxis]], -1).max(-1)
    T_s3 = np.concatenate([(T_UF + T_EF + T_DF)[:, :, :, np.newaxis], T_sum_aa[:, :, :, np.newaxis]], -1).max(-1)
    T_s4 = np.concatenate([(T_UF + T_EF + T_DF)[:, :, :, np.newaxis], T_sum_cc[:, :, :, np.newaxis]], -1).max(-1)

    T_split = (bn[np.newaxis, np.newaxis, :] / 2 * (T_s1 + T_s2 + T_s3 + T_s4) + T_parD + T_parU) * (S_e != S_s)

    E_s1 = ((layer_split3 * flops).sum(-1) * 2 / (T_s1) / fpc) ** 3 * T_s1 * alpha + p_u * T_UB
    E_s1[np.isnan(E_s1)] = 0
    E_s2 = ((layer_split1 * flops).sum(-1) * 2 / (T_s2) / fpc) ** 3 * T_s2 * alpha + p_u * T_UB
    E_s2[np.isnan(E_s2)] = 0
    E_s3 = ((layer_split1 * flops).sum(-1) * 2 / (T_s3 ) / fpc) ** 3 * T_s3 * alpha + p_u * T_UF
    E_s3[np.isnan(E_s3)] = 0
    E_s4 = ((layer_split3 * flops).sum(-1) * 2 / (T_s4 ) / fpc) ** 3 * T_s4 * alpha + p_u * T_UF
    E_s4[np.isnan(E_s4)] = 0

    E_split = (bn[np.newaxis, np.newaxis, :] / 2 * (E_s1 + E_s2 + E_s3 + E_s4)) * (S_e != S_s)

    total_flops = 2*bn[np.newaxis, np.newaxis, :] * np.tile(flops.sum(), [Vars.shape[0], num_slot, num_user])
    T_fed = (total_flops / f_max[np.newaxis, np.newaxis, :] + T_parD + T_parU) * (S_e == S_s)

    T_total = (T_split + T_fed).max(2)
    T_final = T_total.sum(-1)

    T_fed1 = T_total[:, :, np.newaxis].repeat(num_user, axis=2) - T_parD - T_parU
    E_fed = (total_flops / (T_fed1 ) / fpc) ** 3 * T_fed1 * alpha * (S_e == S_s)

    E_final = (T_parU * p_u + E_fed + E_split).sum(-1).sum(-1)

    return np.concatenate([T_final[:, np.newaxis], E_final[:, np.newaxis]], -1)
