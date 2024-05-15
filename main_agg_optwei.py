import argparse
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb
from torch.autograd import Variable
import torch.nn.functional as F

from dataset.CramedDataset import CramedDataset, DatasetSplit
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
from models.basic_model import AVClassifier
from utils import setup_seed, weight_init
import copy

device = torch.device('cuda:0')


def get_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default='CREMAD', type=str,
    #                     help='VGGSound, KineticSound, CREMAD, AVE')
    # parser.add_argument('--modulation', default='OGM', type=str,
    #                     choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--train_path', default='D:\Desktop\ybs\multimodal\dataset\CREMAD\cremad_train1.pkl', type=str)
    parser.add_argument('--test_path', default='D:\Desktop\ybs\multimodal\dataset\CREMAD\cremad_test1.pkl', type=str)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--num_users', default=6, type=int)

    parser.add_argument('--train', default=True, help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=True, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', default=r'D:\Desktop\ybs\fed_agg\cremad_classfi_fed1\results', type=str,
                        help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=1000, type=int)
    # parser.add_argument('--gpu_ids', default='0, 1', type=str, help='GPU ids')

    return parser.parse_args()


l_dim = 512
num_users=9
Band = 0.5e6  # total bandwidth (20 MHz)
N0 = 10 ** (-17)  # noise power (-140 dBm/Hz)
p_u = 0.1  # upload power (W)
f_max = 2 * 1e9
fpc = 16
flop=[(1.82 * 1e9 + 512 * 1024) * 16, (1.79 * 1e9 + 512 * 1024) * 16, 1024 * 6 * 16] #visual, audio, share
parm = [11.18 * 1e6 * 8 + 512 * 1024 * 8, 11.17 * 1e6 * 8 + 512 * 1024 * 8, 1024 * 6 * 8]

def train_epoch(args, model, dataloader, lr, local_ep, modal_w):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.to(device)
    model.train()

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    for ep in range(local_ep):
        for step, (spec, image, label, idx) in enumerate(dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            out = model(spec.unsqueeze(1).float(), image.float(), modal_w)
            loss = criterion(out, label)

            loss.backward()
            optimizer.step()
            _loss += loss.item()

    return _loss / len(dataloader), model


def valid(args, model, dataloader, modal_w):
    softmax = nn.Softmax(dim=1)

    n_classes = 6

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label, idx) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            out = model(spec.unsqueeze(1).float(), image.float(), modal_w)

            prediction = softmax(out)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                num[label[i]] += 1.0

                # pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0

    return sum(acc) / sum(num)


def Weight_Avg(w, alpha, weight, id):
    w_avg = copy.deepcopy(w[id])

    # 用户id自己的参数
    for k in w_avg.keys():
        if 'visual' in k:
            w_avg[k] = torch.mul(w_avg[k], weight[0, id, id])  # 0为visual的聚合系数
        elif 'audio' in k:
            w_avg[k] = torch.mul(w_avg[k], weight[1, id, id])
        elif k == 'fusion_module.fc1.weight':
            w_avg[k][:, :l_dim] = torch.mul(w_avg[k][:, :l_dim].clone(), weight[1, id, id])
            w_avg[k][:, l_dim:] = torch.mul(w_avg[k][:, l_dim:].clone(), weight[0, id, id])
        else:
            w_avg[k] = torch.mul(w_avg[k], weight[2, id, id])

    # 聚合其他用户的参数
    for k in w_avg.keys():
        if 'visual' in k:
            for i in range(len(w)):
                if i != id:
                    w_avg[k] = w_avg[k] + torch.mul(w[i][k], weight[0, id, i])
        elif 'audio' in k:
            for i in range(len(w)):
                if i != id:
                    w_avg[k] = w_avg[k] + torch.mul(w[i][k], weight[1, id, i])
        elif k == 'fusion_module.fc1.weight':
            for i in range(len(w)):
                if i != id:
                    w_avg[k][:, :l_dim] = w_avg[k][:, :l_dim].clone() + torch.mul(w[i][k][:, :l_dim].clone(),
                                                                                  weight[1, id, i])
                    w_avg[k][:, l_dim:] = w_avg[k][:, l_dim:].clone() + torch.mul(w[i][k][:, l_dim:].clone(),
                                                                                  weight[0, id, i])
        else:
            for i in range(len(w)):
                if i != id:
                    w_avg[k] = w_avg[k] + torch.mul(w[i][k], weight[2, id, i])

    return w_avg


def select_wei(weight1, AOU, AOU_th, num_upload, modal_w, gain,time_cost):
    with torch.no_grad():
        ###############################################确定哪些用户上传下载模型
        weight3 = F.softmax(weight1, dim=2).clone().detach()
        alpha = torch.zeros([3, num_users])
        for mm in range(3):
            AOU[mm, :] += 1
            # 挑选N个用户上传下载
            if mm < 2:
                upload_index = np.array(torch.topk((1 - torch.diag(weight3[mm, :, :])) /torch.tensor(time_cost+parm[mm]/
                    (Band * np.log2(1 + p_u * gain[:num_users] / (Band * N0)))).cuda()*torch.tensor(np.array(modal_w).T[mm]).cuda(), num_upload)[1].cpu())
            else:
                upload_index = np.array(torch.topk((1 - torch.diag(weight3[mm, :, :])) / torch.tensor(time_cost + parm[mm] /
                                            (Band * np.log2(1 + p_u * gain[:num_users] / (Band * N0)))).cuda(), num_upload)[1].cpu())
            AOU[mm, upload_index] = 0
            alpha[mm, upload_index] = 1
            if mm < 2:
                alpha[mm, :] = alpha[mm, :] * np.array(modal_w).T[mm]
            time_cost += parm[mm] / (Band * np.log2(1 + p_u * gain[:num_users] / (Band * N0))) * alpha[mm].numpy()

        for mm, uu in np.argwhere(AOU > AOU_th):  # AOU大于等于5就必须上传
            alpha[mm, uu] = 1
            AOU[mm, uu] = 0
        alpha[:2, :] = alpha[:2, :] * np.array(modal_w).T
        print(alpha)
        alpha_mat = torch.ones([3, num_users, num_users])
        ###根据是否上传，调整weight
        for m in range(3):
            for i in range(num_users):
                if alpha[m, i] == 0:
                    for j in range(num_users):
                        if j != i:
                            alpha_mat[m, i, j] = 0
                            alpha_mat[m, j, i] = 0
    return AOU, alpha, alpha_mat


def main(args):
    setup_seed(args.random_seed)

    model = AVClassifier(args)
    # model.apply(weight_init)
    model.to(device)

    # optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 10, 0.8)

    train_dataset = CramedDataset(args, mode='train')
    test_dataset = CramedDataset(args, mode='test')

    dict_train = np.load(args.train_data, allow_pickle=True).item()
    dict_test = np.load(args.test_data, allow_pickle=True).item()

    train_dataloader = []
    test_dataloader = []
    for i in range(num_users):
        train_dataloader.append(
            DataLoader(DatasetSplit(train_dataset, dict_train[i]), batch_size=args.batch_size, shuffle=True,
                       pin_memory=True))
        test_dataloader.append(
            DataLoader(DatasetSplit(test_dataset, dict_test[i]), batch_size=args.batch_size, shuffle=False,
                       pin_memory=True))

    # 模态异构
    modal_users = args.modal_users
    modal_w = []
    for i in range(num_users):
        if modal_users[i] == 1:
            modal_w.append([1, 0])  ###仅有visual
        elif modal_users[i] == 2:
            modal_w.append([0, 1])  ###仅有audio
        else:
            modal_w.append([1, 1])

    # 初始参数
    w_users_trained = []
    for i in range(num_users):
        w_users_trained.append(model.state_dict())
    server_w_agg = copy.deepcopy(w_users_trained)
    server_w_upload = copy.deepcopy(server_w_agg)

    # 保存路径
    writer_path = args.tensorboard_path
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)
    log_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(writer_path, log_name))

    ##channel
    gain = np.load('g.npy')

    # acc等指标
    lr = args.lr
    wei_lr = 0.01
    local_ep = 1
    own_acc = []
    acc = []
    t_acc = []
    train_t_acc = []

    grad_layer_name = []  # 保存有梯度的模型参数名称
    for n in model.state_dict():
        if 'running' not in n and 'batch' not in n:
            grad_layer_name.append(n)

    weight_old=torch.ones((3,num_users,num_users),requires_grad=True) ##visual, audio, share
    weight_old=weight_old.to(device)

    weight_new=copy.deepcopy(weight_old.detach())

    num_upload = args.num_upload
    AOU_th = args.AOU_th
    AOU = np.zeros([3, num_users])
    grad_wei = torch.zeros([3,num_users,num_users]).to(device)
    time_cost=np.zeros(num_users)
    iter_times=np.zeros(num_users)
    for i in range(num_users):
        iter_times[i]=(int(np.ceil(len(train_dataloader[i]) * local_ep)))
    tot_flops=np.array(modal_w).T[0]*flop[0]+np.array(modal_w).T[1]*flop[1]+flop[2]
    AOU, alpha, alpha_mat = select_wei(copy.deepcopy(weight_new), copy.deepcopy(AOU), AOU_th, num_upload, modal_w,
                                       gain[0, :],copy.deepcopy(time_cost))
    weight_list=[]
    alpha_list=[]

    for epoch in range(args.epochs):
        if (epoch+1)%20==0:
            lr=lr*0.5

        own_acc_epo = []
        acc_epo = []
        time_cost = np.zeros(num_users)
        time_cost += (iter_times*tot_flops/f_max/fpc + (alpha[0].numpy()*parm[0]+alpha[1].numpy()*parm[1]+alpha[2].numpy()*parm[2])/
                      (Band * np.log2(1 + p_u * gain[epoch,:num_users] / (Band * N0))))

        #########################################
        # 更新聚合系数
        weight_new[0,:] = weight_new[0,:] - grad_wei[0,:]/(torch.mean(torch.abs(grad_wei[0,:]))+1e-10) * wei_lr
        weight_new[1,:] = weight_new[1,:] - grad_wei[1,:]/(torch.mean(torch.abs(grad_wei[1,:]))+1e-10) * wei_lr
        weight_new[2,:] = weight_new[2,:] - grad_wei[2,:]/(torch.mean(torch.abs(grad_wei[2,:]))+1e-10) * wei_lr
        #########################################
        weight = F.softmax(weight_old, dim=2).clone()
        weight_list.append(weight.detach().cpu().numpy())

        alpha_wei = alpha_mat.cuda() / (alpha_mat.cuda() * weight.detach()).sum(dim=2).view(3, num_users, 1)  # 归一化的梯度
        weight = alpha_wei.detach() * weight

        # weight_list.append(weight.detach().cpu().numpy())
        alpha_list.append(alpha.detach().cpu().numpy())
        ############################################更新服务器的参数
        for i in range(num_users):
            for n in model.state_dict():
                server_w_upload[i][n]=copy.deepcopy(server_w_agg[i][n].detach())
                if 'visual' in n:
                    if alpha[0, i] == 1:
                        server_w_upload[i][n] = copy.deepcopy(w_users_trained[i][n].detach())
                elif 'audio' in n:
                    if alpha[1, i] == 1:
                        server_w_upload[i][n] = copy.deepcopy(w_users_trained[i][n].detach())
                elif n == 'fusion_module.fc1.weight':
                    if alpha[1, i] == 1:
                        server_w_upload[i][n][:, :l_dim] = copy.deepcopy(w_users_trained[i][n][:, :l_dim].detach())
                    if alpha[0, i] == 1:
                        server_w_upload[i][n][:, l_dim:] = copy.deepcopy(w_users_trained[i][n][:, l_dim:].detach())
                else:
                    if alpha[2, i] == 1:
                        server_w_upload[i][n] = copy.deepcopy(w_users_trained[i][n].detach())

        ##################################################
        # 聚合参数 服务器端
        server_w_agg=[]
        for i in range(num_users):
            server_w_agg.append(Weight_Avg(server_w_upload, alpha, weight, i))

        ################本地更新
        for i in range(num_users):
            ####################################终端初始参数
            w_users_agg_ini = copy.deepcopy(w_users_trained[i])
            for n in model.state_dict():
                if 'visual' in n:
                    if alpha[0, i] == 1:
                        w_users_agg_ini[n] = copy.deepcopy(server_w_agg[i][n].detach())
                elif 'audio' in n:
                    if alpha[1, i] == 1:
                        w_users_agg_ini[n] = copy.deepcopy(server_w_agg[i][n].detach())
                elif n == 'fusion_module.fc1.weight':
                    if alpha[1, i] == 1:
                        w_users_agg_ini[n][:, :l_dim] = copy.deepcopy(server_w_agg[i][n][:, :l_dim].detach())
                    if alpha[0, i] == 1:
                        w_users_agg_ini[n][:, l_dim:] = copy.deepcopy(server_w_agg[i][n][:, l_dim:].detach())
                else:
                    if alpha[2, i] == 1:
                        w_users_agg_ini[n] = copy.deepcopy(server_w_agg[i][n].detach())
            #####################################
            model.load_state_dict(w_users_agg_ini)
            acc_epo.append(valid(args, model, test_dataloader[i], modal_w[i]))
            ##模型本地更新
            batch_loss, trained_model = train_epoch(args, model, train_dataloader[i], lr, local_ep, modal_w[i])
            w_users_trained[i]=copy.deepcopy(trained_model.state_dict())
            ##聚合前模型的准确率
            own_acc_epo.append(valid(args, trained_model, test_dataloader[i], modal_w[i]))

        ###每轮的准确率合并
        avg_acc = np.array(acc_epo).mean()
        avg_acc_train = np.array(own_acc_epo).mean()
        print(np.array(own_acc_epo).round(2).T)
        print(np.array(acc_epo).round(2).T)
        print(epoch, avg_acc_train.round(4))
        if epoch==0:
            own_acc.append(acc_epo)
            writer.add_scalar('Trained Total Accuracy', avg_acc, epoch)
            train_t_acc.append(avg_acc)
        own_acc.append(own_acc_epo)
        acc.append(acc_epo)

        writer.add_scalar('Total Accuracy', avg_acc, epoch)
        writer.add_scalar('Trained Total Accuracy', avg_acc_train, epoch+1)
        t_acc.append(avg_acc)
        train_t_acc.append(avg_acc_train)

        ###得到下一轮上传下载的用户
        AOU, alpha, alpha_mat = select_wei(copy.deepcopy(weight_new), copy.deepcopy(AOU), AOU_th, num_upload,
                                           modal_w,gain[epoch, :],copy.deepcopy(time_cost))

        grad_wei = 0
        ##计算参数变化与服务器初始参数，
        for i in range(num_users):
            par_change = []
            for n in grad_layer_name:
                if 'visual' in n:
                    par_change.append(alpha[0, i] * (w_users_trained[i][n] - server_w_agg[i][n]))
                elif 'audio' in n:
                    par_change.append(alpha[1, i] * (w_users_trained[i][n] - server_w_agg[i][n]))
                elif n == 'fusion_module.fc1.weight':
                    par_change1 = alpha[1, i] * (w_users_trained[i][n][:, :l_dim] - server_w_agg[i][n][:, :l_dim])
                    par_change2 = alpha[0, i] * (w_users_trained[i][n][:, l_dim:] - server_w_agg[i][n][:, l_dim:])
                    par_change.append(torch.cat((par_change1, par_change2), dim=1))
                else:
                    par_change.append(alpha[2, i] * (w_users_trained[i][n] - server_w_agg[i][n]))


            #########聚合参数和决策的梯度
            w_users_agg_grad = []
            for n in grad_layer_name:
                w_users_agg_grad.append(server_w_agg[i][n])  # 把有梯度的拿出来作为一个list

            grad_wei += torch.autograd.grad(outputs=w_users_agg_grad,
                                            inputs=weight_old,
                                            grad_outputs=par_change,
                                            allow_unused=True,
                                            retain_graph=True)[0]

        grad_wei=grad_wei+weight_old.detach().clone()
        # grad_wei=torch.clip(grad_wei,-1000,1000)
        print(F.softmax(weight_old,dim=2).detach().cpu().numpy().round(2))
        weight_old=weight_new.clone()
        weight_old.requires_grad = True


    log_dir = os.path.join(writer_path, log_name)
    np.save(log_dir + '//t_acc.npy', np.array(t_acc))
    np.save(log_dir + '//train_t_acc.npy', np.array(train_t_acc))
    np.save(log_dir + '//weight_list.npy', np.array(weight_list))
    np.save(log_dir + '//alpha_list.npy', np.array(alpha_list))


if __name__ == "__main__":
    args = get_arguments()
    # print(args)

    # args.modal_users = np.ones([num_users]) * 3
    # args.modal_users = [1, 2, 3, 1, 3, 3]  # 1/4,1/4,1/2
    args.modal_users = [1, 1, 3, 1, 3, 3, 2, 1, 2]  # 1/3
    # args.modal_users = [1, 2, 3, 1, 3, 1, 2, 3, 3, 2, 3, 3]  # 1/4,1/4,1/2
    # args.modal_users = [1, 2, 3, 3, 2, 1, 1, 2, 3, 3, 2, 1]  # 1/3,1/3,1/3
    # args.modal_users = [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]  # 1/6,1/6,2/3

    # args.train_data = 'dict_train9_1.npy'
    # args.test_data = 'dict_test9_1.npy'
    # args.lr=2e-4
    # args.AOU_th = 10
    # args.num_upload = 3
    # main(args)


    ##############################################################
    args.train_data = 'dict_train9_2.npy'
    args.test_data = 'dict_test9_2.npy'
    args.lr=2e-4
    args.AOU_th = 10
    args.num_upload = 3
    main(args)

    #############################################################
    # args.train_data = 'dict_train9_3.npy'
    # args.test_data = 'dict_test9_3.npy'
    # args.lr = 2e-4
    # args.AOU_th = 10
    # args.num_upload = 3
    # main(args)
