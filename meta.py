import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
from tqdm import tqdm
from util import make_functional
from U_TransNet_meta import U_TransNet

import read_data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    """
        Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float):  Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model1, path):
        """
            定义 __call__ 函数 -> 将一个类视作一个函数
            该函数的目的 类似在class中重载()运算符
            使得这个类的实例对象可以和普通函数一样 call
            即，通过 对象名() 的形式使用
        """

        score = -val_loss

        if self.best_score is None:
            """
                初始化（第一次call EarlyStopping）
                保存检查点
            """
            self.best_score = score
            self.save_checkpoint(val_loss, model1, path)
        elif score < self.best_score + self.delta:
            """
                验证集损失没有继续下降时，计数
                当计数 大于 耐心值时，停止
                注：
                    由于模型性能没有改善，此时是不保存检查点的
            """
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return 0
        else:
            """
                验证集损失下降了，此时从头开始计数
                保存检查点
            """
            self.best_score = score
            self.save_checkpoint(val_loss, model1, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model1, path):
        """
            Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
        """
            保存最优的模型Parameters
        """
        torch.save(model1.state_dict(), path+".pth")

        self.val_loss_min = val_loss
length =1001
histone_name = ['H2AFZ','H3K4me1','H3K4me2','H3K4me3','H3K9ac','H3K9me3','H3K27ac','H3K27me3','H3K36me3','H3K79me2','H4K20me1']
cell = '1001'
name = 'A549'
class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args['update_lr']
        self.meta_lr = args['meta_lr']
        self.update_step = args['update_step']
        self.update_step_test = args['update_step_test']


        self.net = U-TransNet()
        self.meta_optim = optim.SGD(self.net.parameters(), lr=self.meta_lr)


    def reverse(self,x_spt, y_spt, x_qry, y_qry):
        x_spt = torch.transpose(x_spt, 0, 1)
        y_spt = torch.transpose(y_spt, 0, 1)
        x_qry = torch.transpose(x_qry, 0, 1)
        y_qry = torch.transpose(y_qry, 0, 1)
        return x_spt, y_spt, x_qry, y_qry

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry, x_test, y_test):
        """
        :param x_spt:   [batch, channel, length]
        :param y_spt:   [batch, task_num, length]
        :param x_qry:   [batch, channel, length]
        :param y_qry:   [batch, task_num, length]
        :param x_test:   [batch, channel, length]
        :param y_test:   [batch, task_num, length]
        :return:
        """
        # x_spt, y_spt, x_qry, y_qry = self.reverse(x_spt, y_spt, x_qry, y_qry)
        setsz, c_, l = x_spt.size()
        _,task_num,_ = y_qry.size()
        testsz = x_test.size(0)
        early_stopping = EarlyStopping(100, True)

        TrainLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(x_spt, y_spt),
                                                  batch_size=64, shuffle=True, num_workers=0, drop_last=True)
        ValidateLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(x_qry, y_qry),
                                                 batch_size=64, shuffle=True, num_workers=0)
        TestLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(x_test, y_test),
                                                 batch_size=64, shuffle=True, num_workers=0)
        flag = 1
        for epoch in range(1000):
            if flag == 0:
                break
            ProgressBar = tqdm(TrainLoader)
            corrects = [0 for _ in range(task_num + 1)]
            # aver_t = torch.rand(task_num, x_spt.size(0), 1001).to(device)
            # aver_p = torch.rand(task_num, x_spt.size(0), 1001).to(device)
            for ff, data in enumerate(ProgressBar, 0):
                loss = 0
                self.net.train()
                for i in range(task_num):
                    self.net.to(device)
                    inputs, labels = data[ 0 ].to(device), data[ 1 ].to(device)
                    x_qry, y_qry = ValidateLoader.dataset[np.random.choice(range(len(TestLoader.dataset)), 64)]
                    x_qry = x_qry.to(device)
                    y_qry = y_qry.to(device)
                    logits = self.net(inputs.to(device))
                    loss = nn.MSELoss()(logits.squeeze().to(device), labels[:, i, ].to(device))
                    grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

                    f_model = make_functional(self.net)

                    q_res = f_model(x_qry, params=fast_weights)
                    q_loss = F.mse_loss(q_res.squeeze().to(device), y_qry[:,i,].to(device))
                    loss = q_loss if loss == 0 else loss + q_loss



                    # [setsz]
                    # predicted = torch.round(q_res).squeeze()
                    # correct = (predicted.to(device) == y_qry[:,i].to(device)).sum().item()
                    # corrects[0] = corrects[0] + correct
                    # corrects[i] = torch.cat([corrects[i], y_qry.max(2).values.squeeze()], 0)
                    # with torch.no_grad():
                    #     if i == 0:
                    #         if ff == 0 :
                    #             aver_t = y_qry[:,i,].max(1).values.squeeze()  # .max(1).values
                    #             aver_p = q_res.max(2).values.squeeze()
                    #         else:
                    #             aver_t = torch.cat([aver_t, y_qry[:,i,].max(1).values.squeeze()], 0)
                    #             aver_p = torch.cat([aver_p, q_res.max(2).values.squeeze()], 0)
                self.meta_optim.zero_grad()
                ProgressBar.set_postfix(loss=loss.item())
                loss.backward()
                self.meta_optim.step()
                # flag = early_stopping(loss, self.net, './model/U-TransNet')
            # accs = np.array(corrects[0]) / (setsz * task_num)
            # a = np.corrcoef(aver_t.cpu().detach().numpy(), aver_p.cpu().detach().numpy())
            # if epoch % 20 == 0:
            # print('train pearson correlation ', a[0][1])
            # for num in task_num:
            #
            self.net.eval()
            loss = 0
            corrects = [0 for _ in range(task_num + 1)]
            # aver_t = torch.rand(task_num, x_test.size(0), 1001).to(device)
            # aver_p = torch.rand(task_num, x_test.size(0), 1001).to(device)
            with torch.no_grad():
                for ff, data in enumerate(TrainLoader, 0):
                # for valid_samples, valid_labels in TestLoader:  #
                    valid_samples, valid_labels = data[0], data[1],
                    for i in range(task_num):
                        t_res = self.net(valid_samples.to(device))
                        q_loss = F.mse_loss(t_res.squeeze().to(device), valid_labels[:, i, ].to(device))
                        loss = q_loss if loss == 0 else loss + q_loss
                #         if i == 0:
                #             if ff == 0:
                #                 aver_t = valid_labels[:,i,].max(1).values.squeeze()  # .max(1).values
                #                 aver_p = t_res.max(2).values.squeeze()
                #             else:
                #                 aver_t = torch.cat([aver_t, valid_labels[:,i,].max(1).values.squeeze()], 0)
                #                 aver_p = torch.cat([aver_p, t_res.max(2).values.squeeze()], 0)
                # a = np.corrcoef(aver_t.cpu().detach().numpy(), aver_p.cpu().detach().numpy())
                # print('test pearson correlation ', a[0][1])
                # for num in task_num:
                #     a = np.corrcoef(aver_t[num,].cpu().detach().numpy(), aver_p[num,].cpu().detach().numpy())
                #     # if epoch %20 == 0:
                #     print('test pearson correlation ', a[0][1])


                flag = early_stopping(loss, self.net, './model/U-TransNet')
            # acc.append(accs)
        # print(np.mean(acc))
        return 0


