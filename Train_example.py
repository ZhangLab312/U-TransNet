import torch.nn as nn
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import read_data
from tqdm import tqdm
import os
import math
import itertools
import sklearn
from U_TransNet import chromatin_encoding, U_TransNet_multi_DNA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    """
        Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=3, verbose=False, delta=0):
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

    def __call__(self, val_loss, model1,model2, path):
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
            self.save_checkpoint(val_loss, model1,model2, path)
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
            self.save_checkpoint(val_loss, model1,model2, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model1,model2, path):
        """
            Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'\nValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')
        """
            保存最优的模型Parameters
        """
        torch.save(model1.state_dict(), path+"_1.pth")
        torch.save(model2.state_dict(), path+"_2.pth")
        self.val_loss_min = val_loss

files = os.listdir("./1001")
num = 0
histone_name = ['H2AFZ','H3K4me1','H3K4me2','H3K4me3','H3K9ac','H3K9me3','H3K27ac','H3K27me3','H3K36me3','H3K79me2','H4K20me1']
cells = ['A549','GM12878','H1','HepG2','K562','HeLa-S3']
length = 1001
for name in files:
    for cell in cells:
        # print(name[-8:])
        # if os.path.exists("./1001/"+name+"/"+cell+"_pos.fasta") == False:
        #     continue
        # print(name[-8:])
        if name[-8:]=='1x.fasta':
            continue
        X, Y = read_data.Get_DNA_Sequence(cell, name, length)
        histone = read_data.Get_Histone(cell, name, histone_name, len(X), length)
        DNase = read_data.Get_DNase_Score(cell, name)
        DNase = DNase[:, np.newaxis,:].astype(np.float32)
        histone = np.concatenate((histone,DNase),1).astype(np.float32)
        TF_signal = read_data.Get_TF_signal(cell, name)
        TF_signal = TF_signal[:, np.newaxis, :]
        number = len(X) // 10
        X_train = X[ 0:10 * number ]
        Y_train = Y[ 0:10 * number ]
        X_test = X[ 9 * number:10 * number ]
        Y_test = Y[ 9 * number:10 * number ]
        X_validation = X[ 8 * number:9 * number ]
        Y_validation = Y[ 8 * number:9 * number ]
        X_test = torch.from_numpy(X_test)
        Y_test = torch.from_numpy(Y_test)
        X_train = torch.from_numpy(X_train)
        Y_train = torch.from_numpy(Y_train)
        X_validation = torch.from_numpy(X_validation)
        Y_validation = torch.from_numpy(Y_validation)

        histone_train = histone[ 0:10 * number ]
        histone_test = histone[9 * number:10 * number]
        histone_validation = histone[8 * number:9 * number]

        histone_train = torch.from_numpy(histone_train)
        histone_test = torch.from_numpy(histone_test)
        histone_validation = torch.from_numpy(histone_validation)

        TF_signal_train = TF_signal[ 0:10 * number ]
        TF_signal_test = TF_signal[9 * number:10 * number]
        TF_signal_validation = TF_signal[8 * number:9 * number]

        TF_signal_train = torch.from_numpy(TF_signal_train)
        TF_signal_test = torch.from_numpy(TF_signal_test)
        TF_signal_validation = torch.from_numpy(TF_signal_validation)

        TrainLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_train, Y_train, histone_train, TF_signal_train),
                                                  batch_size=64, shuffle=False, num_workers=0)
        TestLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_test, Y_test, histone_test, TF_signal_test),
                                                 batch_size=64, shuffle=True, num_workers=0)
        ValidationLoader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X_validation, Y_validation, histone_validation, TF_signal_validation),
                                                       batch_size=64, shuffle=True, num_workers=0)
        net = U_TransNet_multi_DNA()
        net2 = chromatin_encoding()
        net.to(device)
        net2.to(device)
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(itertools.chain(net.parameters(), net2.parameters()), lr=0.001,)
        path_model = './exp2/'+"1001"+name+cell+'.pth'
        # net.load_state_dict(torch.load(path_model+"_1"))
        # net2.load_state_dict(torch.load(path_model + "_2"))
        # net.load_state_dict(torch.load(path_model))
        early_stopping = EarlyStopping(1, True)
        flag = 1
        for epoch in range(40):
            if flag == 0:
                break
            net.train()
            correct = 0
            total = 0
            running_loss = 0.0
            ProgressBar = tqdm(TrainLoader)
            for i, data in enumerate(ProgressBar, 0):
                ProgressBar.set_description("Epoch %d" % epoch)
                inputs, labels, histone, signal = data[ 0 ].to(device), data[ 1 ].to(device), data[ 2 ].to(device), data[ 3 ].to(device)
                optimizer.zero_grad()
                res2, res3, res4 = net2(histone)
                outputs = net(inputs, res2, res3, res4)
                loss = loss_function(outputs.to(device), signal.to(device))
                ProgressBar.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()
                # predicted = torch.round(outputs)
                running_loss += loss.item()
                # a = predicted[1]
                # correct += (predicted == labels).sum().item()
                # correct += (predicted[ :, 1 ] == labels).sum().item()
                # if i % 50 == 49:  # print every 2000 mini-batches
                if i == 0:
                    aver_t = signal.max(2).values.squeeze()#.max(1).values
                    aver_p = outputs.max(2).values.squeeze()
                else:
                    aver_t = torch.cat([aver_t, signal.max(2).values.squeeze()], 0)
                    aver_p = torch.cat([aver_p, outputs.max(2).values.squeeze()], 0)

            print('train [%d] loss: %.3f' %
                      (epoch + 1, running_loss / (i + 1)))
            running_loss = 0.0
            net.eval()
            a = np.corrcoef(aver_t.cpu().detach().numpy(), aver_p.cpu().detach().numpy())
            print('train pearson correlation ',a[0][1])
            if epoch >= 12:
                # torch.save(net.state_dict(), path_model+'_1')
                # torch.save(net2.state_dict(), path_model+"_2")
                plt.scatter(aver_t.cpu().detach().numpy(), aver_p.cpu().detach().numpy(),1)
                plt.xlim(aver_p.min().cpu().detach().numpy(),aver_p.max().cpu().detach().numpy())
                plt.ylim(aver_p.min().cpu().detach().numpy(),aver_p.max().cpu().detach().numpy())
                plt.show()
            net.eval()
            net2.eval()

            with torch.no_grad():
                for m, data in enumerate(ValidationLoader, 0):
                    inputs, labels, histone, signal = data[0].to(device), data[1].to(device), data[2].to(device), data[
                        3].to(device)
                    res2, res3, res4 = net2(histone)
                    outputs = net(inputs, res2, res3, res4)

                    loss = loss_function(outputs.to(device), signal.to(device))
                    running_loss += loss.item()
                    if m == 0:
                        aver_t = signal.max(2).values.squeeze()
                        aver_p = outputs.max(2).values.squeeze()
                    else:
                        aver_t = torch.cat([aver_t, signal.max(2).values.squeeze()], 0)
                        aver_p = torch.cat([aver_p, outputs.max(2).values.squeeze()], 0)
                flag = early_stopping(running_loss / (m + 1), net, net2, path_model)
                print('validate [%d] loss: %.3f' %
                      (epoch + 1, running_loss / (m + 1)))
                b = np.corrcoef(aver_t.cpu().detach().numpy(), aver_p.cpu().detach().numpy())
                print('validate pearson correlation ', b[0][1])
                if epoch >= 12:
                    plt.scatter(aver_t.cpu().detach().numpy(), aver_p.cpu().detach().numpy(), 1)
                    plt.xlim(aver_p.min().cpu().detach().numpy(), aver_p.max().cpu().detach().numpy())
                    plt.ylim(aver_p.min().cpu().detach().numpy(), aver_p.max().cpu().detach().numpy())
                    plt.show()
            with torch.no_grad():
                for m, data in enumerate(TestLoader, 0):
                    inputs, labels, histone, signal = data[0].to(device), data[1].to(device), data[2].to(device), \
                                                      data[
                                                          3].to(device)
                    res2, res3, res4 = net2(histone)
                    outputs = net(inputs, res2, res3, res4)

                    loss = loss_function(outputs.to(device), signal.to(device))
                    running_loss += loss.item()
                    if m == 0:
                        aver_t = signal.max(2).values.squeeze()
                        aver_p = outputs.max(2).values.squeeze()
                    else:
                        aver_t = torch.cat([aver_t, signal.max(2).values.squeeze()], 0)
                        aver_p = torch.cat([aver_p, outputs.max(2).values.squeeze()], 0)
                flag = early_stopping(running_loss / (m + 1), net, net2, path_model)
                print('test [%d] loss: %.3f' %
                      (epoch + 1, running_loss / (m + 1)))
                b = np.corrcoef(aver_t.cpu().detach().numpy(), aver_p.cpu().detach().numpy())
                print('test pearson correlation ', b[0][1])
                if epoch >= 12:
                    plt.scatter(aver_t.cpu().detach().numpy(), aver_p.cpu().detach().numpy(), 1)
                    plt.xlim(aver_p.min().cpu().detach().numpy(), aver_p.max().cpu().detach().numpy())
                    plt.ylim(aver_p.min().cpu().detach().numpy(), aver_p.max().cpu().detach().numpy())
                    plt.show()
                # print(outputs,predicted[:, 0], labels)
        # torch.save(net.state_dict(), path_model)
        # print('Accuracy of the network on the  test: %.3f %%' % ((
        #         100 * correct / total)))
        running_loss = 0.0

