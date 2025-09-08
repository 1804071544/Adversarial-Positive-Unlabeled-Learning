from other.data_utils.dataset import TrainDataset, TestDataset, TestDataset4HS
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import fnmatch
import datetime
import logging
import shutil
from tqdm import tqdm
from network.unet import Unet
from network.dinknet5 import DinkNet34
from network.freeocnet import FreeOCNet
from other.data_utils.dataloader import HOCDataLoader
from loss_function.TaylorVarPULoss import TaylorVarPULoss

from other.cal_entropy import cal_entropy
from parameters import DEVICE, SET_NAME, LABEL_RATE, EXTRA_WORD, sever_root, BATCH_SIZE, PRE_EPOCH, START_EPOCH, \
    END_EPOCH, test_path, base_root, GAP_EPOCH, data_root_path, model_paths, last_path,Pior
from parameters import config
from other.classmap2rgbmap import classmap2rgbmap, pro2grayim
from other.acc import all_metric


class base_trainer:
    def __init__(self, train_flag=True):
        self.data_root_path = data_root_path
        self.test_path = test_path
        self.train_flag = train_flag
        ####################定义路径和损失####################
        ##########定义训练信息##########
        self.start_epoch = START_EPOCH  # 开始时的已训练epoch数
        self.end_epoch = END_EPOCH  # 结束时的已训练epoch数
        self.pre_epoch = PRE_EPOCH  # 训练开始时的预训练epoch数
        self.gap_epoch = GAP_EPOCH  # 保存的epoch间隔
        self.config = config
        self.write_mode = 'w'
        self.best_F1 = 0
        self.save_path = os.path.join(base_root, 'log', SET_NAME)
        self.check_save_path()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net_g, self.net_d1, self.net_d2 = self.built_network()  # 生成对抗网络

    def built_network(self):
        if self.train_flag:
            logging.basicConfig(filename=os.path.join(self.save_path, "log.log"),  # 将日志保存到filename文件中
                                filemode='w',  # 写入的模式
                                level=logging.DEBUG)  # DEBUG及以上的日志信息都会显示

        # 定义源文件和目标文件的路径
        if self.train_flag:
            src_path = r"D:\MasterProgram\paper2\MyNET\parameters.py"
            dst_path = os.path.join(self.save_path, "parameters.py")
            shutil.copy(src_path, dst_path)  # 复制本次的超参数
        # 模型参数
        in_channel = self.config['model']['params']['in_channels']
        if self.start_epoch == 0:  # 如果是从头开始训练
            if self.pre_epoch == 0:  # 如果resnet没有预训练
                params = self.config['model']['params']
                net_g = FreeOCNet(**params)  # 定义生成网络

            else:  # resnet有预训练
                params = self.config['model']['params']
                net_g = FreeOCNet(**params)  # 定义生成网络
                net_g.load_state_dict(torch.load(model_paths))
                print('load pre_train net_{}'.format(self.pre_epoch))  # 加载生成网络

            net_d1 = Unet(in_channel=in_channel + 1, out_channel=1)  # 定义判别网络1
            net_d2 = Unet(in_channel=in_channel + 1, out_channel=1)  # 定义判别网络2
        else:  # 如果是接着训练
            self.write_mode = 'a'  # 日志模式为添加

            params = self.config['model']['params']
            net_g = FreeOCNet(**params)  # 定义生成网络
            net_d1 = Unet(in_channel=in_channel + 1, out_channel=1)  # 定义判别网络1
            net_d2 = Unet(in_channel=in_channel + 1, out_channel=1)  # 定义判别网络2

        net_g.to(self.device)
        net_d1.to(self.device)
        net_d2.to(self.device)
        return net_g, net_d1, net_d2

    def check_save_path(self):

        if last_path is None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            file_list = os.listdir(self.save_path)
            ex_index = len(file_list) + 1
            pack_name = 'ex' + str(ex_index) + '_' + current_time
            self.save_path = os.path.join(self.save_path, pack_name)
            if self.train_flag:
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
        else:
            self.save_path = last_path

    def check_point(self, f1, epoch):
        if epoch % 20 == 0:
            torch.save(self.net_g.state_dict(), os.path.join(self.save_path, str(epoch) + '_net_g.pth'))
        if f1 > self.best_F1:
            self.best_F1 = f1
            save_best_model_path = os.path.join(self.save_path, 'best_f1')
            if not os.path.exists(save_best_model_path):
                os.makedirs(save_best_model_path)
            torch.save(self.net_g.state_dict(), os.path.join(save_best_model_path, str(epoch) + '_net_g.pth'))

    def loss1(self):
        # #########loss1##########
        out_img = self.net_g(self.image).to(self.device)  # 预测图像
        data_mask = self.positive_train_mask  # 只有正样本是有标签
        # ####D1#####
        # 判别网络1的预测，每个像素值表示是有监督图像的概率
        rate1_img = self.net_d1(torch.cat((self.image, out_img.detach()), 1).detach().to(self.device))  # 预测是否有监督结果
        loss1_d = self.loss_BCE_weight(rate1_img, data_mask)  # 判别网络预测结果与真实数据类型之间的损失

        self.optimizer_d1.zero_grad()
        (self.d1_weight * loss1_d).backward()
        self.optimizer_d1.step()
        # self.lr_scheduler_d1.step()
        loss1_d_val = loss1_d.item()  # 当前batch的loss1_d损失
        self.loss1_d_epoch += loss1_d_val  # 当前batch的loss1_d损失加入判别器1的epoch损失

        #####G1#####
        rate1_img = self.net_d1(torch.cat((self.image, out_img), 1).to(self.device))  # 预测是否有监督结果
        loss1_g = self.loss_BCE_weight(rate1_img, (1 - data_mask))  # 判别网络预测结果与虚假数据类型之间的损失
        # loss_s, p_loss_s, u_loss_s = self.loss_t(rate1_img, self.positive_train_mask, self.unlabeled_train_mask,
        #                                          self.epoch, self.device)
        self.optimizer_g.zero_grad()
        (loss1_g).backward()
        self.optimizer_g.step()
        self.lr_scheduler_d1.step()
        loss1_g_val = loss1_g.item()
        self.loss1_g_epoch += loss1_g_val

        #####.#####
        if loss1_d_val < 0.1 * loss1_g_val:  # 如果判别器损失太小则更改鉴别器权重
            self.d1_weight = 0.1
        else:
            self.d1_weight = 1

    def loss2(self):
        ##########loss2##########
        out_img = self.net_g(self.image).to(self.device)  # 道路预测图像
        entropy_img = cal_entropy(out_img).to(self.device)  # 生成图像的熵图
        data_mask = self.positive_train_mask  # 只有正样本是有标签
        #####D2#####
        rate2_img = self.net_d2(torch.cat((self.image, entropy_img), 1).detach().to(self.device))  # 预测是否有监督结果
        loss2_d = self.loss_BCE_weight(rate2_img, data_mask)  # 判别网络预测结果与真实数据类型之间的损失

        self.optimizer_d2.zero_grad()
        (self.d1_weight * loss2_d).backward()

        self.optimizer_d2.step()
        self.lr_scheduler_d2.step()

        loss2_d_val = loss2_d.item()
        self.loss2_d_epoch += loss2_d_val
        #####.#####

        #####G2#####
        rate2_img = self.net_d2(torch.cat((self.image, entropy_img), 1).to(self.device))  # 预测是否有监督结果
        loss2_g = self.loss_BCE_weight(rate2_img, (1 - data_mask))  # 判别网络预测结果与虚假数据类型之间的损失
        #loss_s, p_loss_s, u_loss_s = self.loss_t(rate2_img, self.positive_train_mask, self.unlabeled_train_mask,
        #self.epoch, self.device)
        self.optimizer_g.zero_grad()
        (loss2_g).backward()
        self.optimizer_g.step()

        loss2_g_val = loss2_g.item()
        self.loss2_g_epoch += loss2_g_val
        #####.#####
        if loss2_d_val < 0.1 * loss2_g_val:  # 如果判别器损失太小则更改鉴别器权重
            self.d2_weight = 0.1
        else:
            self.d2_weight = 1

    def loss3(self):
        out_img = self.net_g(self.image).to(self.device)  # 预测
        self.optimizer_g.zero_grad()
        # loss_f = absNegative(Pior)
        # loss_s, estimated_p_loss, estimated_n_loss, estimated_u_n_loss, estimated_p_n_loss=self.loss_t(out_img, self.positive_train_mask, self.unlabeled_train_mask,
        #                                          self.epoch)
        # loss_s, p_loss_s, u_loss_s = self.loss_t(out_img, self.positive_train_mask, self.unlabeled_train_mask,
        #                                          self.epoch,self.device)
        data_mask = self.positive_train_mask  # 只有正样本是有标签
        loss_s = self.loss_BCE_weight(out_img, data_mask)
        loss_s.backward()
        self.optimizer_g.step()
        self.lr_scheduler_g.step()

        loss3_bce_val = loss_s.item()
        self.loss3_bce_epoch += loss3_bce_val

    def evaluate_fn(self, model, test_dataloader, cls, epoch):
        save_rgb_path = os.path.join(self.save_path, 'classmap')
        if not os.path.exists(save_rgb_path):
            os.makedirs(save_rgb_path)
        meta = self.config['meta']
        model.eval()
        auc, fpr, tpr, threshold, pre, rec, f1 = 0, 0, 0, 0, 0, 0, 0
        num = 0
        with torch.no_grad():
            for (im, mask) in test_dataloader:
                num = num + 1
                im = im.to(self.device)
                pred_pro = model(im).squeeze().cpu()
                pred_class = torch.where(pred_pro > 0.5, 1, 0)

                cls_fig = classmap2rgbmap(
                    pred_class.numpy(),
                    palette=meta['palette'], cls=cls)
                gray_image = pro2grayim(pred_pro.numpy())
                plt.imsave(os.path.join(save_rgb_path, str(epoch) + '.png'), gray_image, cmap='gray')
                #cls_fig.save(os.path.join(save_rgb_path, str(epoch) + "_" + str(num) + '.png'))
                auc_, fpr_, tpr_, threshold_, pre_, rec_, f1_ = all_metric(pred_pro, pred_class, mask[0, :, :])
                auc = auc + auc_
                pre = pre + pre_
                rec = rec + rec_
                f1 = f1 + f1_

        # model.train()
        return auc / num, fpr / num, tpr / num, threshold / num, pre / num, rec / num, f1 / num

    def train(self):
        train_data_set = TrainDataset(self.config, self.data_root_path)
        train_dataloader = HOCDataLoader(train_data_set)

        test_data_set = TestDataset(self.config, self.data_root_path)
        # test_data_set = TestDataset4HS(self.config, self.test_path)
        test_dataloader = HOCDataLoader(test_data_set, shuffle=False)
        ##########损失函数和优化器##########
        loss_val_best = 10000  # 初始化生成器的最佳验证损失
        loss_val_up_num = 0  # 初始化生成器验证损失连续超过最佳验证损失的代数
        # self.loss_t = absNegative(prior=Pior).to(self.device)
        # self.loss_t =TaylorVarPULoss().to(self.device)
        loss_bce = nn.BCELoss().to(self.device)  # BCE损失
        learn_rate_g = 0.001  # 生成器初始学习率
        learn_rate_d1 = 0.001  # 判别器1初始学习率
        learn_rate_d2 = 0.001  # 判别器2初始学习率
        weight_decay = 1e-4

        (loss1_g_val, loss1_d_val, loss2_g_val, loss2_d_val, loss3_bce_val, loss3_dice_val, loss3_dice_unweight_val,
         loss4_val, loss5_val) = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 初始化loss1、2、3、4、5输出值

        # self.optimizer_g = optim.Adam(self.net_g.parameters(), lr=learn_rate_g, weight_decay=weight_decay,
        #                               betas=(0.9, 0.99))  # 生成网络优化器
        # self.optimizer_d1 = optim.Adam(self.net_d1.parameters(), lr=learn_rate_d1, weight_decay=weight_decay,
        #                                betas=(0.9, 0.99))  # 判别网络1优化器
        # self.optimizer_d2 = optim.Adam(self.net_d2.parameters(), lr=learn_rate_d2, weight_decay=weight_decay,
        #                                betas=(0.9, 0.99))  # 判别网络2优化器
        self.optimizer_g = optim.SGD(self.net_g.parameters(), lr=learn_rate_g, weight_decay=weight_decay, momentum=0.9
                                     )  # 生成网络优化器
        self.optimizer_d1 = optim.SGD(self.net_d1.parameters(), lr=learn_rate_d1, weight_decay=weight_decay,
                                      momentum=0.9
                                      )  # 判别网络1优化器
        self.optimizer_d2 = optim.SGD(self.net_d2.parameters(), lr=learn_rate_d2, weight_decay=weight_decay,
                                      momentum=0.9
                                      )  # 判别网络2优化器

        # self.lr_scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_g, mode='min', factor=0.1,
        #                                                                  patience=10, verbose=False,
        #                                                                  threshold=0.0001, threshold_mode='rel',
        #                                                                  cooldown=0, min_1r=0,
        #                                                                  eps=1e-08)

        self.lr_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_g, gamma=0.999)
        self.lr_scheduler_d1 = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d1, gamma=0.995)
        self.lr_scheduler_d2 = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_d2, gamma=0.995)

        ####################迭代训练####################
        self.net_g.train()
        self.net_d1.train()
        self.net_d2.train()
        bar = tqdm(range(self.start_epoch + 1, self.end_epoch + 1))
        for epoch in bar:  # 迭代训练
            print(1)
            self.epoch = epoch
            self.loss1_g_epoch = 0.0  # 初始化生成器当前epoch的loss1损失
            self.loss1_d_epoch = 0.0  # 初始化判别器1当前epoch的loss1损失
            self.loss2_g_epoch = 0.0  # 初始化生成器当前epoch的loss2损失
            self.loss2_d_epoch = 0.0  # 初始化判别器2当前epoch的loss2损失
            self.loss3_bce_epoch = 0.0  # 初始化loss3当前epoch的bce损失
            self.loss3_dice_unweight_epoch_list = []  # 初始化当前epoch的平均不加权dice损失
            self.loss3_dice_epoch = 0.0  # 初始化loss3当前epoch的dice损失
            # self.loss4_epoch = 0.0  # 初始化生成器当前epoch的loss4损失
            # self.loss5_epoch = 0.0  # 初始化生成器当前epoch的loss5损失

            self.d1_weight = 1  # 初始化鉴别器权重
            self.d2_weight = 1  # 初始化鉴别器权重
            for batch, batch_data in enumerate(train_dataloader, 1):  # batch从1开始增加

                actual_batch_size = train_data_set.actual_batch_size
                self.image = batch_data[0].repeat(actual_batch_size, 1, 1, 1).to(self.device)  # batch,chanel,w,h
                self.positive_train_mask = batch_data[1].permute(1, 0, 2, 3).to(self.device)
                self.unlabeled_train_mask = batch_data[2].permute(1, 0, 2, 3).to(self.device)
                # 参与计算loss的像素比例
                pixels_rate = (self.positive_train_mask.sum() + self.unlabeled_train_mask.sum())
                pixels_rate = pixels_rate / self.positive_train_mask.nbytes * self.positive_train_mask.itemsize
                p_u_rate = self.positive_train_mask.sum() / self.unlabeled_train_mask.sum()  # 正例与未标记的比例
                data_weight = (self.positive_train_mask / p_u_rate / pixels_rate +
                               self.unlabeled_train_mask * p_u_rate / pixels_rate)
                # data_weight = self.positive_train_mask.clone()
                # data_weight[data_weight == 0] = 0.0005
                self.loss_BCE_weight = nn.BCELoss(weight=data_weight).to(self.device)  # BCE损失进行加权

                self.loss3()
                if epoch % GAP_EPOCH == 0:
                    self.loss1()
                    self.loss2()
            # epoch 训练结束
            auc, fpr, tpr, threshold, pre, rec, f1 = self.evaluate_fn(self.net_g, test_dataloader, test_data_set.cls,
                                                                      epoch)

            logging_string = "{} epoch, Loss1_g {:.4f}, Loss1_d {:.4f}, Loss2_g {:.4f}, Loss2_d {:.4f}," \
                             " Loss3_Tay {:.4f}, AUC {:.6f}, Precision {:.6f}, Recall {:.6f}, " \
                             "F1 {:.6f}, Best-F1 {:.6f}".format(epoch, self.loss1_g_epoch, self.loss1_d_epoch,
                                                                self.loss2_g_epoch, self.loss2_d_epoch,
                                                                self.loss3_bce_epoch, auc,
                                                                pre, rec, f1, self.best_F1)
            bar.set_description(logging_string)
            logging.info(logging_string)
            self.check_point(f1, epoch)
            # print('Loss1_g:', self.loss1_g_epoch, 'Loss1_d:', self.loss1_d_epoch, 'Loss2_g:', self.loss2_g_epoch,
            #       'Loss2_d:', self.loss2_d_epoch, 'Loss3_Tay:', self.loss3_bce_epoch)
            # print('AUC:', auc, 'Precision:', pre, 'Recall:', rec, 'F1-score:', f1)
        print('END')
