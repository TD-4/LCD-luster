# _*_coding:utf-8_*_
# @auther:FelixFu
# @Date: 2021.4.14
# @github:https://github.com/felixfu520

import os
import time
import numpy as np
from tqdm import tqdm

import torch
from torchvision import transforms

from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.metrics import eval_metrics, AverageMeter, confusionMatrix


class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, train_logger=None, prefetch=True):
        """ Trainer 类
        __init__:
            1、TRANSORMS FOR VISUALIZATION
            2、预读取

        _train_epoch：

        """
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, train_logger)

        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        # 预读取
        if self.device == torch.device('cpu'):
            prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)

        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.enabled = False

    def _train_epoch(self, epoch):
        self.logger.info('\n')

        # 是否freeze_bn
        self.model.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel):
                self.model.module.freeze_bn()
            else:
                self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()  # 重置指标：loss、top1、top2,混淆矩阵
        tbar = tqdm(self.train_loader, ncols=130)
        for batch_idx, (data, target, image_path) in enumerate(tbar):
            self.data_time.update(time.time() - tic)  # 读取数据的时间

            if self.device == torch.device('cuda:0'):
                data, target = data.to(self.device), target.to(self.device)
                self.loss.to(self.device)

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)

            # 是否平均loss
            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.total_loss.update(loss.item())  # 更新loss

            # measure elapsed time
            self.batch_time.update(time.time() - tic)  # batch训练的时间
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.wrt_mode}/losses', loss.item(), self.wrt_step)

            # FOR EVAL
            topk = eval_metrics(output, target, topk=(1, 2))  # topk is tensor
            self._update_metrics(topk)

            # PRINT INFO
            tbar.set_description(
                'TRAIN ({}) | Loss: {:.3f} | Top1Acc {:.2f} Top2Acc {:.2f} | B {:.2f} D {:.2f} |'.format(
                    epoch, self.total_loss.average,
                    self.precision_top1.average.item(), self.precision_top2.average.item(),
                    self.batch_time.average, self.data_time.average))

        # METRICS TO TENSORBOARD
        self.writer.add_scalar(f'{self.wrt_mode}/top1', self.precision_top1.average.item(),
                               self.wrt_step)  # self.wrt_step
        self.writer.add_scalar(f'{self.wrt_mode}/top2', self.precision_top2.average.item(),
                               self.wrt_step)  # self.wrt_step
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'],
                                   self.wrt_step)  # self.wrt_step

        # RETURN LOSS & METRICS
        log = {'losses': self.total_loss.average,
               "top1": self.precision_top1.average.item(),
               "top2": self.precision_top2.average.item()}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            for batch_idx, (data, target, image_path) in enumerate(tbar):
                if self.device == torch.device("cuda:0"):
                    data, target = data.to(self.device), target.to(self.device)
                # LOSS
                output = self.model(data)

                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                topk = eval_metrics(output, target, topk=(1, 2))
                self._update_metrics(topk)
                self.confusion_matrix = confusionMatrix(output, target, self.confusion_matrix)  # 计算混淆矩阵

                # PRINT INFO
                tbar.set_description(
                    'EVAL ({}) | Loss: {:.3f} | Top1Acc {:.2f} Top2Acc {:.2f} |'.format(
                        epoch, self.total_loss.average,
                        self.precision_top1.average.item(), self.precision_top2.average.item()))

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/losses', self.total_loss.average, self.wrt_step)  # self.wrt_step
            self.writer.add_scalar(f'{self.wrt_mode}/top1', self.precision_top1.average.item(),
                                   self.wrt_step)  # self.wrt_step
            self.writer.add_scalar(f'{self.wrt_mode}/top2', self.precision_top2.average.item(),
                                   self.wrt_step)  # self.wrt_step

            # RETURN LOSS & METRICS
            log = {'losses': self.total_loss.average,
                   "top1": self.precision_top1.average.item(),
                   "top2": self.precision_top2.average.item()}

            # print confusion matrix
            confusion_file = open(os.path.join(self.checkpoint_dir, "confusion.txt"), 'a+')
            label_path = os.path.join(self.config["train_loader"]["args"]["data_dir"], "labels.txt")
            labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    labels.append(line.strip().split(":")[1])

            print("{0:10}".format(""), end="")  # 第一行第一个空格
            confusion_file.write("{0:8}".format(""))
            for name in labels:  # 第一行 所有的类别名称
                print("{0:10}".format(name), end="")
                confusion_file.write("{0:8}".format(name))
            print("{0:10}".format("Precision"))  # 第一行 每一类的准确率
            confusion_file.write("{0:8}\n".format("Precision"))

            # 第二行以后
            for i in range(self.train_loader.dataset.num_classes):  # 第N行
                print("{0:10}".format(labels[i]), end="")   # 第N行，第一列，名称
                confusion_file.write("{0:8}".format(labels[i]))
                for j in range(self.train_loader.dataset.num_classes):
                    if i == j:
                        print("{0:10}".format(str("-" + str(self.confusion_matrix[i][j])) + "-"), end="")
                        confusion_file.write("{0:8}".format(str("-" + str(self.confusion_matrix[i][j])) + "-"))
                    else:
                        print("{0:10}".format(str(self.confusion_matrix[i][j])), end="")
                        confusion_file.write("{0:8}".format(str(self.confusion_matrix[i][j])))
                precision = 0.0 + self.confusion_matrix[i][i] / sum(self.confusion_matrix[i])
                print("{0:.4f}".format(precision))
                confusion_file.write("{0:8}\n".format(precision))

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()  # batch训练时间
        self.data_time = AverageMeter()  # 读取数据时间
        self.total_loss = AverageMeter()
        self.precision_top1, self.precision_top2 = AverageMeter(), AverageMeter()
        self.confusion_matrix = [[0 for j in range(self.train_loader.dataset.num_classes)] for i in
                                 range(self.train_loader.dataset.num_classes)]

    def _update_metrics(self, tops):
        self.precision_top1.update(tops[0].item())
        self.precision_top2.update(tops[1].item())