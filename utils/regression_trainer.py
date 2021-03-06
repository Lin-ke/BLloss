from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from  models.vgg import vgg19
from datasets.crowd import Crowd
from losses.post_prob import Post_Prob


def train_collate(batch):
    # batch:tuple([],[],[]..) -> 
    transposed_batch = list(zip(*batch))
    # 按0dim进行堆叠
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes


class RegTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        args = self.args
        # if use
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            self.device = torch.device("cpu")

        self.downsample_ratio = args.downsample_ratio
        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.is_gray, x) for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=6,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}
        self.model =vgg19()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
                    # 引用：bayloss
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_count = 0
        
        self.use_bg = args.use_background
        self.e = 1e-5
        self.l = 0.01
    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_eopch(self):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode

        # Iterate over data.
        # 每次是一个batchsize的list
        #points.shape [x,2] targets.shape[x],标记点数，
        # 得到的prob_list是batchsize 的List，每一个h= 标记点数+1， w=4069,output是64*64
        # 包括：inputs(x,y),points(annotation),targets(annotation),st_sizes(?)
        for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            
            # 每张图里有多少点，points记录的是点的位置
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            # 转换为cpu或gpu类型
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]
            #对一个batch进行loss之类的计算
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                prob_list,ot_list = self.post_prob(points, st_sizes)
                loss1 = 0
                loss2 = 0
                loss3 = 0
        # enumerate(p)-> (1,p1),(2,p2),(3,p3)
                for idx, prob in enumerate(prob_list):  # iterative through each sample
                    if prob is None:  # image contains no annotation points,ot = None
                        # pre_count是这一列的求和
                        pre_count = torch.sum(outputs[idx])
        # what will ot do?
                        # target = 0
                        target = torch.zeros((1,), dtype=torch.float32, device=self.device)
                    else:
                        
                        N = len(prob) # 一张图有n个像素(或者n-1和bg)
                        if self.use_bg:
                            
                            target = torch.zeros((N,), dtype=torch.float32, device=self.device)
                            # 除了最后一个
                            target[:-1] = targets[idx]
                        else:
                            target = targets[idx]
                        # \sum density*prob
                        pre_count = torch.sum(outputs[idx].view((1, -1)) * prob, dim=1)  # flatten into vector
                        # ot_loss
                        ot_l = outputs[idx].view((1, -1))*ot_list[idx]
                        loss1 += self.l*torch.sum(ot_l)
                        loss3 += self.e*torch.sum(ot_l*(torch.log(ot_l)-1)+1)
                    #loss = |total-pred|+lambda* density* [...k,0,...]
            # target = 0/1, precount 看来是对行求和了
                    loss2 += torch.sum(torch.abs(target - pre_count)) 
                loss = loss1+loss2+loss3
                loss = loss / len(prob_list)
                
                logging.info(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                N = inputs.size(0)
                # 本次的count（将所有density求和）
                # outputs是个device：cuda tensor，numpy不能读。为了让numpy读需要.detach.cpu.numpy
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)
        
        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time()-epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models

    def val_epoch(self):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))



