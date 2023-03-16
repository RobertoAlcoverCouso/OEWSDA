import collections
from loader import Loader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from matplotlib import pyplot as plt
import numpy as np
import time
import os
from tqdm import tqdm
from networks.fcn import VGGNet, FCNs
from cfg import *
from utils import *
from networks.pspnet import PSPNet
from networks import deeplabv3_resnet101
from datetime import datetime

class Trainer:
    def __init__(self, train_set, validation_set, validate, train_args):
        self.do_val = validate
        # Build loaders for the train and test set 
        print(train_args.batch_size)
        self.build_train_data(train_set, train_args.batch_size)
        self.build_validation_data(validation_set)
        
        # Load model and build criterion and optimizer
        self.build_model(train_args)
        self.max_performance = 0
        self.show_on_val = train_args.show_on_val
        self.use_gpu = True
        self.max_performance = 0

    def build_train_data(self, train_set, batch_size):
        """
        Input:  train set.  str or dictionary.
        Output:
            train file name for logging purposes, may be skipped.
            train data loader.
        Train set configuration:
            If differents sets are ment to be employed the train set should be a 
            dictionary which has as key the dataset name and as value the
            proportion of data to be used. 
        """
        if isinstance(train_set, dict):
            self.train_file = "" 
            train_dictionary = {}
            for dataset, proportion in train_set.items():
                self.train_file += str(dataset) + "-" + str(proportion) + "_"
                dataset_csv = datasets_names[dataset] 
                train_dictionary[dataset_csv] = proportion
        else:
            train_dictionary = {datasets_names[train_set], 1}
        train_data = Loader(csv_file=train_dictionary, phase='train')
        self.train_data_loader = DataLoader(train_data, batch_size=batch_size, 
                                    shuffle=True, num_workers=6, drop_last=True)

    def build_validation_data(self, validation_set):
        """
        Input:  validation_set.  str
        Output: 
            validation file name for logging purposes
            validation data loader.
        Validation set configuration:
            If all validations sets are ment to be employed, then 
            validation_set = ""
        """
        if validation_set == "":
            self.val_file = ["Cityscapes", "Mapilliary"]
            loader_cityscapes = Loader({datasets_names["Cityscapes_Val"]: 1}, phase='test')
            loader_map = Loader({datasets_names["Mapilliary_Val"]: 1}, phase='test')
            validation_data = [loader_cityscapes, loader_map] 
        else: 
            self.val_file =validation_set
            validation_data = [Loader({datasets_names[validation_set]: 1}, phase='test')]
        
        self.validation_loaders = [DataLoader(val_data, batch_size=1, shuffle=False, 
                                   num_workers=4, drop_last=True) for val_data in validation_data]

    def build_model(self, train_args):
        """
        Input:  net_args.  Ordered dict with training parameters
        Output: 
            model dir: Directory to load and save models
            Models flags: different models have different output formats.

        Validation set configuration:
            If all validations sets are ment to be employed, then 
            validation_set = ""
        """
        self.model_dir = train_args.model_root
        architecture = train_args.architecture
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        # Flags para diferenciar el modelo. 
        self.DL3        = "deeplabv3" == architecture
        self.PSP        = "psp" == architecture
        
        if train_args.train_mode == "finetune":
            if self.DL3:
                self.finetunefile = os.path.join(self.model_dir, "DL/finetune/") 
            elif self.PSP:
                self.finetunefile = os.path.join(self.model_dir, "PSP/finetune/")
            else:
                self.finetunefile = os.path.join(self.model_dir, "FCN/finetune/")
            if not os.path.exists(self.finetunefile):
                os.makedirs(self.finetunefile)
            self.model_dir = self.finetunefile
        self.model_file = os.path.join(self.model_dir, architecture+ str(datetime.now()).replace(" ", "")+ ".pth")
        #Get model 
        if self.DL3:
            self.model  =  torch.load("pretrained/init")
            #self.model = torchvision.models.segmentation.deeplabv3_resnet101(progress=True, num_classes=20)
        elif self.PSP:
            self.model = PSPNet(n_classes=20)
        else:
            vgg_model = VGGNet(requires_grad=True, remove_fc=True)  
            with torch.cuda.device(0):
                self.model = FCNs(pretrained_net=vgg_model, n_class=20)
                self.model.backbone = self.model.backbone.cuda()
        if train_args.restore_file:
            #Load net if needed 
            if os.path.exists(train_args.restore_file):
                state_dict =  torch.load(train_args.restore_file)
                if isinstance(state_dict, collections.OrderedDict):
                    self.model.load_state_dict(state_dict)
                else:
                    self.model = state_dict
        self.model = self.model.cuda()

        self.lr = train_args.lr

        self.optimizer = optim.SGD([{'params':self.model.backbone.parameters(), 'lr':train_args.lr},
                           {'params':self.model.classifier.parameters(), 'lr':train_args.lr*10.}],
                            momentum=train_args.momentum, weight_decay=train_args.w_decay )
        if self.do_val:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.5)
        else:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5)
        
        if train_args.weighted_loss:
            weights = torch.Tensor([0, 0.77, 0.57, .74, .14, .14, .24,  .18, .36,
                                    .79, .23, .7, .33,  .21, .72,  .11, .31, .14, .1, .32])
            
            weights = weights*1./weights.sum()
            weights = 1 - weights
            self.criterion = nn.CrossEntropyLoss(ignore_index=0, weight=weights.cuda()).cuda()
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=0).cuda()

    def train(self, epochs, args):
        """
        Input:  epochs. Integer number of number of epochs to train
        Trains a model for a given number of epochs, if do_val argument is
        true, perform validation after every epoch. 
        """

        if self.do_val:
            self.val(-1)
        if not epochs:
            self.train_till_convergence(args)
            return
        for epoch in range(epochs):
            ts = time.time()
            l_tot = 0
            for iter, batch in tqdm(enumerate(self.train_data_loader)):
                self.optimizer.zero_grad()
                with torch.cuda.device(0):
                    inputs, labels = Variable(batch['X'].cuda()), Variable(batch['Y'].cuda())
                outputs = self.model(inputs)
                if self.DL3:
                    outputs = outputs["out"]
                    
                if self.PSP:

                    outputs = outputs[0]
                loss = self.criterion(outputs, labels.squeeze(1))

                loss.backward()
                self.optimizer.step()
                l_tot += loss.detach()

                
                if iter % 100 == 0 and iter > 0:
                    print("epoch{}, iter{}, loss: {}".format(epoch, iter, l_tot/iter))
            print("Finish epoch {}, time elapsed {}, loss: {}".format(epoch, time.time() - ts, l_tot/iter))
            if self.do_val:

                perf = self.val(epoch)
                self.scheduler.step(perf)
            else:
                self.scheduler.step(l_tot)
    
    def train_till_convergence(self,args):
        while args.lr/8. < self.scheduler._last_lr:
            ts = time.time()
            l_tot = 0
            for iter, batch in tqdm(enumerate(self.train_data_loader)):
                self.optimizer.zero_grad()
                with torch.cuda.device(0):
                    inputs, labels = Variable(batch['X'].cuda()), Variable(batch['Y'].cuda())

                outputs = self.model(inputs)
                if self.DL3:
                    outputs = outputs["out"]
                    
                if self.PSP:

                    outputs = outputs[0]
                loss = self.criterion(outputs, labels.squeeze(1))

                loss.backward()
                self.optimizer.step()
                l_tot += loss.detach()

                
                if iter % 100 == 0 and iter > 0:
                    print("epoch{}, iter{}, loss: {}".format(epoch, iter, l_tot/iter))
            print("Finish epoch {}, time elapsed {}, loss: {}".format(epoch, time.time() - ts, l_tot/iter))
            if self.do_val:
                perf = self.val(epoch)
                self.scheduler.step(perf)
            else:
                self.scheduler.step(l_tot)
        return
    
    def curriculum(self, epochs, args):
        """
        Curriculum learning function.
        """
        decay_idx = 0
        Training_sets_employed = {} 
        for i, subset in enumerate(args.train_set.keys()):
            for trainset, percentage in Training_sets_employed.items():
                if i>=decay_idx:
                    Training_sets_employed[trainset] = min(1, percentage*args.decay)

            if len(Training_sets_employed) == 0:
                Training_sets_employed = {datasets_names[subset]: args.initial}
            else:
                if subset not in real_datasets_train:
                    Training_sets_employed[datasets_names[subset]] = args.initial
            train_data = Loader(csv_file=Training_sets_employed,
                                phase='train')

            self.train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=6, drop_last=True)
            if i != 0:
                self.optimizer = optim.SGD([{'params':self.model.backbone.parameters(), 'lr':args.lr*args.gamma},
                                            {'params':self.model.classifier.parameters(), 'lr':args.lr*10.*args.gamma}],
                                                momentum=args.momentum, weight_decay=args.w_decay )
            self.train(epochs,args)
            if self.do_val:
                self.val(epochs*i)
            torch.save(self.model.state_dict(), self.model_path+"_"+subset+".pth")
        for trainset, percentage in Training_sets_employed.items():
            Training_sets_employed[trainset] = 1
        train_data = Loader(csv_file=Training_sets_employed,
                            phase='train')
        self.train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=6, drop_last=True)
        self.train(None,args)
        torch.save(self.model.state_dict(), self.model_path+"_CL_UDA.pth")


        train_data = Loader(csv_file={datasets_names[list(args.train_set.keys())[-1]], 1},
                            phase='train')
        self.train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=6, drop_last=True)

        self.finetune(None,args)
        torch.save(self.model.state_dict(), self.model_path+"_CL.pth")

    def finetune(self, epochs, args):
        for name, param in self.model.named_parameters():
            if self.DL3:
                if param.requires_grad and 'classifier' not in name:
                    param.requires_grad = False
            elif not self.PSP:
                if param.requires_grad and 'deconv' not in name:
                    param.requires_grad = False

        self.train(epochs,args)
        self.optimizer = optim.SGD([{'params':self.model.backbone.parameters(), 'lr':self.train_args.lr/2.5},
                                    {'params':self.model.classifier.parameters(), 'lr':self.train_args.lr*10./2.5}],
                                        momentum=self.train_args.momentum, weight_decay=self.train_args.w_decay )

        for name, param in self.model.named_parameters():
            if ('weight' in name or 'bias' in name or 'orm.' in name):    
                param.requires_grad = True

        self.train(epochs,args)

    def val(self, epoch, show=False):
        performance = 0
        with torch.no_grad():
            for i, test in enumerate(self.validation_loaders):
                total_ious = []
                pixel_accs = []
                for iter, batch in tqdm(enumerate(test)):
                    with torch.cuda.device(0):
                        if self.use_gpu:
                            inputs = Variable(batch['X'].cuda())
                        else:
                            inputs = Variable(batch['X'])
                        outputs = self.model(inputs)
                        if self.DL3:
                            outputs = outputs["out"]
                        
                        if self.PSP:
                            output = outputs[0].detach().cpu(). numpy()
                        else:
                            output = outputs.data.cpu().numpy()

                    N, _, h, w = output.shape
                    pred = output.transpose(0, 2, 3, 1).reshape(-1, 20).argmax(axis=1).reshape(N, h, w)

                    target = batch['Y'].cpu().numpy().reshape(N, h, w)
                    
                    for p, t in zip(pred, target):
                        total_ious.append(iou(p, t))
                        pixel_accs.append(pixel_acc(p, t))
                if show:
                    for j in range(len(target)):
                        plt.subplot(1,2,1)
                        plt.imshow(label_to_RGB(target[j,...]), interpolation='nearest')
                        plt.axis('off')
                        plt.subplot(1,2,2)
                        plt.imshow(label_to_RGB(pred[j,...]), interpolation='nearest')
                        plt.axis('off')
                        plt.savefig('./results/'+self.train_file+str((iter+1)*len(batch)+ i)+'.png', bbox_inches='tight')
                # Calculate average IoU
                total_ious = np.array(total_ious).T  # n_class * val_len
                ious = np.nanmean(total_ious, axis=1)
                pixel_accs = np.array(pixel_accs).mean()
                
                meanIoU = np.nanmean(ious)
                performance += meanIoU
                print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, meanIoU, ious))
            if performance >= self.max_performance and self.model_file is not None:
                self.max_performance  = performance
                torch.save(self.model, self.model_file)
        return performance
