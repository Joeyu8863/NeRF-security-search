from __future__ import print_function
import numpy as np
import pandas as pd
import torch.nn as nn
import math
import torch.nn.functional as F
import torch

from torch.nn import init
from collections import OrderedDict
import time
import shutil
import xlwt
from xlwt import Workbook 
import torchvision
import argparse
import torch.optim as optim
from torchvision import datasets, transforms
# from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from xlwt import Workbook 
import torch as th
from module import quan_Conv2d,quan_Linear
import operator

def validate1(model, data,target, criterion, val_loader, epoch,xn):

    "this function computes the accuracy for a given data and target on model"    
    model.eval()
    test_loss = 0
    correct = 0
    preds=torch.zeros([10000]) 
    with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            output,_ = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #print(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    
    test_loss /= len(test_loader.dataset)
    print('\nSubTest set: Average loss: {:.4f}, Attack Success Rate: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, xn,
        100. * correct /xn))
    
    return (100. * correct /xn)


def validate(model, fn,device, criterion, data,dir,target, epoch):  
    "test function" 
    model.eval()
    test_loss = 0
    correct = 0
    preds=torch.zeros([10000]) 
    with torch.no_grad():
        #for data,dir in enumerate(val_loader):
	    #data, target = data.cuda(), target.cuda()
        print(type(data))
        print(type(dir))
	#output,_ = model(data)
        output = fn(data.to('cpu'), dir.to('cpu'), model.to('cpu'))
        test_loss += criterion(output, target).item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        #print(pred,target)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, val_loader.sampler.__len__(),
        100. * correct / val_loader.sampler.__len__() ))
    
    return test_loss, 100. * correct / val_loader.sampler.__len__()


def validate2(model,fn, device, criterion, val_loader, target,epoch):
    "this function computes the attack success rate of  all data to target class toattack"    
    model.eval()
    test_loss = 0
    correct = 0
    preds=torch.zeros([10000]) 
    with torch.no_grad():
        for data,dir in enumerate(val_loader):
            #data, target = data.cuda(), target.cuda()
            target[:]=1
            #output,_ = model(data)
            output = fn(data,dir,model)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #print(pred,target)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: loss: {:.4f}, Attack success rate: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, val_loader.sampler.__len__(),
        100. * correct / val_loader.sampler.__len__() ))
    
    return test_loss, 100. * correct / val_loader.sampler.__len__()


def validate3(model, device, criterion, data1,target1, epoch):   
    "this function computes the accuracy of a given data1 and target1 batchwise"
    model.eval()
    test_loss = 0
    correct = 0
    n=0
    m=100
    preds=torch.zeros([10000]) 
    with torch.no_grad():
        for i in range((9000-args.testsamp)//100):
            data, target = data1[n:m,:,:,:].cuda(), target1[n:m].cuda()
            m+=100
            n+=100
            output,_ = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #print(pred,target)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    
    test_loss /= 9000-args.testsamp
    print('\nSub Test set: loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, (9000-args.testsamp) ,
        100. * correct / (9000-args.testsamp)  ))
    
    return test_loss, 100. * correct / (9000-args.testsamp)    
def int2bin(input, num_bits):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    '''
    output = input.clone()
    output[input.lt(0)] = 2**num_bits + output[input.lt(0)]
  
    return output


def bin2int(input, num_bits):
    '''
    convert the unsigned integer (2's complement equivantly) back to the signed integer format
    with the bitwise operations. Note that, in order to perform the bitwise operation, the input
    tensor has to be in the integer format.
    '''
    mask = 2**(num_bits-1) - 1
    output = -(input & ~mask) + (input & mask)
    return output


def weight_conversion(model):
    '''
    Perform the weight data type conversion between:
        signed integer <==> two's complement (unsigned integer)

    Note that, the data type conversion chosen is depend on the bits:
        N_bits <= 8   .char()   --> torch.CharTensor(), 8-bit signed integer
        N_bits <= 16  .short()  --> torch.shortTensor(), 16 bit signed integer
        N_bits <= 32  .int()    --> torch.IntTensor(), 32 bit signed integer
    '''
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            w_bin = int2bin(m.weight.data, m.N_bits).char()
            
            m.weight.data = bin2int(w_bin, m.N_bits).float()
    return

class BFA1(object):
    def __init__(self, criterion, fc, k_top=10):

        self.criterion = criterion
        # init a loss_dict to log the loss w.r.t each layer
        self.loss_dict = {}
        self.N_bits = 8
        self.bit_counter = 0
        self.k_top = k_top
        self.n_bits2flip = 0
        self.loss = 0
        self.fc=fc  
        self.b_w = nn.Parameter(
            2**torch.arange(start=self.N_bits-1,end=-1, step=-1).unsqueeze(-1).float(),
            requires_grad = False)
        
        self.b_w[0] = -self.b_w[0]
    def flip_bit(self, m,offs):
        '''
        the data type of input param is 32-bit floating, then return the data should
        be in the same data_type.
        '''
        self.k_top=  m.weight.grad.detach().abs().view(-1).size()[0]
        
        # 1. flatten the gradient tensor to perform topk
        w_grad_topk, w_idx_topk = m.weight.grad.detach().abs().view(-1).topk(
            self.k_top)
        # update the b_grad to its signed representation
        w_grad_topk = m.weight.grad.detach().view(-1)[w_idx_topk]
        
        #print(w_grad_topk)
        # 2. create the b_grad matrix in shape of [N_bits, k_top]
        b_grad_topk = w_grad_topk * self.b_w.data

        # 3. generate the gradient mask to zero-out the bit-gradient
        # which can not be flipped
        b_grad_topk_sign = (b_grad_topk.sign() +
                            1) * 0.5  # zero -> negative, one -> positive
        # convert to twos complement into unsigned integer
        w_bin = int2bin(m.weight.detach().view(-1), 8).short()
        w_bin_topk = w_bin[w_idx_topk]  # get the weights whose grads are topk
        # generate two's complement bit-map
        b_bin_topk = (w_bin_topk.repeat(8,1) & self.b_w.abs().repeat(1,self.k_top).short()) \
        // self.b_w.abs().repeat(1,self.k_top).short()
        grad_mask = b_bin_topk ^ b_grad_topk_sign.short()
        
        #grad_mask=(torch.ones(grad_mask.size()).short()-grad_mask).short() #target bfa
       
        # 4. apply the gradient mask upon ```b_grad_topk``` and in-place update it
        b_grad_topk *= grad_mask.float()

        # 5. identify the several maximum of absolute bit gradient and return the
        # index, the number of bits to flip is self.n_bits2flip
        #print(self.n_bits2flip)
        #self.n_bits2flip=2
        grad_max = b_grad_topk.abs().max()
        _, b_grad_max_idx = b_grad_topk.abs().view(-1).topk(self.n_bits2flip)
        bit2flip = b_grad_topk.clone().view(-1).zero_()
        
        if grad_max.item() != 0:  # ensure the max grad is not zero
            bit2flip[b_grad_max_idx] = 1
            bit2flip = bit2flip.view(b_grad_topk.size())
        else:
            pass

   
        #print(bit2flip)

# 6. Based on the identified bit indexed by ```bit2flip```, generate another
# mask, then perform the bitwise xor operation to realize the bit-flip.
        #print(w_bin_topk.size)
        w_bin_topk_flipped = (bit2flip.short() * self.b_w.abs().short()).sum(0, dtype=torch.int16) \
                ^ w_bin_topk
        weight_changed=w_bin_topk-w_bin_topk_flipped
        idx=(weight_changed!=0).nonzero() ## index of the weight  changed  
        # 7. update the weight in the original weight tensor
        w_bin[w_idx_topk] = w_bin_topk_flipped  # in-place change
        param_flipped = bin2int(w_bin,
                                self.N_bits).view(m.weight.data.size()).float()
        offse=(w_idx_topk[idx]) 

        return param_flipped,offse


    def progressive_bit_search(self, model,fn, data_p,data_d,data, target,data1,target1,render_kwargs_train):
        ''' 
        This function is only for type 1 and type 2 attack.
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped. 
        '''
        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        model.eval()
        #target[:]=3
        # 1. perform the inference w.r.t given data and target
        #output,_ = model(data.cuda())
        
        #output = fn(data_p.to('cpu'), data_d.to('cpu'), model.to('cpu'))
        #render_kwargs_train['network_fn'] = model
        #render_kwargs_train['network_fine'] = model
        #output,disp, acc, extras = fn(render_kwargs_train,data)
        output = fn(data_p, data_d, model)
        #         _, target = output.data.max(1)
        self.loss = self.criterion(output, target)
        # 2. zero out the grads first, then get the grads
        '''
        for m in model.modules():
            print(m)
            print(type(m))
            if isinstance(m, torch.nn.quantized.dynamic.modules.Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()
        '''    
        for m in model.modules():
            #if isinstance(m, torch.nn.Linear):
            if isinstance(m, quan_Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()
        self.loss.backward(retain_graph=True)
        #print(self.loss)
        # init the loss_max to enable the while loop
        self.loss_max = self.loss.item()
        #print(self.loss_max)
        # 3. for each layer flip #bits = self.bits2flip
        
        for j in range(1): 
            self.n_bits2flip += 1
            # iterate all the quantized conv and linear layer
            n=0
            offs=0
            for name, module in model.named_modules():
                if isinstance(module,quan_Linear):
                    n=n+1
                    
                    #if n<220:#include all layers
                    if n == 8: ## attack specific layer from 1 to 12
                        #print(module.weight)
                        clean_weight = module.weight.data.detach()
                        attack_weight,_ = self.flip_bit(module,offs)
                    # change the weight to attacked weight and get loss
                        module.weight.data = attack_weight
                        #output = fn(data_p.to('cpu'), data_d.to('cpu'), model.to('cpu'))
                        output = fn(data_p, data_d, model)
                        #render_kwargs_train['network_fn'] = model
                        #render_kwargs_train['network_fine'] = model
                        #output,disp, acc, extras = fn(render_kwargs_train,data)
                        self.loss_dict[name] = self.criterion(output,target).item()
                        
                        
                    # change the weight back to the clean weight
                        module.weight.data = clean_weight
                    if n<self.fc:
                        w=module.weight.size()
                        
                        offs+=w[0]*w[1]  ## keeping track of the offset 
                    else:
                        w=module.weight.size()
                        offs+=w[0]*w[1]   

            # after going through all the layer, now we find the layer with max loss
            max_loss_module = min(self.loss_dict.items(),
                                  key=operator.itemgetter(1))[0]
            self.loss_max = self.loss_dict[max_loss_module]
            #print(self.loss_dict)
        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change the that layer's weight without putting back the clean weight
        n=0
        offs=0
        for name, module in model.named_modules():
            if isinstance(module, quan_Linear):
                n=n+1
                #print(n,name)
                if name == max_loss_module:
                    print(n,name)
                #                 print(name, self.loss.item(), loss_max)
                    attack_weight,offset = self.flip_bit(module,offs)
                    #print(n,offset) 
                    #print(f'{module.weight.data} -> {attack_weight}') 
                    module.weight.data = attack_weight
                    nn=n
                    
                      
                if n<self.fc:
                    w=module.weight.size()
                    offs+=w[0]*w[1]  ## keeping track of the offset
 
                else:
                    w=module.weight.size()
                    offs+=w[0]*w[1]
                        
        # reset the bits2flip back to 0
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0

        return  nn, offset     

    def progressive_bit_search1(self, model, data, target,data1,target1):
        ''' 
        This function is only for type 3 attack
        Given the model, base on the current given data and target, go through
        all the layer and identify the bits to be flipped. 
        '''
        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        model.eval()
        #target[:]=3
        # 1. perform the inference w.r.t given data and target
        output,_ = model(data.cuda())
        
        #         _, target = output.data.max(1)
        self.loss = self.criterion(output, target.cuda())
        output1,_ = model(data1)
        self.loss +=self.criterion(output1,target1).item()
        # 2. zero out the grads first, then get the grads
        for m in model.modules():
            if isinstance(m, nn.quantized.dynamic.modules.Linear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()

        self.loss.backward()
        # init the loss_max to enable the while loop
        self.loss_max = self.loss.item()

        # 3. for each layer flip #bits = self.bits2flip
        
        for j in range(1): 
            self.n_bits2flip += 1
            # iterate all the quantized conv and linear layer
            n=0
            offs=0
            for name, module in model.named_modules():
                if isinstance(module, quan_Conv2d) or isinstance(
                        module, quan_Linear):
                    n=n+1
                    if n<220:
                    #print(n,name)
                        clean_weight = module.weight.data.detach()
                        attack_weight,_ = self.flip_bit(module,offs)
                    # change the weight to attacked weight and get loss
                        module.weight.data = attack_weight
                        output,_ = model(data)
                        self.loss_dict[name] = self.criterion(output,target).item()
                        output1,_ = model(data1)
                        xx=self.criterion(output1,target1).item()
                        print(xx,self.loss_dict[name])
                        self.loss_dict[name]+=xx
                        
                    # change the weight back to the clean weight
                        module.weight.data = clean_weight
                    if n<self.fc:
                        w=module.weight.size()
                        offs+=w[0]*w[1]*w[2]*w[3]  ## keeping track of the offset 
                    else:
                        w=module.weight.size()
                        offs+=w[0]*w[1]   

            # after going through all the layer, now we find the layer with max loss
            max_loss_module = min(self.loss_dict.items(),
                                  key=operator.itemgetter(1))[0]
            self.loss_max = self.loss_dict[max_loss_module]

        # 4. if the loss_max does lead to the degradation compared to the self.loss,
        # then change the that layer's weight without putting back the clean weight
        n=0
        offs=0
        for name, module in model.named_modules():
            if isinstance(module, nn.quantized.dynamic.modules.Linear):
                n=n+1
                #print(n,name)
                if name == max_loss_module:
                #print(n,name)
                #                 print(name, self.loss.item(), loss_max)
                    attack_weight,offset = self.flip_bit(module,offs)
                    module.weight.data = attack_weight
                    print(n,offset)    
                if n<self.fc:
                    w=module.weight.size()
                    offs+=w[0]*w[1]*w[2]*w[3]  ## keeping track of the offset
 
                else:
                    w=module.weight.size()
                    offs+=w[0]*w[1]
                        
        # reset the bits2flip back to 0
        self.bit_counter += self.n_bits2flip
        self.n_bits2flip = 0

        return  n, offset  
