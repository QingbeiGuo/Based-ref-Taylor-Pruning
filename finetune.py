#pytorch -0.2.1
#python -3.6.2
#torchvision - 0.1.9

import torch
from torch.autograd import Variable
from torchvision import models
#import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset_mnist
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import os

class ModifiedLeNetModel(torch.nn.Module):
	def __init__(self):
		super(ModifiedLeNetModel, self).__init__()

		self.features = nn.Sequential(
		    nn.Conv2d(1, 32, kernel_size=5, stride=1),
        nn.ReLU(inplace=True),
		    nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=5, stride=1),
		    nn.ReLU(inplace=True),
		    nn.MaxPool2d(kernel_size=2))                       

		for param in self.features.parameters():
			param.requires_grad = True

		self.classifier = nn.Sequential(
		    nn.Linear(5*5*64, 120),
		    nn.ReLU(inplace=True),
		    nn.Linear(120, 84),
		    nn.ReLU(inplace=True),                      
		    nn.Linear(84, 10))                       

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)

		return x

class FilterPrunner:
	def __init__(self, model):
		self.model = model
		self.reset()
	
	def reset(self):
		# self.activations = []
		# self.gradients = []
		# self.grad_index = 0
		# self.activation_to_layer = {}
		self.filter_ranks = {}
		self.activation_to_layer = {}

	def forward(self, x, layer_index):
#################################################################################
		self.activations = []
		self.gradients = []
		self.deltaxs = []
		activation_index = 0
		self.grad_index = 0
		self.layer_index = layer_index

		next_layer_index = -1
		next_net_index = -1
		for layer, (name, module) in enumerate(self.model.features._modules.items()):
		    x = module(x)
		    if isinstance(module, torch.nn.modules.conv.Conv2d):
		        if (next_net_index == -1) and (layer == self.layer_index):
		            next_conv = None
		            offset = 1
		            while layer+offset < len(self.model.features._modules.items()):
		                res =  self.model.features[layer+offset]
		                if isinstance(res, torch.nn.modules.conv.Conv2d):
		                    next_conv = res
		                    break
		                offset = offset + 1

		            deltax = []
		            if not next_conv is None:
		                x_1 = Variable(torch.zeros(x.size()).cuda())
		                x_1[:, :, :, :] = x[:, :, :, :]

		                for i in range(layer+1, layer+offset+1, 1):
		                    x_1 = self.model.features[i](x_1)

		                for i in range(x.size(1)):                                                  
		                    x_2 = Variable(torch.zeros(x.size()).cuda())
		                    x_2[:, :, :, :] = x[:, :, :, :]
		                    x_2[:, i, :, :] = 0                                                   

		                    for j in range(layer+1, layer+offset+1, 1):
		                        x_2 = self.model.features[j](x_2)

		                    deltax.append(x_1-x_2)

		                x.register_hook(self.compute_rank)
		                self.activations.append(x)
		                self.deltaxs.append(deltax)
		                self.activation_to_layer[layer] = layer
		                activation_index += 1
		                next_layer_index = layer+offset
		                next_net_index = 0
		                del x_1
		                del x_2
		            else:
		                new_linear = None
		                offset = 1
		                for _, module in model.classifier._modules.items():
		                    if isinstance(module, torch.nn.Linear):
		                        new_linear = module                                                 
		                        break
		                    offset = offset + 1
		            
		                if not new_linear is None:
		                    x_1 = Variable(torch.zeros(x.size()).cuda())
		                    x_1[:, :, :, :] = x[:, :, :, :]

		                    for i in range(layer+1, len(self.model.features._modules.items()), 1):
		                        x_1 = self.model.features[i](x_1)
		                    x_1 = x_1.view(x_1.size(0), -1)
		                    for i in range(0, offset, 1):
		                        x_1 = self.model.classifier[i](x_1)

		                    for i in range(x.size(1)):                                              
		                        x_2 = Variable(torch.zeros(x.size()).cuda())
		                        x_2[:, :, :, :] = x[:, :, :, :]
		                        x_2[:, i, :, :] = 0

		                        for j in range(layer+1, len(self.model.features._modules.items()), 1):
		                            x_2 = self.model.features[j](x_2)
		                        x_2 = x_2.view(x_2.size(0), -1)
		                        for j in range(0, offset ,1):
		                            x_2 = self.model.classifier[j](x_2)

		                        deltax.append(x_1-x_2)

		                    x.register_hook(self.compute_rank)
		                    self.activations.append(x)
		                    self.deltaxs.append(deltax)
		                    self.activation_to_layer[layer] = layer
		                    activation_index += 1
		                    next_layer_index = offset-1
		                    next_net_index = 1
		                    del x_1
		                    del x_2
		        elif (next_net_index == 0) and (layer == next_layer_index):
		            x.register_hook(self.compute_rank)
		            self.activations.append(x)
		            #self.activation_to_layer[layer] = layer
		            activation_index += 1

		x = x.view(x.size(0), -1)
		for layer, (name, module) in enumerate(self.model.classifier._modules.items()):
		    x = module(x)
		    if (next_net_index == 1) and (layer == next_layer_index) and isinstance(module, torch.nn.Linear):
		        x.register_hook(self.compute_rank)
		        self.activations.append(x)
		        #self.activation_to_layer[layer] = len(self.model.features._modules.items()) + layer
		        activation_index += 1
		
		return x

	def compute_rank(self, grad):
#################################################################################
		activation_index = len(self.activations) - self.grad_index - 1
		self.gradients.append(grad)
		if self.grad_index == 1:
		    activation = self.activations[activation_index]                             # activation = [32 64 32 32]...
		    deltax = self.deltaxs[activation_index]
		    gradient = self.gradients[len(self.activations) - activation_index - 2]
		    values = torch.zeros(len(deltax)).cuda()
		    for i in range(len(deltax)):
		        values[i] = torch.sum(torch.from_numpy(abs(deltax[i].data.cpu().numpy()) * abs(gradient.data.cpu().numpy())))

		    # Normalize the rank by the filter dimensions
		    values = values * len(deltax)
		    for i in range(activation.dim()):
		        values = values / activation.size(i)

		    if self.layer_index not in self.filter_ranks:
		        self.filter_ranks[self.layer_index] = \
                torch.FloatTensor(activation.size(1)).zero_().cuda()
		    self.filter_ranks[self.layer_index] += values
		    
		self.grad_index += 1
#################################################################################

	def lowest_ranking_filters(self, num):
		data = []
		for i in sorted(self.filter_ranks.keys()):
			for j in range(self.filter_ranks[i].size(0)):
				data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

		return nsmallest(num, data, itemgetter(2))

	def normalize_ranks_per_layer(self):
		for i in self.filter_ranks:
			v = torch.abs(self.filter_ranks[i])                                         #here, filter_ranks is deltaE
			v = v / np.sqrt(torch.sum(v * v))
			self.filter_ranks[i] = v.cpu()

	def get_prunning_plan(self, num_filters_to_prune):
		filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

		# After each of the k filters are prunned,
		# the filter index of the next filters change since the model is smaller.
		filters_to_prune_per_layer = {}
		for (l, f, _) in filters_to_prune:
			if l not in filters_to_prune_per_layer:
				filters_to_prune_per_layer[l] = []
			filters_to_prune_per_layer[l].append(f)

		for l in filters_to_prune_per_layer:
			filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
			for i in range(len(filters_to_prune_per_layer[l])):
				filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

		filters_to_prune = []
		for l in filters_to_prune_per_layer:
			for i in filters_to_prune_per_layer[l]:
				filters_to_prune.append((l, i))

		return filters_to_prune				

class PrunningFineTuner_LeNet:
	def __init__(self, train_path, test_path, model):
		self.train_data_loader = dataset_mnist.train_loader(train_path)
		self.test_data_loader = dataset_mnist.test_loader(test_path)

		self.model = model
		self.criterion = torch.nn.CrossEntropyLoss()
		self.prunner = FilterPrunner(self.model) 
		self.model.train()

		self.accuracys1 = []
		self.accuracys5 = []
		self.accuracys11 = []
		self.accuracys55 = []

	def test(self, flag = -1):
		self.model.eval()

		#correct = 0
		correct1 = 0
		correct5 = 0
		total = 0

		print("Testing...")
		for i, (batch, label) in enumerate(self.test_data_loader):
			batch = batch.cuda()
			output = model(Variable(batch))
			pred = output.data.max(1)[1]
			#correct += pred.cpu().eq(label).sum()
			cor1, cor5 = accuracy(output.data, label, topk=(1, 5))                      # measure accuracy top1 and top5
			correct1 += cor1
			correct5 += cor5
			total += label.size(0)

		if flag != -1:
		    self.accuracys1.append(float(correct1.numpy()) / total)
		    self.accuracys5.append(float(correct5.numpy()) / total)

		print("Accuracy Top1:", float(correct1.numpy()) / total)
		print("Accuracy Top5:", float(correct5.numpy()) / total)
  
		self.accuracys11.append(float(correct1.numpy()) / total)
		self.accuracys55.append(float(correct5.numpy()) / total)

		self.model.train()                                                              

	def train(self, optimizer = None, epoches = 10, batches = -1, flag = -1):
		if optimizer is None:
			optimizer = \
                optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
                #optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
                #optim.SGD(model.features.parameters(), lr=0.001, momentum=0.9)

        #resume
		if flag == 0:
			if os.path.isfile("epoches_training") and os.path.isfile("model_training"):
			    list_epoches = torch.load("epoches_training")
			    self.model = torch.load("model_training")
			    print("model_training resume:", self.model)

			    self.accuracys1 = torch.load("accuracys1_trainning")
			    self.accuracys5 = torch.load("accuracys5_trainning")
			    print("accuracys1_trainning resume:", self.accuracys1)
			    print("accuracys5_trainning resume:", self.accuracys5)
			else:
			    list_epoches = list(range(epoches))
		elif flag == 1:
			if os.path.isfile("epoches_training_pruned") and os.path.isfile("model_training_pruned"):
			    list_epoches = torch.load("epoches_training_pruned")
			    self.model = torch.load("model_training_pruned")
			    print("model_training_pruned resume:", self.model)

			    self.accuracys1 = torch.load("accuracys1_training_pruned")
			    self.accuracys5 = torch.load("accuracys5_training_pruned")
			    print("accuracys1_training_pruned resume:", self.accuracys1)
			    print("accuracys5_training_pruned resume:", self.accuracys5)
			else:
			    list_epoches = list(range(epoches))
		else:
			list_epoches = list(range(epoches))

		list_ = list_epoches[:]
		#for i in range(epoches):
		for i in list_epoches[:]:
			print("Epoch: ", i)
			self.train_epoch(i, batches, optimizer)
			self.test(flag)

			list_.remove(i)                                                             #update list_epoches

            #save
			if flag == 0:
			    torch.save(list_, "epoches_training")
			    torch.save(self.model, "model_training")
			    torch.save(self.accuracys1, "accuracys1_trainning")
			    torch.save(self.accuracys5, "accuracys5_trainning")
			elif flag == 1:
			    torch.save(list_, "epoches_training_pruned")
			    torch.save(self.model, "model_training_pruned")
			    torch.save(self.accuracys1, "accuracys1_training_pruned")
			    torch.save(self.accuracys5, "accuracys5_training_pruned")

		print("Finished fine tuning.")

	def train_batch(self, optimizer, batch, label, rank_filters, layer_index):
		self.model.zero_grad()
		input = Variable(batch)                                                         #Tensor->Variable

		if rank_filters:
			output = self.prunner.forward(input, layer_index)
			#loss=self.criterion(output, Variable(label))
			#loss.backward()
			self.criterion(output, Variable(label)).backward()                          
		else:
		    #output = self.model(input)
		    #loss = self.criterion(output, Variable(label))
		    #loss.backward()
			self.criterion(self.model(input), Variable(label)).backward()               #1. output=self.model(input) 2.loss=self.criterion(output, Variable(label)) 3.loss.backward()
			optimizer.step()                                                            #update parameters

	def train_epoch(self, epoch, batches, optimizer = None, rank_filters = False, layer_index = -1):
		for step, (batch, label) in enumerate(self.train_data_loader):
			if (step == batches):
			    break
			print("Epoch-step: ", epoch, "-", step)
			self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters, layer_index)

	'''def get_candidates_ranks(self):
		for layer, (name, module) in enumerate(self.model.features._modules.items()):
			if isinstance(module, torch.nn.modules.conv.Conv2d):
			    self.prunner.activations_count = module.out_channels
			    self.prunner.values = torch.zeros(self.prunner.activations_count).cuda()
			    for i in range(module.out_channels): 
			        #print("layer-i:", layer, "-", i)
			        self.train_epoch(epoch = -1, batches = 1, rank_filters = True, layer_index = layer, filter_index = i)'''

	def get_candidates_ranks(self):
		for step, (batch, label) in enumerate(self.train_data_loader):
			if (step == 32):
			    break

			for layer, (name, module) in enumerate(self.model.features._modules.items()):
			  if isinstance(module, torch.nn.modules.conv.Conv2d):
			    print("step-layer:", step, "-", layer)
			    self.train_batch(optimizer = None, batch = batch.cuda(), label = label.cuda(), rank_filters = True, layer_index = layer)

	def get_candidates_to_prune(self, num_filters_to_prune = 1):
		self.prunner.reset()

		self.get_candidates_ranks()

		self.prunner.normalize_ranks_per_layer()

		return self.prunner.get_prunning_plan(num_filters_to_prune)
		
	def total_num_filters(self):
		filters = 0
		for name, module in self.model.features._modules.items():
			if isinstance(module, torch.nn.modules.conv.Conv2d):
				filters = filters + module.out_channels
		return filters

	def prune(self, num_filters_to_prune = 1):
	#def prune(self, num_filters_to_prune = 16):
	#def prune(self, num_filters_to_prune = 8):
		#Get the accuracy before prunning
		self.test()

		self.model.train()

		#Make sure all the layers are trainable
		for param in self.model.features.parameters():
			param.requires_grad = True

		num_of_filters = self.total_num_filters()
		num_filters_to_prune_per_iteration = num_filters_to_prune
		iterations = int(float(num_of_filters) / num_filters_to_prune_per_iteration)

		#iterations = int(iterations * 5.0 / 6)
		#iterations = int(iterations * 11.0 / 12)

		print("Number of prunning iterations to reduce 67% filters:", iterations)

		prune_targets_ = []
		layers_prunned_ = []
        #resume
		if os.path.isfile("iterations_prunned") and os.path.isfile("model_prunned"):
			list_iterations = torch.load("iterations_prunned")
			print("list_iterations_ resume:", list_iterations)
			prune_targets_ = torch.load("prune_targets_")
			print("prune_targets_ resume:", prune_targets_)
			layers_prunned_ = torch.load("layers_prunned_")
			print("layers_prunned_ resume:", layers_prunned_)

			self.accuracys1 = torch.load("accuracys1_prunned")
			self.accuracys5 = torch.load("accuracys5_prunned")
			print("accuracys1_prunned resume:", self.accuracys1)
			print("accuracys5_prunned resume:", self.accuracys5)

			self.accuracys11 = torch.load("accuracys11_prunned")
			self.accuracys55 = torch.load("accuracys55_prunned")
			print("accuracys11_prunned resume:", self.accuracys11)
			print("accuracys55_prunned resume:", self.accuracys55)
		else:
			list_iterations = list(range(iterations))

		list_ = list_iterations[:]
		#for i in range(iterations):                                                        #for each iterations
		for i in list_iterations[:]:                                                        #for each iterations
			print(iterations, "-", i, "Ranking filters...")
			prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)

			print("prune_targets", prune_targets)
			prune_targets_.append(prune_targets)

			layers_prunned = {}
			for layer_index, filter_index in prune_targets:
				if layer_index not in layers_prunned:
					layers_prunned[layer_index] = 0
				layers_prunned[layer_index] = layers_prunned[layer_index] + 1

			print("Layers that will be prunned", layers_prunned)
			layers_prunned_.append(layers_prunned)

			print("Prunning filters.. ")
			model = self.model.cpu()

			for layer_index, filter_index in prune_targets:
				model = prune_lenet_conv_layer(model, layer_index, filter_index)

			self.model = model.cuda()

			message = str(100*float(self.total_num_filters()) / num_of_filters) + "%"
			print("Filters prunned", str(message))

			self.test(flag = 1)                                                             

			print("Fine tuning to recover from prunning iteration.")
			if ((i > 0) and (i % 8) == 0):
				optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
				self.train(optimizer, epoches = 5)                                               
			else:
				optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
				self.train(optimizer, epoches = 1, batches = 32)                                               

			self.test(flag = 1)

			list_.remove(i)                                                                 #update list_epoches

      #save
			torch.save(self.model, "model_prunned")
			torch.save(list_, "iterations_prunned")
			torch.save(prune_targets_, "prune_targets_")
			torch.save(layers_prunned_, "layers_prunned_")
			torch.save(self.accuracys1, "accuracys1_prunned")
			torch.save(self.accuracys5, "accuracys5_prunned")
			torch.save(self.accuracys11, "accuracys11_prunned")
			torch.save(self.accuracys55, "accuracys55_prunned")
			print("model_prunned:", self.model)

		print("Finished. Going to fine tune the model a bit more")
		self.train(optimizer, epoches = 15, flag = 1)

		torch.save(self.accuracys11, "accuracys11_prunned")
		torch.save(self.accuracys55, "accuracys55_prunned")

def accuracy(output, target, topk=(1,)):                                               
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)                                          
    pred = pred.t()                                                                     
    correct = pred.cpu().eq(target.view(1, -1).expand_as(pred))                               

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)                                 
        res.append(correct_k)
    return res

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    #parser.add_argument("--train_path", type = str, default = "train")
    #parser.add_argument("--test_path", type = str, default = "test")
    parser.add_argument("--train_path", type = str, default = "/root/work/Datasets/MNIST/")
    parser.add_argument("--test_path", type = str, default = "/root/work/Datasets/MNIST/")
    parser.set_defaults(train=False)
    parser.set_defaults(prune=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = get_args()
	print("args:", args)

	if args.train:
		model = ModifiedLeNetModel().cuda()
	elif args.prune:
		model = torch.load("model_prunned").cuda()
	print("model_prunned:", model)

	fine_tuner = PrunningFineTuner_LeNet(args.train_path, args.test_path, model)

	if args.train:
		fine_tuner.train(epoches = 100, flag = 0)
		torch.save(model, "model")
		print("model:", model)

	elif args.prune:
		fine_tuner.prune()
		torch.save(fine_tuner.model, "model_prunned_final")
		print("model_prunned_final:", fine_tuner.model)
