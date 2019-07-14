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

class PrunningFineTuner_LeNet:
	def __init__(self, model):
		self.model = model
		self.compress_rates = []

	def total_num_filters(self):
		filters = 0
		for name, module in self.model.features._modules.items():
			if isinstance(module, torch.nn.modules.conv.Conv2d):
				filters = filters + module.out_channels
		return filters

	def total_num_weights(self):
		weights = 0
		for name, module in self.model.features._modules.items():
			if isinstance(module, torch.nn.modules.conv.Conv2d):
				weights = weights + module.in_channels * module.out_channels
		return weights

	def prune(self):
		#Get the accuracy before prunning

		num_of_filters = self.total_num_filters()
		print("num_of_filters:", num_of_filters)

		num_of_weights = self.total_num_weights()
		print("num_of_weights:", num_of_weights)

        #resume
		if os.path.isfile("prune_targets_"):
			prune_targets = torch.load("prune_targets_")
			print("prune_targets_ resume:", prune_targets)

			self.accuracys1 = torch.load("accuracys1_prunned")
			self.accuracys5 = torch.load("accuracys5_prunned")
			print("accuracys1_prunned resume:", self.accuracys1)
			print("accuracys5_prunned resume:", self.accuracys5)
		else:
			sys.exit()

		for i in range(num_of_filters - 2):
			print(i, "-", "prune_targets:", prune_targets[i])

			layers_prunned = {}
			for layer_index, filter_index in prune_targets[i]:
				if layer_index not in layers_prunned:
					layers_prunned[layer_index] = 0
				layers_prunned[layer_index] = layers_prunned[layer_index] + 1

			print("Prunning filters.. ")
			model = self.model.cpu()

			for layer_index, filter_index in prune_targets[i]:
				model = prune_lenet_conv_layer(model, layer_index, filter_index)

			compress_rate = float(self.total_num_weights()) / num_of_weights
			print("compress rate:", compress_rate)

			self.compress_rates.append(compress_rate)

			self.model = model.cuda()
		

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    #parser.add_argument("--train_path", type = str, default = "train")
    #parser.add_argument("--test_path", type = str, default = "test")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = get_args()
	print("args:", args)

	model = ModifiedLeNetModel().cuda()
	print("model:", model)

	fine_tuner = PrunningFineTuner_LeNet(model)

	fine_tuner.prune()
	torch.save(fine_tuner.compress_rates, "compress_rates")
	print("model_prunned_final:", fine_tuner.model)
