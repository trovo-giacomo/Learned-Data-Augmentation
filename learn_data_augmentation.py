# -*- coding: utf-8 -*-
# Authors:
# Giacomo Trovò
# Giovanni Grego
# Nazer Hdaifeh
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from libcpab.pytorch import cpab
from libcpab.helper.utility import show_images
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import random as rand
import pickle

lab=3
dataset = datasets.MNIST(root='./data', download=True, transform=None)
idx = (dataset.train_labels == lab)
train_labels = dataset.train_labels[idx]
train_data = dataset.train_data[idx]
print(np.shape(train_data))
print(dataset)

num_imgs = 10;

thetas = torch.Tensor(num_imgs*(num_imgs-1),34)

loss_rate = []
xn_meno_xm = []
lrate=0.001
maxiter = 500
row = 0
for k in range(num_imgs):
    for j in range(num_imgs):
        if(k != j):
            print("k: ",k," j: ",j)
            xn = train_data[k]/255;
            xm = train_data[j]/255;
            plt.imshow(xm, cmap="gray")
            plt.show()
            plt.imshow(xn, cmap="gray")
            plt.show()
            
            #print(np.shape(xm))
            N = 1
            xm = np.tile(np.expand_dims(xm, 2), [N,1,1,1])
            #print(np.shape(xm))
            xm = torch.Tensor(xm).permute(0,3,1,2)
            #print(np.shape(xm))
            
            T2 = cpab(tess_size=[3,3], device='cpu')
            theta_est = T2.identity(1, epsilon=1e-4)
            theta_est.requires_grad = True
              
            # Pytorch optimizer
            optimizer = torch.optim.Adam([theta_est], lr=lrate)
            
            # Optimization loop
            for i in range(maxiter):
                trans_est = T2.transform_data(xm, theta_est, outsize=(28, 28))
                loss = (xn - trans_est).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Iter', i, ', Loss', np.round(loss.item(), 4), '||xn - xm◦Ttheta||: ',
                         np.linalg.norm((xn-trans_est.cpu().detach()).numpy().round(4)))
                if i == maxiter-1:
                    loss_rate.append(np.round(loss.item(), 4))
                    xn_meno_xm.append(np.linalg.norm((xn-trans_est.cpu().detach()).numpy().round(4)))
            #print(theta_est)
            print(np.shape(theta_est))
            print("Loss", np.round(loss.item(), 4), '||xn - xm◦Ttheta||: ',
                  np.linalg.norm((xn-trans_est.cpu().detach()).numpy().round(4)))
            thetas[row] = theta_est
            print(np.shape(theta_est))
            row+=1;
            # Show the results
            plt.subplots(1,3, figsize=(10, 15))
            plt.subplot(1,3,1)
            plt.imshow(np.squeeze(xm.permute(0,2,3,1).cpu().numpy()[0]), cmap="gray")
            plt.axis('off')
            plt.title('Source')
            plt.subplot(1,3,2)
            plt.imshow(xn, cmap="gray")
            plt.axis('off')
            plt.title('Target')
            plt.subplot(1,3,3)
            plt.imshow(np.squeeze(trans_est.permute(0,2,3,1).cpu().detach().numpy()[0]), "gray")
            plt.axis('off')
            plt.title('Estimate')
            if j==1:
                plt.savefig("results/intermediate/"+str(lab)+str(k)+str(j)+".png", transparent=True, bbox_inches="tight")
            plt.show()
 
matrix_thetas = thetas.detach().numpy()

mu = np.mean(matrix_thetas,0)
print("mean dimension: ",np.shape(mu))

sigma = np.cov(matrix_thetas,rowvar=False)
print("sigma shape:",np.shape(sigma))


import os

os.makedirs("results/final/"+str(num_imgs)+"images, lr="+str(lrate)+", maxiter="+str(maxiter))



theta_star =  np.random.multivariate_normal(mu,sigma,1)
print(np.shape(theta_star))
theta_star = torch.from_numpy(theta_star).float()

for i in range(5,10):
    
    theta_star =  np.random.multivariate_normal(mu,sigma,1)
    print(np.shape(theta_star))
    theta_star = torch.from_numpy(theta_star).float()
    
    source_img = train_data[num_imgs+200+i];
    source_img = np.tile(np.expand_dims(source_img, 2), [N,1,1,1])
    source_img = torch.Tensor(source_img).permute(0,3,1,2)
    
    
    T_new = cpab(tess_size=[3,3], device='cpu')
    transformed_data = T_new.transform_data(source_img, theta_star, outsize=(28, 28))
    
    
    plt.subplots(1,2, figsize=(10, 15))
    plt.subplot(1,2,1)
    plt.imshow(np.squeeze(source_img.permute(0,2,3,1).cpu().numpy()[0]), cmap="gray")
    plt.axis('off')
    plt.title('Source')
    plt.subplot(1,2,2)
    plt.imshow(np.squeeze(transformed_data.permute(0,2,3,1).cpu().detach().numpy()[0]), cmap="gray")
    plt.axis('off')
    plt.title('New Data')
    plt.savefig("results/final/"+str(num_imgs)+"images, lr="+str(lrate)+", maxiter="+str(maxiter)+"/"+str(lab)+"_"+str(i)+".png", transparent=True, bbox_inches="tight")
    plt.show()
    

open_file = open("results/final/"+str(num_imgs)+"images, lr="+str(lrate)+", maxiter="+str(maxiter)+"/losses", "wb")
pickle.dump(loss_rate, open_file)
open_file.close()

open_file = open("results/final/"+str(num_imgs)+"images, lr="+str(lrate)+", maxiter="+str(maxiter)+"/difference", "wb")
pickle.dump(xn_meno_xm, open_file)
open_file.close()


#open_file = open(file_name, "rb")
#loaded_list = pickle.load(open_file)
#open_file.close()
