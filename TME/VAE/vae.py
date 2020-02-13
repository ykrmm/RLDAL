#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:14:23 2019

auteurs : Yannis Karmim & Xavier le gentil.
"""


from datamaestro import prepare_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from torch.autograd import gradcheck
from torch import nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import numpy as np
import random
#############

class Mnist_dataset(Dataset):

    def __init__(self, X, y):

        liste_x= []
        self.labels = torch.from_numpy(y)#.double()
        data = X/255#.double()
        for d in data:
            liste_x.append(d.reshape((784,))) # On met sous forme de vecteur sinon la shape 28x28 ne passe pas
        
        self.data = torch.Tensor(liste_x)
        self.data = self.data.float()#double()

    def __getitem__(self, index):

        return self.data[index], self.labels[index]

    def __len__(self):

        return len(self.labels)

    def re_normalize(self):

        self.data*=255
    
    def compare_images(self,index,prediction,save=False,fname=None):

        """
            Compare our predictions with matplotlib and save it. 
            prediction -> Tensor 
            index -> int to use the __getitem__ function.
        """
        xtrue = self.__getitem__(index)[0]
        xtrue = xtrue * 255 # On remet des valeurs entre 0 et 255 pour l'affichage
        xtrue = xtrue.view((28,28)) # On remet au format 28x28 pixels
        prediction = prediction*255
        prediction = prediction.view((28,28))
        _, axarr = plt.subplots(2,sharex=True,sharey=True)
        axarr[0].set_title('original image')
        axarr[0].imshow(xtrue)
        axarr[1].set_title('output autoencoder')
        axarr[1].imshow(prediction)
        if fname is not None :
            path = os.path.join('figures',fname)
            plt.savefig(path)


class VAE(torch.nn.Module):

    def __init__(self, d_in):

        super(VAE, self).__init__()
        # Encodeur
        self.encode_1 = nn.Linear(d_in,400)
        self.encode_2 = nn.Linear(400,200)
        self.encode_mu = nn.Linear(200,100)
        self.encode_sigma = nn.Linear(200,100)



        # Decodeur
        self.decode_1 = nn.Linear(100, 200)
        self.decode_2 = nn.Linear(200,400)
        self.decode_3 = nn.Linear(400,d_in)

        # Activation 
        self.relu = torch.nn.ReLU()
        self.sigmo = torch.nn.Sigmoid()

    def encode(self,x):
        x = self.encode_1(x)
        x = self.relu(x)
        x = self.encode_2(x)
        x = self.relu(x)
        mu = self.encode_mu(x)
        sigma = self.encode_sigma(x)
        sigma = torch.log(sigma**2) # il faut apprendre log(sigma**2) pour stabilité 
        return mu,sigma
    
    def decode(self,z):
        z = self.decode_1(z)
        z = self.relu(z)
        z = self.decode_2(z)
        z = self.relu(z)
        z = self.decode_3(z)
        y = self.sigmo(z)
        return y

    def reparametrization(self,mu,sigma):
        eps = torch.randn(mu.size())
        z = mu + eps*sigma
        return z

    def forward(self, x):

        mu,sigma = self.encode(x)
        z = self.reparametrization(mu,sigma)
        y = self.decode(z)

        return y,mu,sigma



if __name__ == '__main__':
    

    ds = prepare_dataset("com.lecun.mnist")
    train_images ,  train_labels = ds.train.images.data(), ds.train.labels.data()
    test_images ,  test_labels = ds.test.images.data(), ds.test.labels.data()

    dataset_train = Mnist_dataset(train_images, train_labels)
    dataset_test = Mnist_dataset(test_images,test_labels)
    batch_size = 65
    train_loader = DataLoader(dataset_train,shuffle=True,batch_size=batch_size)
    test_loader = DataLoader(dataset_test,shuffle=True,batch_size=batch_size)

    writer = SummaryWriter()

    savepath = "save_net/auto_encoder.model"

    
    model = VAE(784)
    #model = model.double() # Sinon bug... Jsp pourquoi
    learning_rate= 10e-4 
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    
    def loss_VAE(x,x_decode,mu,sigma):

        KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        criterion = torch.nn.BCELoss(reduction='sum') 
        BCE = criterion(x_decode,x)

        return KLD + BCE

    criterion = torch.nn.BCELoss()
    epoch = 30
    print(" ------------ ENTRAINEMENT RESEAU DE NEURONES ---------------")
    for ep in range(epoch):
        print("EPOCHS : ",ep)
        for i, (x, y) in enumerate(train_loader):
            model.train()
            pred,mu,sigma = model(x)#.double()
            loss = loss_VAE(x,pred, mu,sigma)
            writer.add_scalar('Loss/train', loss, ep)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            
        for i,(x,y) in enumerate(test_loader):
            with torch.no_grad():

                model.eval() 
                pred,mu,sigma = model(x)#.double()
                loss = loss_VAE(x,pred, mu,sigma)
                writer.add_scalar('Loss/test', loss, ep)
    try:
        torch.save(model.state_dict(), savepath)
        print("model successfully saved in",savepath)
    except:
        print("something wrong with torch.save(model.state_dict(),savepath)")


    # Affichage image
    index = random.randint(0,len(test_images))
    x_to_pred = dataset_test.data[index]
    with torch.no_grad():
        model.eval()
        pred,mu,sigma = model(x_to_pred)
    dataset_test.compare_images(index,pred,save=True,fname='test_VAE.png')