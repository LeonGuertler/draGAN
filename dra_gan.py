import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import dra_gan_network


class dragan_agent:
    def __init__(self, model):
        self.model = model

        # define hyperparameters
        self.z_size = 64
        self.draGAN_EPOCHS = 1_000
        self.Critic_EPOCHS = 25
        self.batch_size = 1
        self.MAX_LEN = self.batch_size * 256
        self.nr_samples = 256

        self.g_lr = 0.01
        self.d_lr = 0.001


    def _one_hot_to_label(self, matrix, enforce=True):
        if len(np.unique(np.argmax(matrix, axis=1))) == self.nr_labels or enforce==False:
            return np.argmax(matrix, axis=1)
        else:
            # if there are no minority class labels, force one (otherwise some
            # classifiers will throw an error)

            # check which label doesn't exists
            missing_label = 1 - np.unique(np.argmax(matrix, axis=1))[0]

            # assign a high number to the one with the highest prob
            matrix[np.argmax(matrix[:, missing_label]), missing_label] = 999_999
            return np.argmax(matrix, axis=1)


    def train(self, X, y, verbose=1):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(2)

        out_size = np.shape(X)[1]+2
        self.nr_labels = len(np.unique(y))

        G = dra_gan_network.Generator(self.z_size, out_size, self.nr_samples)
        G.to(device)
        C = dra_gan_network.Critic(out_size, self.nr_samples)
        C.to(device)

        loss_G = nn.MSELoss()
        loss_C = nn.MSELoss()
        optimizer_G = optim.Adam(G.parameters(), lr=self.g_lr)
        optimizer_C = optim.Adam(C.parameters(), lr=self.c_lr)

        G.eval()
        C.eval()

        X_train_discriminator = []  # for training the Critic
        y_train_discriminator = []  # for training the Critic

        c_loss, g_loss = 0, 0
        best_train_auc = 0

        for epoch in range(self.draGAN_EPOCHS):
            # generate random noise
            z = torch.autograd.Variable(torch.normal(
            torch.Tensor([0]*self.z_size*self.batch_size),
            torch.Tensor([.1]*self.z_size*self.batch_size)
            )).view(self.batch_size, self.z_size)
            z = z.to(device)


            # generate training data
            optimizer_G.zero_grad()
            train_data = G(z.clone().detach())


            # assess the auc of the training data for each on in the batch
            for i in range(len(train_data)):
                train_data_numpy = train_data[i].cpu().clone().detach().numpy()

                X_train_d = train_data_numpy[:, :-self.nr_labels]
                y_train_d = self._one_hot_to_label(train_data_numpy[:, -self.nr_labels:])

                # train the classification model on the generated data
                model = LogisticRegression()
                model.fit(
                    X_train_d,
                    y_train_d
                )

                # assess the classification model on the actual training data
                val_data_auc = roc_auc_score(
                    y,
                    model.predict_proba(X)[:,1]
                )

                # check performance
                if val_data_auc >= best_train_auc:
                    best_train_auc = val_data_auc
                    # the best results are store for final training
                    X_train_gen = X_train_d.copy()
                    y_train_gen = y_train_d.copy()


                # randomly prune memory recall if it exceeds MAX_LEN
                if len(X_train_discriminator) >= self.MAX_LEN:
                    th = len(X_train_discriminator) / self.MAX_LEN - 1
                    rndm_idx = np.where(np.random.uniform(size=(len(X_train_discriminator),))>=th)
                    X_train_discriminator = X_train_discriminator[rndm_idx]
                    y_train_discriminator = y_train_discriminator[rndm_idx]


                # append to critic training data
                if len(X_train_discriminator) == 0:
                    new_data = torch.tensor(train_data_numpy, device=device)
                    new_data = new_data[None,]
                    X_train_discriminator = new_data
                    y_train_discriminator = (torch.ones((1, 1))*val_data_auc).to(device)

                else:
                    new_data = torch.tensor(train_data_numpy, device=device)
                    new_data = new_data[None,]
                    X_train_discriminator = torch.cat((
                        X_train_discriminator,
                        new_data
                    ), 0)
                    y_train_discriminator = torch.cat((
                        y_train_discriminator,
                        (torch.ones((1, 1))*val_data_auc).to(device)
                    ), 0)


            if verbose:
                print(f"{epoch} / {self.draGAN_EPOCHS}\t"+\
                    f"auc: {val_data_auc:.4f}\t"+\
                    f"best_train_auc: {best_train_auc:.4f}", end="\r")

            # re-train the Critic
            for d_epoch in range(self.Critic_EPOCHS):
                rndm_idx = np.random.shuffle(np.arange(len(y_train_discriminator)))
                optimizer_C.zero_grad()
                y_pred = C(X_train_discriminator)
                loss_c = loss_C(
                    y_pred,
                    y_train_discriminator
                )
                loss_c.backward()
                optimizer_D.step()
            c_loss = loss_c.item()

            # train the generator
            loss = loss_G(
                C(train_data),
                torch.ones((len(train_data), 1)).to(device)
            )
            loss.backward()
            optimizer_G.step()
            g_loss = loss.item()

        self.model.fit(X_train_gen, y_train_gen)

    def predict(self, X):
        return self.model.predict_proba(X)[:,1]
