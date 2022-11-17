import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import draGAN_network


class dragan_agent:
    def __init__(self, model, value_function):
        """
        Inputs:
            model: an uninstantiate classifier object
            value_function: a callable function that evaluates the performance.
                            (the higher the better, and optimally in range [0,1])
        """
        self.model = model

        # define hyperparameters
        self.z_size = 512
        self.EPOCHS = 1750
        self.Critic_EPOCHS = 2
        self.batch_size = 16
        self.max_memory_factor = 124
        self.nr_samples_generated_factor = 1.7934693188444824#1.7935
        self.G_LR = 0.0002660257499561004#0.000266
        self.C_LR = 0.03628406973687752#0.036284
        self.early_stopping_after = 921
        self.value_function = value_function

        self.MAX_LEN = self.batch_size * self.max_memory_factor

    def _one_hot_to_label(self, matrix, enforce_balance=True):
        if len(np.unique(np.argmax(matrix, axis=1))) == self.nr_labels or enforce_balance==False:
            return np.argmax(matrix, axis=1)
        else:
            # check which label doesn't exists
            missing_label = 1 - np.unique(np.argmax(matrix, axis=1))[0]

            # assign a high number to the one with the highest prob
            matrix[np.argmax(matrix[:, missing_label]), missing_label] = 999_999
            return np.argmax(matrix, axis=1)


    def train(self, X, y, verbose=1):
        """
        Inputs:
            X (numpy array): The independent features
            y (numpy array): The target features
            verbose (boolean): Whether progress in printed
        Returns:
            A trained version of the classifier passed into draGAN during
                instantiation.
        """
        self.len_train = len(X)
        self.nr_samples = int(self.len_train * self.nr_samples_generated_factor)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set all random seeds for reproducibility
        seed = 489
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.out_size = np.shape(X)[1]+2
        self.nr_labels = len(np.unique(y))

        X_train_discriminator = torch.zeros(
            (self.MAX_LEN, self.nr_samples, self.out_size),
            device=device,
            dtype=torch.float
        )
        y_train_discriminator = torch.zeros(
            (self.MAX_LEN, 1, 1),
            device=device,
            dtype=torch.float
        )
        train_discriminator_populated = np.zeros(
            (self.MAX_LEN),
            dtype=np.float
        )


        # instantiate networks
        G = draGAN_network.Generator(
            z_size=self.z_size,
            out_size=self.out_size,
            batch_size=self.nr_samples
        )
        G.to(device)

        C = draGAN_network.Critic(
            out_size=self.out_size,
            batch_size=self.nr_samples
        )
        C.to(device)

        # Define loss function
        loss_G = nn.MSELoss()
        loss_C = nn.MSELoss()

        # Define optimizer
        optimizer_G = optim.RMSprop(G.parameters(), self.G_LR)
        optimizer_C = optim.Adam(C.parameters(), self.C_LR)

        G.eval()
        C.eval()


        same_for = 0
        best_train_score = 0
        for epoch in range(self.EPOCHS):
            # generate the Gaussian noies vector
            z = torch.autograd.Variable(torch.normal(
            torch.Tensor([0]*self.z_size*self.batch_size),
            torch.Tensor([.1]*self.z_size*self.batch_size)
            )).view(self.batch_size, self.z_size)
            z = z.to(device)

            # alternate between train and eval
            if not epoch%3:
                G.eval()
                C.eval()
            else:
                G.train()
                C.train()

            # generate training data using the Gaussian Noise
            optimizer_G.zero_grad()
            train_data = G(z.clone().detach())


            # assess the auc of the training data
            train_data_batch_numpy = train_data.cpu().clone().detach().numpy()
            for i in range(len(train_data)):
                train_data_numpy = train_data_batch_numpy[i]

                X_train_d = train_data_numpy[:, :-self.nr_labels]
                y_train_d = self._one_hot_to_label(train_data_numpy[:, -self.nr_labels:])

                # instantiate and train the actual classifier on the generated data
                model = self.model()
                model.fit(
                    X_train_d,
                    y_train_d
                )

                # evalute the performance of the classifier on the passed data
                # with the evaluation metric provided.
                y_val = model.predict_proba(X)[:,1]
                val_score = self.value_function(y, y_val)

                # keep track of the best performing batch for final training
                if val_score >= best_train_score:
                    if val_score > best_train_score:
                        same_for = 0
                    best_train_score = val_score
                    X_train_gen = X_train_d.copy()
                    y_train_gen = y_train_d.copy()

                # check if early stopping condition is triggered
                same_for += 1
                if same_for > self.early_stopping_after*len(train_data):
                    break


                # append to discriminator training data
                rndm_idx = np.random.choice(
                    np.arange(len(X_train_discriminator)),
                    replace=False,
                    size=1
                )
                X_train_discriminator[rndm_idx] = torch.tensor(
                    train_data_numpy.copy(),
                    device=device,
                    dtype=torch.float
                )
                y_train_discriminator[rndm_idx] = val_score
                train_discriminator_populated[rndm_idx] = 1
                torch.cuda.empty_cache()

            if verbose:
                print(f"{epoch} / {self.EPOCHS}\t"+\
                    f"auc: {val_score:.4f}\t"+\
                    f"best_train_score: {best_train_score:.4f}", end="\r")


            # re-train the Critic
            for c_epoch in range(self.Critic_EPOCHS):
                optimizer_C.zero_grad()
                use_idx = np.where(train_discriminator_populated==1)
                y_pred = C(
                    X_train_discriminator[use_idx],
                )
                loss_c = loss_C(
                    y_pred,
                    y_train_discriminator[use_idx][:,:,0],
                )
                loss_c.backward()
                optimizer_C.step()

            torch.cuda.empty_cache()


            # train the generator
            loss = loss_G(
                C(train_data),
                torch.ones((len(train_data), 1)).to(device)
            )
            loss.backward()
            optimizer_G.step()

        # instantiate the passed classifier one more time and train it on the
        # most promising set of generated data
        self.model = self.model()
        self.model.fit(X_train_gen, y_train_gen)
        return self.model

    def predict(self, X):
        # predict the probabilities for y-labels based on the passed X-data
        return self.model.predict_proba(X)[:,1]
