import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import random, torch
import imbalanced_databases as imdb

import draGAN

# set seed
seed = 489
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# load data
data_dict = imdb.load_abalone_17_vs_7_8_9_10(encode=True)

# train test split
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(data_dict["data"], data_dict["target"]):
    X_train, y_train = data_dict["data"][train_index], data_dict["target"][train_index]
    X_test, y_test = data_dict["data"][test_index], data_dict["target"][test_index]

    # instantiate draGAN
    model = draGAN.dragan_agent(
        model=LogisticRegression,
        value_function=roc_auc_score
    )
    # train draGAN
    model.train(
        X=X_train,
        y=y_train
    )

    # test draGAN
    y_pred = model.predict(X_test)
    print(f"\nTest AUC-Score: {roc_auc_score(y_test, y_pred)}")
