import pandas as pd
import numpy as np
import datetime, os
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

from data_organizer import *
import preprocessing
import smote, ksmote, ac_gan, cWGAN_gan, dra_gan, vanilla, dra_gan_test

# set the seed
seed = 489
np.random.seed(seed)
num_runs = 10
n_splits = 10

# evalute and reformat performances for df
def evaluate(perf_dict, metric):
    return_list = []
    for model in perf_dict.keys():
        return_list.append(f"{np.mean(perf_dict[model][metric]):.4f}")
        return_list.append(f"{np.std(perf_dict[model][metric]):.4f}")
    return return_list


# define the models
# you can add any data generation model here, as long as it is wrapped
# into a class that accepts a classification model as input and has as
# a "train" method and a "predict" function.
model_dict = {
    "draGAN": dra_gan.dragan_agent,
}

# define preprocessing function
preprocessing_dict = {
    #"None": preprocessing.none,
    "MinMax": preprocessing.minmax,
    #"Normalize": preprocessing.normalize,
}

# create relevant folders for the current run
folder_name = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
os.mkdir(f"outputs/{folder_name}")
os.mkdir(f"outputs/{folder_name}/raw_predictions")
directory_path = f"outputs/{folder_name}"


for prep_fun in preprocessing_dict.keys():

    # create dataframes for perfomance tracking
    columns = ["Dataset"]
    for k in model_dict.keys():
        columns.append(f"{k}_score")
        columns.append(f"{k}_std")
    auc_df = pd.DataFrame(columns=columns)

    for i, data_set in enumerate(data_dict.keys()):
        X_data = data_dict[data_set]["X_data"]
        y_data = data_dict[data_set]["y_data"]

        perf_dict = {key:{"f1":[], "auc":[]} for key in model_dict.keys()}
        for ii in range(num_runs):
            skf = StratifiedKFold(n_splits=n_splits)

            for iii, (train_index, test_index) in enumerate(skf.split(X_data, y_data)):
                # the data is split so that the model trains on what is supposed
                # to be the test data, and tested on what is supposed to be the
                # training data. This was done to minimize the data each model
                # has available
                X_train, y_train = X_data[test_index], y_data[test_index]
                X_test, y_test = X_data[train_index], y_data[train_index]

                # preprocess both X_train and X_test
                X_train, X_test = preprocessing_dict[prep_fun](X_train, X_test)


                for iv, model_name in enumerate(model_dict.keys()):
                    model = model_dict[model_name](LogisticRegression())

                    try:
                        model.train(X_train, y_train)
                        y_pred = model.predict(X_test)

                        # save the raw predictions
                        path = f"{directory_path}/raw_predictions/{i}_{ii}_"+\
                            f"{iii}_{iv}_{model_name}"
                        np.save(f"{path}_pred.npy", y_pred)
                        np.save(f"{path}_test.npy", y_test)

                        # evaluate the model
                        auc = roc_auc_score(y_test, y_pred)
                    except Exception as exc:
                        print(exc)
                        auc = 0

                    # temporarily store model performance
                    perf_dict[model_name]["auc"].append(auc)

                    print(f"{i} / {len(data_dict)}\t"+\
                        f"{ii} / {num_runs}\t"+\
                        f"{iii} / {n_splits}\t"+\
                        f"{iv} / {len(model_dict)}\t"+\
                        f"{model_name}: auc: {auc:.4f}")

        # append performances to data frame
        auc_df.loc[len(auc_df)] = data_set, *evaluate(perf_dict, "auc")

        #save to directory path
        auc_df.to_csv(f"{directory_path}/{prep_fun}_auc.csv")
