import numpy as np
import pandas as pd

data_dict = {}

# abalone9v18
df = pd.read_csv("data/abalone.data", header=None).sample(frac=1)
df.columns = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "label")
df = df[df["label"].isin([9,18])]
df["label"] = df["label"].eq(18).astype(np.int8)
df = df.drop(columns=["f1"])
data_dict["Abalone9v18"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}


# abalone9vREST
df = pd.read_csv("data/abalone.data", header=None).sample(frac=1)
df.columns = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "label")
df["label"] = df["label"].eq(9).astype(np.int8)
df = df.drop(columns=["f1"])
data_dict["Abalone9vREST"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}

# abalone 7v17
df = pd.read_csv("data/abalone.data", header=None).sample(frac=1)
df.columns = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "label")
df = df[df["label"].isin([7,17])]
df["label"] = df["label"].eq(17).astype(np.int8)
df = df.drop(columns=["f1"])
data_dict["Abalone7v17"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}

# abalone 19vRest
df = pd.read_csv("data/abalone.data", header=None).sample(frac=1)
df.columns = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "label")
df["label"] = df["label"].eq(19).astype(np.int8)
df = df.drop(columns=["f1"])
data_dict["Abalone19vREST"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}

# Yest 1v3
df = pd.read_fwf("data/yeast.data", header=None).sample(frac=1).sample(frac=1)
df.columns = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "label")
df = df[df["label"].isin(["NUC","ME3"])]
df["label"] = df["label"].eq("ME3").astype(np.int8)
df = df.drop(columns=["f1"])
data_dict["Yeast1v3"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}

# Yest 2v4
df = pd.read_fwf("data/yeast.data", header=None).sample(frac=1).sample(frac=1)
df.columns = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "label")
df = df[df["label"].isin(["MIT","ME2"])]
df["label"] = df["label"].eq("ME2").astype(np.int8)
df = df.drop(columns=["f1"])
data_dict["Yeast2v4"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}

# Yest 1v7
df = pd.read_fwf("data/yeast.data", header=None).sample(frac=1).sample(frac=1)
df.columns = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "label")
df = df[df["label"].isin(["NUC","VAC"])]
df["label"] = df["label"].eq("VAC").astype(np.int8)
df = df.drop(columns=["f1"])
data_dict["Yeast1v7"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}

# Yest 6vRest
df = pd.read_fwf("data/yeast.data", header=None).sample(frac=1).sample(frac=1)
df.columns = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "label")
df["label"] = df["label"].eq("EXC").astype(np.int8)
df = df.drop(columns=["f1"])
data_dict["Yeast6vREST"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}


# Ecoli 0v1
df = pd.read_csv("data/ecoli.csv", header=None).sample(frac=1)
df.columns = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "label")
df = df[df["label"].isin(["cp","im"])]
df["label"] = df["label"].eq("im").astype(np.int8)
data_dict["Ecoli0v1"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}

# Ecoli 3
df = pd.read_csv("data/ecoli.csv", header=None).sample(frac=1)
df.columns = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "label")
df["label"] = df["label"].eq("imU").astype(np.int8)
data_dict["Ecoli3vREST"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}

# Ecoli 4
df = pd.read_csv("data/ecoli.csv", header=None).sample(frac=1)
df.columns = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "label")
df["label"] = df["label"].eq("om").astype(np.int8)
data_dict["Ecoli4vREST"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}


# ILPD 1v2
df = pd.read_csv("data/ILPD.csv", header=None)
# drop cat features
df = df.drop(columns=[1])
df = df.rename(columns={10:"label"})
df["label"] = df["label"].eq(1).astype(np.int8)
data_dict["ILPD1v2"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}



# wine 1vREST
df = pd.read_csv("data/wine.data", header=None)
df = df.rename(columns={0:"label"})
df["label"] = df["label"].eq(1).astype(np.int8)
data_dict["wine1vREST"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}


# wine 2vREST
df = pd.read_csv("data/wine.data", header=None)
df = df.rename(columns={0:"label"})
df["label"] = df["label"].eq(2).astype(np.int8)
data_dict["wine2vREST"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}

# wine 3vREST
df = pd.read_csv("data/wine.data", header=None)
df = df.rename(columns={0:"label"})
df["label"] = df["label"].eq(3).astype(np.int8)
data_dict["wine3vREST"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}


# hmeq
df = pd.read_csv("data/hmeq.csv")
df = df.fillna(df.mean())
# drop categorical
df = df.drop(columns=["REASON", "JOB"])
df = df.rename(columns={"BAD":"label"})
data_dict["Loan"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}


# glass 1vREST
df = pd.read_csv("data/glass.data", header=None, index_col=[0])
df = df.rename(columns={10:"label"})
df["label"] = df["label"].eq(1).astype(np.int8)
data_dict["glass1vREST"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}


# glass 2vREST
df = pd.read_csv("data/glass.data", header=None, index_col=[0])
df = df.rename(columns={10:"label"})
df["label"] = df["label"].eq(2).astype(np.int8)
data_dict["glass2vREST"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}


# glass 4vREST
df = pd.read_csv("data/glass.data", header=None, index_col=[0])
df = df.rename(columns={10:"label"})
df["label"] = df["label"].eq(4).astype(np.int8)
data_dict["glass4vREST"] = {
    "X_data": df.drop(columns=["label"]).astype("float32").to_numpy(),
    "y_data": df["label"].to_numpy()
}
