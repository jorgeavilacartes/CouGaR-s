"""
As mentioned in the paper, perform clustering using the embeddings 
and k-NN

1. Use train+val used for the embeddings as train set for k-NN
    - generate embeddings
2. Train and Test (same test set than embeddings) using k-NN, and report
    - accuracy, precision, recall
    - AUC
"""
import json
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
from collections import namedtuple

import matplotlib.pyplot as plt 
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

# load list of images for train and validation sets
with open("data/train/datasets.json","r") as f:
    datasets = json.load(f)

with open("data/train/labels.json","r") as f:
    labels = json.load(f)

# define train and test sets
# input
X_train = pd.concat([
    pd.read_csv("data/test/embeddings-train.tsv",sep="\t",header=None),
    pd.read_csv("data/test/embeddings-val.tsv",sep="\t", header=None)
    ],
    axis=0
)

# labels
y_train = labels["train"] + labels["val"]

# Train k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Save trained model
PATH_CLUSTERING = Path("data/knn")
PATH_CLUSTERING.mkdir(parents=True, exist_ok=True)
with open(PATH_CLUSTERING.joinpath("knn.pkl"),"wb") as fp: 
    pickle.dump(knn, fp)

# Test model
X_test = pd.read_csv("data/test/embeddings-test.tsv", sep="\t", header=None)
y_test = labels["test"]
y_pred = knn.predict(X_test)

df = pd.DataFrame({"id": datasets["test"], "gt": y_test, "pred":y_pred})

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred,)
plt.title(f"Confusion matrix")
plt.savefig(PATH_CLUSTERING.joinpath(f"confusion_matrix.pdf"))

# Metrics
# load order output model
with open("data/train/order_output.json") as fp: 
    ORDER_OUTPUT = json.load(fp)
precision, recall, fscore, support = precision_recall_fscore_support(
                                    y_true=y_test, 
                                    y_pred=y_pred, 
                                    average=None, 
                                    labels=ORDER_OUTPUT,
                                    zero_division=0
                                    )


list_metrics = []
Metrics = namedtuple("Metrics", ["clade","precision", "recall", "fscore", "support"])
for j,label in enumerate(ORDER_OUTPUT): 
    list_metrics.append(
        Metrics(label, precision[j], recall[j], fscore[j], support[j])
    )

df_metrics = pd.DataFrame(list_metrics)
df_metrics.to_csv(PATH_CLUSTERING.joinpath("metrics.csv"))
