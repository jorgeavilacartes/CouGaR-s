import logging
import json
import pandas as pd
import numpy as np 
from itertools import combinations

from sklearn.metrics import (
    silhouette_score,
    #silhouette_samples,
    calinski_harabasz_score,
)
from src.gdv import GeneralizedDiscriminationValue as GDV

# logger
logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s - %(message)s")
_logger = logging.getLogger(__name__)


with open("data/train/order_output.json") as fp: 
    ORDER_OUTPUT = json.load(fp)

with open("data/train/labels.json") as fp:
    labels = json.load(fp)["test"]


embeddings  = pd.read_csv("data/test/embeddings-test.tsv",sep="\t",header=None)

# Global Calinski Harabasz Score
global_calinski_harabasz = calinski_harabasz_score(X=embeddings, labels = labels)
_logger.info("Calinski")

# Global silhouette score
global_silhouette = silhouette_score(X=embeddings, labels=labels)
_logger.info("Silhouette")

# Generalized Discrimination Value
# gdv = GDV(classes=ORDER_OUTPUT)
# global_gdv = gdv(embeddings, labels)
# _logger.info("GDV")
clust_metrics = {
    "silhouette": global_silhouette,
    "calinski_harabasz": global_calinski_harabasz,
    # "GDV": global_gdv
}
pd.DataFrame.from_dict(clust_metrics, orient='index').T.to_csv("data/test/clustering_metrics.csv")