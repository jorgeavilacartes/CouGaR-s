"""
Evaluation of embeddings based on 
paper Facenet
"""
import json
import numpy as np
import pandas as pd
from collections import namedtuple
from pathlib import Path
from tqdm import tqdm 

PATH_TEST = Path("data/test")

# load embeddings
embeddings = pd.read_csv(PATH_TEST.joinpath("embeddings.tsv"), sep="\t", header=None)
embeddings = embeddings.to_numpy()

# load labels
labels = pd.read_csv(PATH_TEST.joinpath("metadata.tsv"),header=None)
labels = labels[0].tolist()

## compute L2 distances D(xi,xj)
N_embeddings = embeddings.shape[0]
D = dict() # set to save distance between embeddings

# define sets according to the paper
P_same = set()
P_diff = set()

total = N_embeddings*(N_embeddings-1) - (N_embeddings-1)*N_embeddings/2
pbar = tqdm(total=total)# = N_embeddings**2 / 2 - N_embeddings)
for i in range(N_embeddings-1):
    for j in range(i+1,N_embeddings):
        
        # l2 distance between embeddings
        l2_dist = np.linalg.norm( embeddings[i,:] - embeddings[j,:] )
        D[(i,j)] = l2_dist
        
        # classify based on label: "same" or "diff"
        if labels[i] == labels[j]:
            P_same.update({(i,j)})
        else: 
            P_diff.update({(i,j)})

        pbar.update(1)
pbar.close()

##  Metrics defined in the paper
metrics = []
Metrics = namedtuple("MetricsFaceNet", ["d","P_same","P_diff", "TA", "FA", "VAL","FAR"])

# size of P_same and P_diff
P_same_size = len(P_same)
P_diff_size = len(P_diff)
# Compute VAL and FAR for different distance thresholds
for d in np.linspace(1e-10, 0.1, 10):
    print(d)
    # VAL: Validation rate
    TA = [pair for pair in P_same if D[pair] <= d] 
    TA_size = len(TA)
    VAL = TA_size / P_same_size

    # FAR: False accept rate
    FA = [pair for pair in P_diff if D[pair] <= d]
    FA_size = len(FA)
    FAR = FA_size / P_diff_size

    metrics.append(
        Metrics(d, P_same_size, P_diff_size, TA_size, FA_size, VAL, FAR)
    )

pd.DataFrame(metrics).to_csv("data/test/eval-embeddings.tsv",sep="\t")