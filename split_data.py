from collections import defaultdict
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from src.data_selector import DataSelector

print(">> train val test split <<")

# Load parameters
with open("parameters.yaml") as fp: 
    PARAMETERS = yaml.load(fp, Loader=yaml.FullLoader)

KMER = PARAMETERS["KMER"]
CANONICAL_KMERS = PARAMETERS["CANONICAL_KMERS"]
SPACED_KMERS = PARAMETERS["SPACED_KMERS"]
MIN_SAMPLES_LABEL = PARAMETERS["MIN_SAMPLES_LABEL"]
MAX_SAMPLES_LABEL = PARAMETERS["MAX_SAMPLES_LABEL"]
SEED = PARAMETERS["SEED"]


# folder with fcgr
if CANONICAL_KMERS is True:
    FOLDER_FCGR = Path(f"data/fcgr-{KMER}mers-canonical")
elif SPACED_KMERS is True:
    FOLDER_FCGR = Path(f"data/fcgr-{KMER}mers-spaced")
else:
    FOLDER_FCGR = Path(f"data/fcgr-{KMER}mers")

PATH_LABELS = Path(PARAMETERS["PATH_LABELS"])
LABELS_TO_USE = PARAMETERS["LABELS_TO_USE"]
LIST_FCGR   = list(FOLDER_FCGR.rglob("*npy"))
TRAIN_SIZE   = float(PARAMETERS["TRAIN_SIZE"]) 

PATH_SAVE = Path("data/train/")
PATH_SAVE.mkdir(exist_ok=True, parents=True)

## Input for DataSelector
# sra-id
sra_id  = [path.stem for path in LIST_FCGR] # SRA_ID to get labels

# labels
df_labels = pd.read_csv(PATH_LABELS, sep="\t")
# df_labels.drop_duplicates("SRR_ID", inplace=True)

# consider only lables with representativity >= MIN_SAMPLE_LABEL 
col_label = "Clade" if LABELS_TO_USE=="GISAID" else "PANGO_LINEAGE"
label_dist = df_labels.groupby(col_label).size()
admisible_labels = label_dist[label_dist >= MIN_SAMPLES_LABEL].index
dict_labels = {sra_id: label for sra_id,label in zip(df_labels["SRR_ID"], df_labels[col_label])}

# select (filtered by frequency) SRR_ID and labels
count_by_label = defaultdict(int)
id_labels = []
labels    = []
for path in LIST_FCGR:
    label = dict_labels.get(path.stem)

    # check if label has enough representativity MIN_SAMPLES_LABEL
    if label in admisible_labels:

        # include sample if and only if not exceed MAX_SAMPLES_LABEL
        if count_by_label.get(label,0) < MAX_SAMPLES_LABEL:
            id_labels.append(str(path)) # path to fcgr
            labels.append(label)
            
            # update count label
            count_by_label[label] += 1

# Instantiate DataSelector
ds = DataSelector(id_labels, labels, seed=SEED)

# Get train, test and val sets
ds(train_size=TRAIN_SIZE, balanced_on=labels)

with open(PATH_SAVE.joinpath("datasets.json"), "w", encoding="utf-8") as f: 
    json.dump(ds.datasets["id_labels"], f, ensure_ascii=False, indent=4)

with open(PATH_SAVE.joinpath("labels.json"), "w", encoding="utf-8") as f: 
    json.dump(ds.datasets["labels"], f, ensure_ascii=False, indent=4)

# Summary of data selected 
summary_labels =  pd.DataFrame(ds.get_summary_labels())
summary_labels["Total"] = summary_labels.sum(axis=1)
summary_labels.to_csv(PATH_SAVE.joinpath("summary_labels.csv"))