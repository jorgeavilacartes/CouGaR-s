import yaml
import io
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.model_loader import ModelLoader
from src.data_generator import DataGenerator

# Load parameters
with open("parameters.yaml") as fp: 
    PARAMETERS = yaml.load(fp, Loader=yaml.FullLoader)

# load order output model
with open("data/train/order_output.json") as fp: 
    ORDER_OUTPUT = json.load(fp)

print(">> test model <<")

PATH_TEST = Path("data/test")
PATH_TEST.mkdir(exist_ok=True,parents=True)

BATCH_SIZE = PARAMETERS["BATCH_SIZE"]
KMER = PARAMETERS["KMER"]
EMBEDDING_SIZE = PARAMETERS["EMBEDDING_SIZE"]

# get best weights
CHECKPOINTS  = [str(path) for path in Path("data/train/checkpoints").rglob("*.hdf5")]
epoch_from_chkp = lambda chkp: int(chkp.split("/")[-1].split("-")[1])
CHECKPOINTS.sort(key = epoch_from_chkp)
BEST_WEIGHTS =  CHECKPOINTS[-1]
print(f"using weights {BEST_WEIGHTS} to test")

# -1- Load model
loader = ModelLoader()
model  = loader(
            k=KMER,
            embedding_size=EMBEDDING_SIZE,
            model_name="cnn_kmers", 
            weights_path=BEST_WEIGHTS,
            ) # model from src/models

# -2- Datasets
# load list of images for train and validation sets
with open("data/train/datasets.json","r") as f:
    datasets = json.load(f)
list_test = datasets["test"]

with open("data/train/labels.json","r") as f:
    labels = json.load(f)

labels_test = labels["test"]

ds_test = DataGenerator(
    list_test,
    labels_test,
    order_output_model = ORDER_OUTPUT,
    batch_size = BATCH_SIZE,
    shuffle=False
)

# Save test embeddings for visualization in projector
results = model.predict(ds_test)
np.savetxt(PATH_TEST.joinpath("embeddings.tsv"), results, delimiter="\t")

# Save labels to plot
out_m = io.open(PATH_TEST.joinpath("metadata.tsv"), "w", encoding="utf-8")
for label in labels_test:
    [out_m.write(str(label) + "\n")]
out_m.close()