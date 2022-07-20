import yaml
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from src.model_loader import ModelLoader
from src.data_generator import DataGenerator    
from src.callbacks import CSVTimeHistory
from src.pipeline import Pipeline


# Load parameters
with open("parameters.yaml") as fp: 
    PARAMETERS = yaml.load(fp, Loader=yaml.FullLoader)

print(">> train model <<")

# General parameters
KMER = PARAMETERS["KMER"]
EMBEDDING_SIZE = PARAMETERS["EMBEDDING_SIZE"]

# Train parameters
BATCH_SIZE = PARAMETERS["BATCH_SIZE"]
EPOCHS     = PARAMETERS["EPOCHS"]
WEIGHTS    = PARAMETERS["WEIGHTS_PATH"]
PREPROCESSING = PARAMETERS["PREPROCESSING"]
PATIENTE_EARLY_STOPPING = PARAMETERS["PATIENTE_EARLY_STOPPING"]
PATIENTE_EARLY_LR = PARAMETERS["PATIENTE_REDUCE_LR"]

SEED = PARAMETERS["SEED"]

# set seed for reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)

# -1- Model selection
loader = ModelLoader()
model  = loader(
    k=KMER,
    embedding_size=EMBEDDING_SIZE,
    model_name="cnn_kmers",
    weights_path=WEIGHTS,
    )

model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=tfa.losses.TripletSemiHardLoss()
)

preprocessing = Pipeline(PREPROCESSING)
Path("data/train").mkdir(exist_ok=True, parents=True)
preprocessing.asJSON("data/train/preprocessing.json")


# -2- Datasets
# load list of fcgr for train and validation sets
with open("data/train/datasets.json","r") as f:
    datasets = json.load(f)

list_train = datasets["train"]
list_val   = datasets["val"]

with open("data/train/labels.json","r") as f:
    labels = json.load(f)

labels_train = labels["train"]
labels_val = labels["val"]

# order labels in alphabetical order
order_output = list(set(labels_train).union(set(labels_val)))
order_output.sort()
with open("data/train/order_output.json", "w") as fp:
    json.dump(order_output, fp,)

# Instantiate DataGenerator for training set
ds_train = DataGenerator(
    list_train,
    labels_train,
    order_output_model = order_output,
    batch_size = BATCH_SIZE,
    shuffle=True
)

# Instantiate DataGenerator for validation set
ds_val = DataGenerator(
    list_val,
    labels_val,
    order_output_model = order_output,
    batch_size = BATCH_SIZE,
    shuffle=False,
) 

# -3- Training
# - Callbacks
# checkpoint: save best weights
Path("data/train/checkpoints").mkdir(exist_ok=True, parents=True)
cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='data/train/checkpoints/weights-{epoch:02d}-{val_loss:.3f}.hdf5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

# reduce learning rate
cb_reducelr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    mode='min',
    factor=0.1,
    patience=PATIENTE_EARLY_LR,
    verbose=1,
    min_lr=0.00001
)

# stop training if
cb_earlystop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    min_delta=0.001,
    patience=PATIENTE_EARLY_STOPPING,
    verbose=1
)

# save history of training
Path("data/train").mkdir(exist_ok=True, parents=True)
cb_csvlogger = tf.keras.callbacks.CSVLogger(
    filename='data/train/training_log.csv',
    separator=',',
    append=False
)

cb_csvtime = CSVTimeHistory(
    filename='data/train/time_log.csv',
    separator=',',
    append=False
)

model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
    callbacks=[
        cb_checkpoint,
        cb_reducelr,
        cb_earlystop,
        cb_csvlogger,
        cb_csvtime
        ]
)