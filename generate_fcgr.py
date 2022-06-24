import yaml
import tqdm
import numpy as np 
from pathlib import Path
from src.canonical_fcgr import FCGRKmc

# Load parameters
with open("parameters.yaml") as fp: 
    PARAMETERS = yaml.load(fp, Loader=yaml.FullLoader)

KMER = PARAMETERS["KMER"]
CANONICAL_KMERS = PARAMETERS["CANONICAL_KMERS"]
PATH_KMER_COUNT = PARAMETERS["PATH_KMER_COUNT"]

# save data generated
PATH_DATA = Path("data/fcgr-{KMER}mers")
PATH_DATA.mkdir(exist_ok=True, parents=True)

# instantiate fcgr class
fcgr = FCGRKmc(k=KMER, use_canonical_kmers=CANONICAL_KMERS)

# load path to files 
files = list(Path(PATH_KMER_COUNT).rglob("*txt.gz"))

for filename in tqdm(files, desc="Generating FCGR"):
    array = fcgr(filename)
    
    # save numpy 
    path_save = PATH_DATA.joinpath( filename.name.replace(".txt.gz",".npy") )
    np.save(path_save, array)

