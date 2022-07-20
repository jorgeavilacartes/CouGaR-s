import yaml
from tqdm import tqdm
import numpy as np 
from pathlib import Path
from src.canonical_fcgr import FCGRKmc
from src.spaced_fcgr import FCGRSpacedKmer

# Load parameters
with open("parameters.yaml") as fp: 
    PARAMETERS = yaml.load(fp, Loader=yaml.FullLoader)

KMER = PARAMETERS["KMER"]
CANONICAL_KMERS = PARAMETERS["CANONICAL_KMERS"]
SPACED_KMERS = PARAMETERS["SPACED_KMERS"]
SPACED_PATTERN = PARAMETERS["SPACED_PATTERN"]
PATH_KMER_COUNT = PARAMETERS["PATH_KMER_COUNT"]

# instantiate fcgr class
if CANONICAL_KMERS is True:
    # save data generated
    PATH_DATA = Path(f"data/fcgr-{KMER}mers-canonical")
    PATH_DATA.mkdir(exist_ok=True, parents=True)

    fcgr = FCGRKmc(k=KMER, use_canonical_kmers=CANONICAL_KMERS)
elif SPACED_KMERS is True:
    PATH_DATA = Path(f"data/fcgr-{KMER}mers-spaced")
    PATH_DATA.mkdir(exist_ok=True, parents=True)
    fcgr = FCGRSpacedKmer(k=KMER,spaced_pattern=SPACED_KMERS)
else: 
    PATH_DATA = Path(f"data/fcgr-{KMER}mers")
    PATH_DATA.mkdir(exist_ok=True, parents=True)
    fcgr = FCGRKmc(k=KMER, use_canonical_kmers=False)
    
# load path to files 
files = list(Path(PATH_KMER_COUNT).rglob("*txt.gz"))

for filename in tqdm(files, desc="Generating FCGR"):
    array = fcgr(filename)
    
    # save numpy 
    path_save = PATH_DATA.joinpath( filename.name.replace(".txt.gz",".npy") )
    np.save(path_save, array)

