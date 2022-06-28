"""
FCGR from spaced kmers 

"#": match character
"-": space

> pattern = "##-##--##" 
"""
import gzip
import numpy as np
from pathlib import Path
from tqdm import tqdm


class FCGRSpacedKmer(FCGR):
    
    def __init__(self, k, spaced_pattern: str): 
        super().__init__(k)
        self.spaced_pattern = spaced_pattern # "##--###-#"
        assert spaced_pattern.count("#") == self.k, "number of '#' in the pattern must be equal to kmer size"
        
    def __call__(self, kmc_output,):
        
        # collecto kmer counting in a dictionary
        kmer_count = {}
        if str(kmc_output).endswith(".txt"):
            with open(kmc_output) as fp:
                for line in fp:
                    kmer, freq = line.split("\t")
                    kmer_count[kmer]=int(freq)
                    
        else:
            with gzip.open(kmc_output,'rt') as f:
                for line in f:
                    line = line.strip()
                    kmer, freq = line.split()
                    kmer_count[kmer]=int(freq)

        # Create an empty array to save the FCGR values
        array_size = int(2**self.k)
        fcgr = np.zeros((array_size,array_size))
        
        for spaced_kmer, freq in kmer_count.items(): 
            kmer = "".join([spaced_kmer[j] for j,c in enumerate(self.spaced_pattern) if c=="#" ])
            pos_x, pos_y = self.kmer2pixel[kmer]
            fcgr[int(pos_x)-1,int(pos_y)-1] += freq
        
        return fcgr