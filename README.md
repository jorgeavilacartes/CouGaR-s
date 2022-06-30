# CouGaR-s

Classification of Sars-Cov2 samples with Similarity Learning and k-NN

1. FCGR is created for each sample, using either:
    - canonical k-mers, or
    - spaced k-mers, or
    - all k-mers

2. An embeddings space is generated from FCGR through a Deep Neural Network using Triplet loss.

3. k-NN is used to train a classifier taking as input the embeddings obtained in step 2.