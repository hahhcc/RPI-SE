# RPI-SE

we proposed an ensemble learning framework to predict ncRNA-Protein interactions using PSSM with legendre moments and TruncatedSVD for protein sequences and k-mer sparse matrix with SVD for (non-coding)RNA sequences, named RPI-SE, which made use of high-level evolutionary information and further improve its performance using stacked ensembling.

Dependency: 
python 3.5
Numpy
XGBoost
scikit-learn

Usage: python RPI-SE.py -datatype=RPI369 
where RPI369 is RNA-protein interaction dataset, and RPI-SE will do 5-fold cross-validation for it. you can also choose other datasets, such as RPI488, RPI1807. 

Reference 
Not available, we will add it later.
