# RPI-SE: A Stacking Ensemble Learning Framework for ncRNA-Protein Interactions Prediction Using Sequence Information

The interactions between non-coding RNAs and proteins play an essential role in many biological processes. A various of high-throughput experimental methods have been applied to detect ncRNA-protein interactions. However，they are time-consuming and expensive. In this study, we presented a novel stacking heterogeneous ensemble learning framework, RPI-SE, using logistic regression to integrate gradient boosting machine, support vector machine and extremely randomized tree algorithms, for effectively predicting ncRNA-protein interactions. More specifically, to fully exploit protein and RNA sequence information, Position Weight Matrix combined with Legendre Moments was applied to obtain protein evolutionary information. Meanwhile, k-mer sparse matrix was employed to extract efficient features from ncRNA sequences. Then, these discriminative features were fed into the ensemble learning framework to learn how to predict ncRNA-protein interactions. The accuracy and robustness of RPI-SE is evaluated on different types of dataset, including RNA-protein interactions and long ncRNA-protein interactions datasets, and compared with other state-of-the-art methods. Experimental results demonstrate that our method RPI-SE is competent for ncRNA-protein interactions prediction task and can achieve great prediction performance with high accurate and robustness. It’s anticipated that this study could advance the ncRNA-protein interactions related biomedical research.  

## Dependency:  
python 3.5  
Numpy  
XGBoost  
scikit-learn  

## Usage:  
python RPI-SE.py -datatype=RPI369  
Where RPI369 is RNA-protein interaction dataset, and RPI-SE will do 5-fold cross-validation for it. you can also choose other datasets, such as RPI488, RPI1807.  

## Reference:  
Yi, H., You, Z., Wang, M. et al. RPI-SE: a stacking ensemble learning framework for ncRNA-protein interactions prediction using sequence information. BMC Bioinformatics 21, 60 (2020).
