#!/usr/bin/env python
# coding: utf-8

# # Initialisation

# In[2]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))


# In[5]:


import os


import re
import pickle
import numpy as np
import seaborn as sns


from pytz import timezone
from datetime import datetime
from collections import namedtuple
from sklearn.preprocessing import OrdinalEncoder

from sklearn.cluster import KMeans
from sklearn.neighbors import DistanceMetric


def SaveVariable(Variable, FileName):
    with open(FileName, 'wb') as io:
        pickle.dump(Variable, io)
    
def LoadVariable(FileName):
    with open(FileName, "rb") as io:
        Res = pickle.load(io)
    return Res

print("Num threads set to:", os.cpu_count())
print("Ran on (" + datetime.now(timezone('Europe/Amsterdam')).strftime("%a, %Y-%m-%d %H:%M %Z %z") + ")")

from performance_function import calculate_performance_numpy

from python_speech_features import mfcc


# # Data

# In[4]:


SupervisedType = "Classification"
RandomState = 1337

X_Train, Y_Train = LoadVariable(f"{os.getcwd()}/raw_data.pkl")
X_Test = LoadVariable(f"{os.getcwd()}/test_data (2).pkl")[0]
IntEncoder = OrdinalEncoder()
Y_Train = IntEncoder.fit_transform([[x] for x in Y_Train]).squeeze()
# Y_Test = IntEncoder.transform([[x] for x in Y_Test]).squeeze()

N = X_Train.shape[0]
Y_TrainUnique = set(Y_Train)
# Y_TestUnique = set(Y_Test)
print(f"Unique Values: Y_Train ({len(Y_TrainUnique)})") #, Y_Test ({len(Y_TestUnique)}), all values are the same: {Y_TrainUnique == Y_TestUnique}")


print("X_Train:", type(X_Train), X_Train.shape, X_Train.min(), X_Train.max())
print("X_Test :", type(X_Test), X_Test.shape, X_Test.min(), X_Test.max())

print("Y_Train:", type(Y_Train), Y_Train.shape, min(Y_Train), max(Y_Train), len(set(Y_Train)))
# print("Y_Test: ", type(Y_Test), Y_Test.shape, min(Y_Test), max(Y_Test), len(set(Y_Test)))

print("\nY_Train[:10]:", Y_Train[:10])

# number of classes
K = len(set(Y_Train)) if SupervisedType.lower() == "classification" and len(set(Y_Train)) != 2 else (Y_Train.shape[1] if SupervisedType.lower() == "multivariateregression" else 1) #An output_size (K) > 1 can be either Multiclass or Multivariate-Regression, like Lat/Lon coordinates

if 'train_dataset' in locals() or 'train_dataset' in globals():
    tmpX, tmpY = next(iter(train_loader))
    NonSingularDims = np.sum([1 for DimVal in tmpX.shape if DimVal > 1])
    if NonSingularDims == 2:
        N, D = [DimVal for DimVal in tmpX.shape if DimVal > 1]
        H1, W1 = (0, 0)
    elif NonSingularDims == 3:
        D = 0
        N, H1, W1 =[DimVal for DimVal in tmpX.shape if DimVal > 1] #This is RNN NxTxD
    elif NonSingularDims == 4:
        N, H1, W1, D = [DimVal for DimVal in tmpX.shape if DimVal > 1]
    
else:
    if len(X_Train.shape) == 2:
        N, D = X_Train.shape
        H1, W1 = (0, 0)
    elif len(X_Train.shape) == 3:
        D = 0
        N, H1, W1 = X_Train.shape #This is a Picture with no Colour, not RNN
    elif len(X_Train.shape) == 4:
        N, H1, W1, D = X_Train.shape

print()
print("X_Train.shape", X_Train.shape, " Y_Train.shape", Y_Train.shape)
print("X_Test.shape ", X_Test.shape)#, " Y_Test.shape ", Y_Test.shape)
print("K         ", K)
print("N:", N, "H1:", H1, "W1:", W1, "D:", D)
if 'train_dataset' in locals() or 'train_dataset' in globals():
    print(f"\nData after transformation with batch size = {batch_size}:")
    print("X.shape", tuple(tmpX.shape), "\tY.shape", tuple(tmpY.shape))


# In[5]:


MFCC = np.array([mfcc(X_Train[i], 11025).flatten() for i in range(X_Train.shape[0])])
print("MFCC.shape", MFCC.shape)


# In[ ]:


# from python_speech_features import fbank
# FBANK = np.array([(np.append(fbank(X_Train[i], 11025)[0].flatten(), fbank(X_Train[i], 11025)[1].flatten())) for i in range(X_Train.shape[0])])
# print("fbank.shape", FBANK.shape)


# In[54]:


# from python_speech_features import logfbank
# LogFBANK = np.array([logfbank(X_Train[i], 11025).flatten() for i in range(X_Train.shape[0])])
# print("logfbank.shape", LogFBANK.shape)


# In[61]:


# from python_speech_features import ssc
# SSC = np.array([ssc(X_Train[i], 11025).flatten() for i in range(X_Train.shape[0])])
# print("ssc.shape", SSC.shape)


# # Model

# Euclidean Distance: sqrt(sum((x - y)^2))

# In[ ]:


Euclidean = DistanceMetric.get_metric('euclidean')
EucDist = Euclidean.pairwise(MFCC)

cmc_scores, acc = calculate_performance_numpy(EucDist, Y_Train)
print(f"{(acc * 100):.2f}% accuracy with the (Euclidean) Metric")
print("The cmc scores are:", cmc_scores)


# Manhattan Distance sum(|x - y|)

# In[ ]:


Manhattan = DistanceMetric.get_metric('manhattan')
ManhDist = Manhattan.pairwise(MFCC)

cmc_scores, acc = calculate_performance_numpy(ManhDist, Y_Train)
print(f"{(acc * 100):.2f}% accuracy with the (Manhattan) Metric")
print("The cmc scores are:", cmc_scores)


# Chebyshev Distance max(|x - y|)

# In[ ]:


Chebyshev = DistanceMetric.get_metric('chebyshev')
ChebDist = Chebyshev.pairwise(MFCC)

cmc_scores, acc = calculate_performance_numpy(ChebDist, Y_Train)
print(f"{(acc * 100):.2f}% accuracy with the (Chebyshev) Metric")
print("The cmc scores are:", cmc_scores)


# Hamming Distance N_unequal(x, y) / N_tot

# In[ ]:


Hamming = DistanceMetric.get_metric('hamming')
HammDist = Hamming.pairwise(MFCC)

cmc_scores, acc = calculate_performance_numpy(HammDist, Y_Train)
print(f"{(acc * 100):.2f}% accuracy with the (Hamming) Metric")
print("The cmc scores are:", cmc_scores)


# Canberra Distance sum(|x - y| / (|x| + |y|))

# In[ ]:


Canberra = DistanceMetric.get_metric('canberra')
CanbDist = Canberra.pairwise(MFCC)

cmc_scores, acc = calculate_performance_numpy(CanbDist, Y_Train)
print(f"{(acc * 100):.2f}% accuracy with the (Canberra) Metric")
print("The cmc scores are:", cmc_scores)


# BrayCurtis Distance sum(|x - y|) / (sum(|x|) + sum(|y|))

# In[6]:


Braycurtis = DistanceMetric.get_metric('braycurtis')
BraycDist = Braycurtis.pairwise(MFCC)

cmc_scores, acc = calculate_performance_numpy(BraycDist, Y_Train)
print(f"{(acc * 100):.2f}% accuracy with the (Braycurtis) Metric")
print("The cmc scores are:", cmc_scores)


# In[6]:


Braycurtis = DistanceMetric.get_metric('braycurtis')
BraycDist = Braycurtis.pairwise(MFCC)

cmc_scores, acc = calculate_performance_numpy(BraycDist, Y_Train)
print(f"{(acc * 100):.2f}% accuracy with the (Braycurtis) Metric")
print("The cmc scores are:", cmc_scores)


# # Saving the Results

# In[8]:


MFCC_Test = np.array([mfcc(X_Test[i], 11025).flatten() for i in range(X_Test.shape[0])])
print("MFCC_Test.shape", MFCC_Test.shape)

Euclidean = DistanceMetric.get_metric('euclidean')
EucDist = Euclidean.pairwise(MFCC_Test)
np.savetxt("euclidean_answer.txt", EucDist, delimiter=';')

Manhattan = DistanceMetric.get_metric('manhattan')
ManhDist = Manhattan.pairwise(MFCC_Test)
np.savetxt("manhattan_answer.txt", ManhDist, delimiter=';')

Chebyshev = DistanceMetric.get_metric('chebyshev')
ChebDist = Chebyshev.pairwise(MFCC_Test)
np.savetxt("chebyshev_answer.txt", ChebDist, delimiter=';')

Hamming = DistanceMetric.get_metric('hamming')
HammDist = Hamming.pairwise(MFCC_Test)
np.savetxt("hamming_answer.txt", HammDist, delimiter=';')

Canberra = DistanceMetric.get_metric('canberra')
CanbDist = Canberra.pairwise(MFCC_Test)
np.savetxt("canberra_answer.txt", CanbDist, delimiter=';')

Braycurtis = DistanceMetric.get_metric('braycurtis')
BraycDist = Braycurtis.pairwise(MFCC_Test)
np.savetxt("braycurtis_answer.txt", BraycDist, delimiter=';')

