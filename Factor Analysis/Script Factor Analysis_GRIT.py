# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 18:53:24 2023

@author: darod
"""

# Import libraries


import pandas as pd
import os
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import numpy as np

# Get the working directory

print(os.getcwd())


# Import data and transform it into pandas data frame

GRIT = pd.read_csv('Matriz GRIT.csv', delimiter=";")
print(GRIT)

# Pre-processing of data

GRIT.dropna(inplace=True)
GRIT.info()
GRIT.head()

# Adequacy Test

    # Bartlestt test: checks whether or not the observed variables intercorrelate at all using the observed correlation matrix against the identity matrix
    #If is insignificant, you should not employ a factor analysis.
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(GRIT)
chi_square_value, p_value

    #Kaiser_Mayer_Olkin test: measures the suitability of data for factor analysis.
    #It determines the adequacy for each observed variable and for the complete model
    #KMO values range between 0 and 1. Value of KMO less than 0.6 is considered inadequate.
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(GRIT)
kmo_model

# Choosing the Number of Factors to use

    # Kaiser Criterion
    
    # Perform factor analysis Exploratory Factorial Analysis (EFA)
EFA_G = FactorAnalyzer(10, rotation=None)
EFA_G.fit(GRIT) 
EFA_fit_G = EFA_G.fit_transform(GRIT)
EFA_fit_G

    # Check Eigenvalues, Eigenvalues represent variance explained by each factor from the total variance
    #Eigenvalues greater than 1 means its a factor
ev_G = EFA_G.get_eigenvalues()
print (ev_G)

    # Get factor loadings, factor loading is a matrix which shows the relationship of each variable to the underlying factor. It shows the correlation coefficient for observed variable and factor 

EFA_G.loadings_
Loadings_G_df = pd.DataFrame(EFA_G.loadings_, index = GRIT.columns, columns = ["Factor 1", "Factor 2", "Factor 3", "Factor 4", "Factor 5", "Factor 6", "Factor 7", "Factor 8", "Factor 9", "Factor 10"])
print (Loadings_G_df)
    #Scree plot

GRIT.shape[1] # Get the lenght of the Antifragility dataframe
len(ev_G)                # Lenght of the array 

plt.scatter(range(1,GRIT.shape[1]+1),ev_G[1])    #Sequence of numbers that represent the number of fators. The range starts in 1 and ends in the number of columns of Antifragility df + 1
plt.plot(range(1,GRIT.shape[1]+1),ev_G[1])
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.axhline(y = 1 , color='r', linestyle='--')
plt.show()

# Now, we are making Confirmatory Factor Analysis (CFA) with the numbers of factors {x} obtained from the EFA

CFA_G = FactorAnalyzer(3, rotation="varimax")
CFA_G.fit(GRIT)
CFA_G.loadings_
Loadings_cfa_G = pd.DataFrame(CFA_G.loadings_, index = GRIT.columns, columns = ["Factor 1", "Factor 2", "Factor 3"])
print(Loadings_cfa_G)
# Get variance of each factors if the CFA is well-performed
FV_GRIT = CFA_G.get_factor_variance()
print(FV_GRIT)

