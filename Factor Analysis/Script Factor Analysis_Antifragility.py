
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

Antifragility = pd.read_csv('Matriz antifragilidad.csv', delimiter=";")
print(Antifragility)

# Pre-processing of data

Antifragility.dropna(inplace=True)
Antifragility.info()
Antifragility.head()

# Adequacy Test

    # Bartlestt test: checks whether or not the observed variables intercorrelate at all using the observed correlation matrix against the identity matrix
    #If is insignificant, you should not employ a factor analysis.
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(Antifragility)
chi_square_value, p_value

    #Kaiser_Mayer_Olkin test: measures the suitability of data for factor analysis.
    #It determines the adequacy for each observed variable and for the complete model
    #KMO values range between 0 and 1. Value of KMO less than 0.6 is considered inadequate.
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(Antifragility)
kmo_model

# Choosing the Number of Factors to use

    # Kaiser Criterion
    
    # Perform factor analysis Exploratory Factorial Analysis (EFA)
EFA = FactorAnalyzer(10, rotation=None)
EFA.fit(Antifragility) 
EFA_fit = EFA.fit_transform(Antifragility)
EFA_fit

    # Check Eigenvalues, Eigenvalues represent variance explained by each factor from the total variance
    #Eigenvalues greater than 1 means its a factor
ev = EFA.get_eigenvalues()
print (ev)

    # Get factor loadings, factor loading is a matrix which shows the relationship of each variable to the underlying factor. It shows the correlation coefficient for observed variable and factor 

EFA.loadings_
Loadings_df = pd.DataFrame(EFA.loadings_, index = Antifragility.columns, columns = ["Factor 1", "Factor 2", "Factor 3", "Factor 4", "Factor 5", "Factor 6", "Factor 7", "Factor 8", "Factor 9", "Factor 10"])
print (Loadings_df)
    #Scree plot

Antifragility.shape[1] # Get the lenght of the Antifragility dataframe
len(ev)                # Lenght of the array 

plt.scatter(range(1,Antifragility.shape[1]+1),ev[1])    #Sequence of numbers that represent the number of fators. The range starts in 1 and ends in the number of columns of Antifragility df + 1
plt.plot(range(1,Antifragility.shape[1]+1),ev[1])
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.axhline(y = 1 , color='r', linestyle='--')
plt.show()

# Now, we are making Confirmatory Factor Analysis (CFA) with the numbers of factors {x} obtained from the EFA

CFA = FactorAnalyzer(4, rotation="varimax")
CFA.fit(Antifragility)
CFA.loadings_
Loadings_cfa = pd.DataFrame(CFA.loadings_, index = Antifragility.columns, columns = ["Factor 1", "Factor 2", "Factor 3", "Factor 4"])
print(Loadings_cfa)
# Get variance of each factors if the CFA is well-performed
CFA.get_factor_variance()


