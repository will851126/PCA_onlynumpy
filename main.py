import  numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

import pandas as pd

from Algorithm import standardize_data
iris=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
,header=None)

iris.columns=["sepal_length","sepal_width",'petal_length','petal_width','species']

iris.dropna(how='all',inplace=True)



# Visualizing Data
# Plotting data using seaborn
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (12,8)
sns.scatterplot(x = iris.sepal_length, y=iris.sepal_width,
               hue = iris.species, style=iris.species)

# Standardizing the Data

def standardize_data(arr):
         
    '''
    This function standardize an array, its substracts mean value, 
    and then divide the standard deviation.
    
    param 1: array 
    return: standardized array
    '''    
    rows, columns = arr.shape
    
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    
    for column in range(columns):
        
        mean = np.mean(X[:,column])
        std = np.std(X[:,column])
        tempArray = np.empty(0)
        
        for element in X[:,column]:
            
            tempArray = np.append(tempArray, ((element - mean) / std))
 
        standardizedArray[:,column] = tempArray
    
    return standardizedArray

# Standardizing data
X = iris.iloc[:, 0:4].values
y = iris.species.values

X = standardize_data(X)


# Computing the Eigenvectors and Eigenvalues

covariance_matrix = np.cov(X.T)

eign_values,eign_vector= np.linalg.eig(covariance_matrix)

variance_explained = []
for i in eign_values:
     variance_explained.append((i/sum(eign_values))*100)
        




cumulative_variance_explained=np.cumsum(variance_explained)

sns.lineplot(x=[1,2,3,4], y=cumulative_variance_explained)

plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.title('Explained variance vs Number of components')

# Using two first components (because those explain mire than 95%)

project_matrix=(eign_vector.T[:][:2]).T

x_pca=X.dot(project_matrix)
print(x_pca)

