# Principal Component Analysis with NumPy

## Description
In this project, I will apply PCA to a dataset without using any of the popular machine learning libraries such as scikit-learn and statsmodels. The goal of this document is to have a deeper understanding of the PCA fundamentals using functions just from NumPy library.

### Loading libraries and data
```%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
```
### Dataset
This is the classic database to be found in the pattern recognition literature. The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. Retrieved from UCI Machine Learning

### Visualizing Data
```plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12,8)
sns.scatterplot(x = iris.sepal_length, y=iris.sepal_width,
               hue = iris.species, style=iris.species)
```

### Standardizing the Data
Before applying PCA, the variables will be standardized to have a mean of 0 and a standard deviation of 1. This is important because all variables go through the origin point (where the value of all axes is 0) and share the same variance.
```
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
```

### Computing the Eigenvectors and Eigenvalues

1. Calculating the covariance matrix

Now I will find the covariance matrix of the dataset by multiplying the matrix of features by its transpose. It is a measure of how much each of the dimensions varies from the mean with respect to each other. Covariance matrices, like correlation matrices, contain information about the amount of variance shared between pairs of variables.

Eigenvectors are the principal components. The first principal component is the first column with values of 0.52, -0.26, 0.58, and 0.56. The second principal component is the second column and so on. Each Eigenvector will correspond to an Eigenvalue, each eigenvector can be scaled of its eigenvalue, whose magnitude indicates how much of the data’s variability is explained by its eigenvector.


### Picking Principal Components Using the Explained Variance

I want to see how much of the variance in data is explained by each one of these components. It is a convention to use 95% explained variance

72.77% of the variance on our data is explained by the first principal component, the second principal component explains 23.03% of data.



### Determining how many components

Some rules to guide in choosing the number of components to keep:

* Keep components with eigenvalues greater than 1, as they add value (because they contain more information than a single variable). This rule tends to keep more components than is ideal

* Visualize the eigenvalues in order from highest to lowest, connecting them with a line. Upon visual inspection, keep all the components whose eigenvalue falls above the point where the slope of the line changes the most drastically, also called the “elbow”

* Including variance cutoffs where we only keep components that explain at least 95% of the variance in the data

* Keep comes down the reasons for doing PCA


### Project Data Onto Lower-Dimensional Linear Subspace

In this last step, I will compute the PCA transformation on the original dataset, getting the dot product of the original standardized X and the eigenvectors that I got from the eigendecomposition.




### Conclusions

* PCA transformation was implemented using this NumPy functions:
    * np.mean()
    * np.std()
    * np.zeros()
    * np.empty()
    * np.cov( )
    * np.linalg.eig()
    * np.linalg.svd() It is an alternative to get eigenvalues and eigenvectors
    * np.cumsum()
    * np.dot()

* The benefit of PCA is that there will be fewer components than variables, thus simplifying the data space and mitigating the curse of dimensionality

* PCA is also best used when the data is linear because it is projecting it onto a linear subspace spanned by the eigenvectors

* Using PCA, it is going to project our data into directions that maximize the variance along the axes

* Scikit-learn has libraries to apply PCA