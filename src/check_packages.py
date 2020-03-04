import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import pandas as pd
from IPython.display import display


# Create a 2D NumPy array with a diagonal of ones, and zeros everywhere else
eye = np.eye(4)
print('Numpy Array:\n{}'.format(eye))
# Convert the NumPy array to a SciPy sparse matrix in CSR format
# Only the nonzero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))

arr = np.array([[1,2,3], [4,5,6]])
print('Array:\n{}'.format(arr))

# Generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# Create a second array using sine
y = np.sin(x)
# The plot function makes a line chart of one array against another
plt.plot(x, y, marker='x')

# create a simple dataset of people
data = {'Name': ["John", "Anna", "Peter", "Linda"],
'Location' : ["New York", "Paris", "Berlin", "London"],
'Age' : [24, 13, 53, 33]
}

data_pandas = pd.DataFrame(data)
# IPython.display allows "pretty printing" of dataframes
# in the Jupyter notebook
display(data_pandas)
display(data_pandas[data_pandas.Age>30])