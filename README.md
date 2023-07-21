## LogisticPCA: Principal Component Analysis with Binary Data
LogisticPCA provides a method to perform Principal Component Analysis (PCA) on binary data using methods outlined in the original paper, [Landgraf and Lee, 2015](https://arxiv.org/pdf/1510.06112.pdf).

### Features
The LogisticPCA package provides a python implementation of logisitc PCA. A LogisticPCA object can be trained on data and used to transform new data in the same feature space. For example usage, see the demonstration.ipynb notebook. 

### Installation
This code doesn't require a special installation process if Python and necessary libraries are already installed. Download the LPCA.py file and import the LogisticPCA class in your Python script:

```python
from LPCA import LogisticPCA
```

The required dependencies are: NumPy, Joblib, and h5py. 

### Disclaimer
This package is a demonstration and may not be suited for production-level tasks without additional modifications and error handling. Use it at your own risk.

