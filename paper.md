---
title: 'Evolvingfuzzysystems: A new Python library for evolving fuzzy inference'
authors:
  - name: "Kaike Sa Teles Rocha Alves"
    orcid: 0000-0002-3258-0025
    affiliation: 1
  - name: "Eduardo Aguiar"
    orcid: 0000-0001-7458-8976
    affiliation: 1
affiliations:
  - id: 1
    name: "Federal University of Juiz de Fora"
date: 2025-03-21
---

# Summary

This paper presents evolvingfuzzysystems, a Python library that provides implementations of several well-established evolving Fuzzy Systems (eFS's), including ePL-KRLS-DISCO [7], ePL+ [6], eMG [5], ePL [4], exTS [3], Simpl_eTS [2], and eTS [1]. The library facilitates model evaluation and comparison by offering built-in tools for training, visualization, and performance assessment. The code is available on the [github](https://github.com/kaikerochaalves/evolvingfuzzysystems.git)

# Statement of Need

The evolvingfuzzysystems is a Python package that contains seven well-known machine learning models. These models are widely used in the scientific literature on time series forecasting, but until now, they were not freely available. To address this issue, this paper presents [evolvingfuzzysystems](https://pypi.org/project/evolvingfuzzysystems/0.0.8/), a new Python library that includes the following eFS models: ePL-KRLS-DISCO, ePL+, eMG, ePL, exTS, Simpl\_eTS, and eTS. The package can be easily installed in Python using pip with the command: pip install evolvingfuzzysystems. It is expected that the library will attract new users and foster further applications of eFS models. This paper is intended as an introduction to the library, as well as an invitation to provide feedback and recommendations.

# Implementation and Features

To import the models simply type `import evolvingfuzzysystems.eFS as efs`. Then, each one of the six models can be used by typing: `efs.model` (replace model by `eTS`, `Simpl_eTS`, `exTS`, `ePL`, `eMG`, `ePL+`, or `ePL-KRLS-DISCO`). The hyperparameters of the models are detailed below:

- **eTS:**
  - **omega:** int, default=1000. Initialize the covariance matrix of the adaptive filter with the identity matrix (main diagonal) scaled by omega.
  - **r:** positive, default=0.1. Positive constant that defines the zone of influence of the rule.

- **Simpl_eTS:**
  - **omega:** int, default=1000. Initialize the covariance matrix of the adaptive filter with the identity matrix composed of omega in the main diagonal.
  - **r:** positive, default=0.1. Positive constant that defines the zone of influence of the rule.
  - **epsilon:** positive, default=0.01. Threshold to remove the rules.

- **exTS:**
  - **omega:** int, default=1000. Initialize the covariance matrix of the adaptive filter with the identity matrix composed of omega in the main diagonal.
  - **mu:** positive, default=1/3. Threshold to check if a rule is compatible with a rule.
  - **epsilon:** positive, default=0.01. Threshold to remove the rules.
  - **rho:** positive, default=1/2. Rate at which the radius is updated.

- **ePL:**
  - **alpha:** float in the range [0,1], default=0.001. How fast the rule center is updated.
  - **beta:** float in the range [0,1], default=0.5. The speed at which the arousal index is updated.
  - **lambda1:** float in the range [0,1], default=0.35. Threshold for the similarity index between two rules.
  - **tau:** float in the range [0,1] or None, default=None. Threshold for creating a new rule.
  - **s:** int, default=1000. Initialize the identity matrix of the covariance matrix for the adaptive filter.
  - **sigma:** positive float, default=0.25. Initialize the covariance matrix of the adaptive filter with the identity matrix composed of sigma in the main diagonal.

- **eMG:**
  - **alpha:** float in the range [0,1], default=0.001. How fast the rule center is updated.
  - **lambda1:** float in the range [0,1], default=0.1. The confidence level to compute thresholds for the compatibility measure and arousal index.
  - **w:** int, default=10. The windows size that defines the length of the anomaly pattern needed to classify data either as a new cluster or as a noise outlier.
  - **sigma:** positive float, default=0.05. Initialize the covariance matrix of the M-distance with the identity matrix composed of sigma in the main diagonal.
  - **omega:** int, default=100. Initialize the covariance matrix of the adaptive filter with the identity matrix composed of sigma in the main diagonal.

- **ePL+:**
  - **alpha:** float in the range [0,1], default=0.001. How fast the rule center is updated.
  - **beta:** float in the range [0,1], default=0.5. The speed at which the arousal index is updated.
  - **lambda1:** float in the range [0,1], default=0.35. Threshold for the similarity index between two rules.
  - **tau:** float in the range [0,1] or None, default=None. Threshold for creating a new rule.
  - **omega:** int, default=1000. Initialize the covariance matrix of the adaptive filter with the identity matrix composed of omega in the main diagonal.
  - **sigma:** positive float, default=0.25. Rate at which the radius is updated.
  - **e_utility:** positive float, default=0.05. Threshold to merge two rules.
  - **pi:** positive float, default=0.5. Learning rate for the radius of the rules (zone of influence).

- **ePL-KRLS-DISCO:**
  - **alpha:** float in the range [0,1], default=0.001. How fast the rule center is updated.
  - **beta:** float in the range [0,1], default=0.05. The speed at which the arousal index is updated.
  - **lambda1:** float in the range [0,1], default=0.0000001. Initialize the covariance matrix of the adaptive filter with the identity matrix composed of lambda1 in the main diagonal.
  - **tau:** float in the range [0,1] or None, default=0.05. Threshold for creating a new rule.
  - **sigma:** positive float, default=0.25. Rate at which the radius is updated.
  - **omega:** int, default=1. Initialize the last element of the main diagonal when a new vector is included into the dictionary and the covariance matrix needs to increase.
  - **e_utility:** positive float, default=0.05. Threshold to merge two rules.

After the models have been imported, the function `fit` can be invoked to train the models with the pair `X_train`, `y_train`. Then, the method `predict` can be called to estimate the output for any given `X_test`. Furthermore, to revise the model's structure with new pairs of `X` and `y` (`X_new`, `y_new`), just call the method `evolve`. 

- **Methods for train and test:**
  - `model.fit(X_train, y_train)`: Fit the model with the predictors and the predictions.
  - `model.predict(X_test)`: Make predictions for `X` values.
  - `model.evolve(X_new, y_new)`: When new input/output pairs are available, the model can be updated using the `evolve` method.

Finally, some other methods can be evoked to see characteristics of the formed rules, such as presented below:

- **Characteristics of the rules:**
  - `model.show_rules()`: Print the mean and standard deviation of all rules for each attribute.
  - `model.plot_rules()`: Plot the graphic of the rules.
  - `model.plot_gaussians()`: Plot the Gaussian fuzzy sets for each rule and each attribute.
  - `model.plot_rules_evolution()`: Plot the number of rules at each step of the training phase.
  - `model.plot_2d_projections()`: Plot the covariance between two attributes (only valid for eMG).

The methods that plot graphics have the following extra arguments:

- **Arguments for the methods that plot graphics:**
  - `num_points`: int, default=100. Number of points for the graphic.
  - `grid`: bool, default=False. Include a grid in the graphic.
  - `save`: bool, default=True. Save the graphic in the same folder as the code.
  - `format_save`: figure formats (i.e., 'jpg', 'png'), default='eps'. Format to save the graphic.
  - `dpi`: int, default=1200. Resolution of the graphic. Reduce for speed.

# Example Usage
Here is an example of using the library for time series forecasting:

```python
#-------------------
# Libraries
#-------------------

import math
import statistics as st
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from evolvingfuzzysystems.eFS import eTS

#-------------------
# Get data
#-------------------

# Load the California housing dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['Target'])

# Clean the data (remove NaNs if any)
X = X.dropna()
y = y.loc[X.index]  # Ensure target matches cleaned feature indices

# Convert to numpy
X, y = X.values, y.values

# Split the data into train, validation, and test
n = X.shape[0]
training_size = round(n*0.6)
validation_size = round(n*0.8)
X_train, X_val, X_test = X[:training_size,:], X[training_size:validation_size,:], X[validation_size:,:]
y_train, y_val, y_test = y[:training_size], y[training_size:validation_size:], y[validation_size:]

# Normalize features between 0 and 1
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define Grid Search parameters
parameters = {'omega': [50, 100, 250, 500, 1000, 10000], 'r': [0.1, 0.3, 0.5, 0.7, 5, 10, 50]}
grid = ParameterGrid(parameters)

lower_rmse = np.inf
for param in grid:
    
    print(".", end="")

    # Optimize parameters
    model = eTS(**param)
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculating the error metrics
    # Compute the Root Mean Square Error
    RMSE = math.sqrt(mean_squared_error(y_val, y_pred))
    
    if RMSE < lower_rmse:
        lower_rmse = RMSE
        best_eTS_params = param

# Optimized parameters
model = eTS(**best_eTS_params)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculating the error metrics
# Compute the Root Mean Square Error
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))
print("\nRMSE:", RMSE)
# Compute the Normalized Root Mean Square Error
NRMSE = RMSE/(y_test.max() - y_test.min())
print("NRMSE:", NRMSE)
# Compute the Non-Dimensional Error Index
NDEI = RMSE/st.stdev(np.asarray(y_test.flatten(), dtype=np.float64))
print("NDEI:", NDEI)
# Compute the Mean Absolute Error
MAE = mean_absolute_error(y_test, y_pred)
print("MAE:", MAE)
# Compute the Mean Absolute Percentage Error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", MAPE)
# Number of rules
print("Rules:", model.n_rules())
```

# Acknowledgements
The authors acknowledge the Federal University of Juiz de Fora for essential support during this work and the anonymous referees for their valuable comments. This work has been supported by the Brazilian agencies: (i) National Council for Scientific and Technological Development (CNPq), Grant no. 310625/2022-0 and 311260/2022-5; (ii) Coordination for the Improvement of Higher Education Personnel (CAPES), Grant no. 88881.690114/2022-01 and 88887.649818/2021-00; (iii) Foundation for Research of the State of Minas Gerais (FAPEMIG), Grant no. APQ-02922-18, (iv) Sao Paulo Research Foundation (FAPESP).

# References

1. **Angelov, Plamen P. and Filev, Dimitar P.** (2004). [An approach to online identification of Takagi-Sugeno fuzzy models](https://doi.org/10.1109/TSMCB.2003.817053). *IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics)*, 34(1), 484–498.

2. **Angelov, Plamen and Filev, Dimitar** (2005). [Simpl_eTS: A simplified method for learning evolving Takagi-Sugeno fuzzy models](https://doi.org/10.1109/FUZZY.2005.1452543). *The 14th IEEE International Conference on Fuzzy Systems, 2005. FUZZ'05.*, 1068–1073.

3. **Angelov, Plamen and Zhou, Xiaowei** (2006). [Evolving fuzzy systems from data streams in real-time](https://doi.org/10.1109/ISEFS.2006.251157). *2006 International Symposium on Evolving Fuzzy Systems*, 29–35.

4. **Lima, E., Hell, M., Ballini, R., and Gomide, F.** (2010). [Evolving fuzzy modeling using participatory learning](https://doi.org/10.1002/9780470569962.ch4). *Evolving Intelligent Systems: Methodology and Applications*, 67–86.

5. **Lemos, Andre, Caminhas, Walmir, and Gomide, Fernando** (2010). [Multivariable gaussian evolving fuzzy modeling system](https://doi.org/10.1109/TFUZZ.2010.2087381). *IEEE Transactions on Fuzzy Systems*, 19(1), 91–104.

6. **Maciel, Leandro, Gomide, Fernando, and Ballini, Rosangela** (2012). [An enhanced approach for evolving participatory learning fuzzy modeling](https://doi.org/10.1109/EAIS.2012.6232799). *2012 IEEE Conference on Evolving and Adaptive Intelligent Systems*, 23–28.

7. **Alves, Kaike Sa Teles Rocha and de Aguiar, Eduardo Pestana** (2021). [A novel rule-based evolving fuzzy system applied to the thermal modeling of power transformers](https://doi.org/10.1016/j.asoc.2021.107764). *Applied Soft Computing*, 112, 107764.
