# evolving Fuzzy Systems (eFS)

## Project description:

Package created by Kaike Sa Teles Rocha Alves

The evolvingfuzzysystems is a package that contains many evolving Fuzzy Systems (eFSs) in the context of machine learning models, including the ones developed by Kaike Alves during his Master degree under the supervision of professor Eduardo Pestana de Aguiar. 

    Author: Kaike Sa Teles Rocha Alves (PhD)
    Email: kaikerochaalves@outlook.com or kaike.alves@estudante.ufjf.br

Doi for ePL-KRLS-DISCO paper: https://doi.org/10.1016/j.asoc.2021.107764
Doi to cite the code: https://doi.org/10.5281/zenodo.15748291
Git hub repository: https://github.com/kaikerochaalves/evolvingfuzzysystems.git

Cite: SA TELES ROCHA ALVES, K. (2025). Evolvingfuzzysystems: a new Python library. Zenodo. https://doi.org/10.5281/zenodo.15748291

It provides the following models:

1. evolving Participaroty Learning with Kernel Recursive Least Square and Distance Correlation (ePL-KRLS-DISCO)
2. enhanced evolving Participatory Learning (ePL+)
3. evolving Multivariable Gaussian (eMG)
4. evolving Participatory Learning (ePL)
5. extended Takagi-Sugeno (eTS)
6. Simplified evolving Takagi-Sugeno (Simpl_eTS)
7. evolving Takagi-Sugeno (eTS)

*- Note: normalize the data in the range [0,1] for performance*

Summary:

The evolvingfuzzysystems library is a Python package that provides implementations of various Evolving Fuzzy Systems (eFS) models. These models are a class of machine learning algorithms capable of adaptively updating their structure in response to data dynamics while maintaining interpretability. The library aims to address the limited public availability of eFS model implementations, thereby fostering broader accessibility and adoption in research and practical applications.

Key features and capabilities of evolvingfuzzysystems include:

- Implemented eFS Models: The library offers several well-established eFS models, such as ePL-KRLS-DISCO, ePL+, eMG, ePL, exTS, SimpleTS, and eTS.

- Adaptive Learning: eFS models can update their internal structure without requiring retraining, which is a significant advantage over traditional machine learning models in dynamic environments. They can autonomously develop their structure, capture data stream dynamics, and produce accurate results even with nonlinear data.

- Interpretability: eFS models offer interpretability, combining accuracy, flexibility, and simplicity.

- Performance Evaluation Tools: The library includes built-in tools for training, visualization, and performance assessment, facilitating model evaluation and comparison.

- Computational Efficiency: Models within the library implement adaptive filters like Recursive Least Squares (RLS) and Weighted Recursive Least Squares (wRLS) for estimating consequent parameters.

- Visualization: The library provides functions to visualize fuzzy rules and their evolution during the training phase, enhancing the interpretability of the models. This includes: show_rules(), plot_rules(), plot_gaussians(), plot_rules_evolution(), and plot_2d_projections().

- Installation: The package can be easily installed using pip with the command: pip install evolvingfuzzysystems.

The library evaluates its models using the California housing dataset, measuring performance with metrics like normalized root-mean-square error (NRMSE), non-dimensional error index (NDEI), and mean absolute percentage error (MAPE). Computational complexity is also analyzed by measuring execution times and rule evolution during training and testing phases. Notably, the ePL model demonstrates a balance between accuracy and computational cost, making it suitable for real-world applications.

## Instructions

To install the library use the command: 

    pip install evolvingfuzzysystems

### ePL-KRLS-DISCO

To import the ePL-KRLS-DISCO, simply type the command:

    from evolvingfuzzysystems.eFS import ePL_KRLS_DISCO

Hyperparameters:

    alpha: float, possible values are in the interval [0,1], default=0.001
    This parameter controls the learning rate for updating the rule centers. A higher value means faster adaptation of rule centers to new data.

    beta: float, possible values are in the interval [0,1], default=0.05
    This parameter determines the adaptation rate of the arousal index, which influences the creation of new rules.

    sigma: float, must be a positive float, default=0.5
    This parameter defines the width of the Gaussian membership functions for the antecedent part of the rules. A smaller sigma leads to narrower, more specific fuzzy sets, while a larger sigma creates broader, more general fuzzy sets.

    lambda1: float, possible values are in the interval [0,1], default=0.0000001
    This acts as a regularization parameter in the KRLS algorithm. It helps prevent overfitting and improves the stability of the inverse matrix calculation.
    
    e_utility: float, possible values are in the interval [0,1], default=0.05
    This is the utility threshold for pruning rules. Rules whose utility falls below this value are considered for removal, aiming to maintain a parsimonious model.
    
    tau: float, possible values are in the interval [0,1], default=0.05
    This is the threshold for the arousal index. If the minimum arousal index among all existing rules exceeds tau, a new rule is considered for creation.

    omega: int, must be a positive integer, default=1
    This parameter is used in the initialization of the Q matrix within the KRLS algorithm. It can be seen as an initial regularization term for the covariance matrix estimate.

Examples:

    from evolvingfuzzysystems.eFS import ePL_KRLS_DISCO
    model = ePL_KRLS_DISCO()
    model.fit(X_train, y_train)
    model.evolve(X_val, y_val)
    y_pred = model.predict(y_test)

Notes:

1. The hyperparameters alpha, beta, and sigma are the most relevant for performance
2. The hyperparameters are in the range [0,1] considering the data are normalized in the range [0.1]

Example:

    from sklearn.preprocessing import MinMaxScaler
    from evolvingfuzzysystems.eFS import ePL_KRLS_DISCO
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    model = ePL_KRLS_DISCO(alpha = 0.01, beta=0.06, tau=0.04, sigma=10)
    model.fit(X_train, y_train)
    model.evolve(X_val, y_val)
    y_pred = model.predict(y_test)

### ePL+

To import the ePL+, simply type:

    from evolvingfuzzysystems.eFS import ePL_plus

Hyperparameters:

    alpha: float, possible values are in the interval [0,1], default = 0.001
    This parameter controls the learning rate for updating the rule centers. 
    A smaller alpha means slower adaptation of rule centers to new data, 
    while a larger alpha results in faster adaptation.

    beta: float, possible values are in the interval [0,1], default = 0.1
    This parameter controls the learning rate for updating the arousal index. 
    The arousal index helps determine when a new rule should be created. 
    A higher beta makes the system more responsive to new patterns, 
    potentially leading to more rules.

    tau: float, possible values are in the interval [0,1] or None, 
    default = None (defaults to beta)
    This parameter serves as a threshold for the arousal index to 
    determine whether a new rule needs to be created. If the minimum 
    arousal index among existing rules exceeds tau, a new rule is considered.
    If tau is None, it automatically takes the value of beta.

    e_utility: float, possible values are in the interval [0,1], default = 0.05
    This parameter is a threshold for the utility measure of a rule. R
    ules whose utility (which relates to their contribution over time) 
    falls below this threshold are considered for removal, helping to prune
    redundant or inactive rules.

    pi: float, possible values are in the interval [0,1], default = 0.5
    This parameter is a forgetting factor for updating the rule's radius (sigma). 
    It controls how much influence new observations have on adapting the spread 
    of a rule, balancing between current data and historical information.

    sigma: float, possible values are in the interval [0,1], default = 0.25
    This parameter represents the initial radius or spread for the 
    Gaussian membership functions of new rules. It influences how broadly 
    a new rule covers the input space.

    lambda1: float, possible values are in the interval [0,1], default = 0.35
    This parameter is a threshold for the similarity index. If the 
    compatibility between two rules (or a new data point and an existing rule) 
    is greater than or equal to lambda1, it can trigger rule merging or 
    influence how existing rules are updated.

    omega: int, must be a positive integer, default = 1000
    This parameter is used to initialize the P matrix (covariance matrix inverse) 
    in the weighted Recursive Least Squares (wRLS) algorithm. A larger 
    omega generally indicates less confidence in initial parameters,
    allowing for faster early adaptation.

Notes:

1. The hyperparameters alpha, beta, tau, e_utility, and pi are the most relevant for performance. If tau is None it receives beta value
2. The hyperparameters are in the range [0,1] considering the data are normalized in the range [0.1]

Example:

    from sklearn.preprocessing import MinMaxScaler
    from evolvingfuzzysystems.eFS import ePL_plus
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    model = ePL_plus(alpha = 0.1, beta=0.2, sigma=0.3)
    model.fit(X_train, y_train)
    model.evolve(X_val, y_val)
    y_pred = model.predict(y_test)

### eMG

To import the eMG, type:

    from evolvingfuzzysystems.eFS import eMG

Hyperparameters:

    alpha: float, possible values are in the interval [0,1], default = 0.01
    This parameter controls the learning rate for updating the rule centers
    and covariance matrices. A smaller alpha means slower adaptation, 
    while a larger alpha leads to faster changes in rule parameters.

    w: int, must be an integer greater than 0, default = 10
    This parameter defines the window size for computing the arousal index.
    The arousal index, which influences rule creation, is based on the recent 
    history of data points falling within or outside the confidence region 
    of a rule, over this w number of samples.

    sigma: float, must be a positive float, default = 0.05
    This parameter represents the initial diagonal value for the covariance 
    matrix (Sigma) of newly created rules. It essentially defines the initial 
    spread of a new Gaussian rule in each dimension.

    lambda1: float, possible values are in the interval [0,1], default = 0.1
    This parameter defines a significance level for the chi-squared test used 
    to determine the thresholds for rule creation and merging (Tp). 
    It influences how "novel" a data point needs to be to potentially 
    trigger a new rule or how "similar" two rules need to be to be merged.

    omega: int, must be a positive integer, default = 100
    This parameter is used to initialize the Q matrix 
    (inverse of the covariance matrix) in the Recursive Least Squares (RLS) 
    algorithm, which estimates the consequent parameters of each rule. 
    A larger omega implies a larger initial uncertainty in the consequent 
    parameters, allowing for faster early adjustments.

1. The hyperparameters alpha and w are the most relevant for performance. If tau is None it receives beta value
2. The hyperparameters are in the range [0,1] considering the data are normalized in the range [0.1]

Example:

    from sklearn.preprocessing import MinMaxScaler
    from evolvingfuzzysystems.eFS import eMG
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    model = eMG(alpha = 0.1, w=25)
    model.fit(X_train, y_train)
    model.evolve(X_val, y_val)
    y_pred = model.predict(y_test)

### ePL

To import the ePL, type:

    from evolvingfuzzysystems.eFS import ePL

Hyperparameters:

    alpha: float, possible values are in the interval [0,1], default = 0.001
    This parameter controls the learning rate for updating the rule centers. 
    A smaller alpha means slower adaptation, while a larger alpha leads to 
    faster changes in rule centers.

    beta: float, possible values are in the interval [0,1], default = 0.5
    This parameter controls the learning rate for updating the arousal index. 
    The arousal index influences the creation of new rules; a higher beta 
    makes the system more sensitive to new patterns and potentially creates 
    new rules more readily.

    tau: float, possible values are in the interval [0,1] or None, default = None (defaults to beta)
    This parameter serves as a threshold for the arousal index. 
    If the minimum arousal index among existing rules exceeds tau, 
    a new rule is considered for creation. If tau is None, it automatically 
    takes the value of beta.

    lambda1: float, possible values are in the interval [0,1], default = 0.35
    This parameter is a threshold for the similarity index. If the compatibility 
    between two rules (or a new data point and an existing rule) is greater 
    than or equal to lambda1, it can trigger rule merging.

    sigma: float, must be a positive float, default = 0.25
    This is the fixed bandwidth parameter for the Gaussian membership functions. 
    It determines the spread of the Gaussian functions; a smaller sigma leads 
    to narrower, more localized rules, while a larger sigma creates broader, 
    more generalized rules.

    omega: int, must be a positive integer, default = 1000
    This parameter is used to initialize the P matrix (covariance matrix inverse) 
    in the weighted Recursive Least Squares (wRLS) algorithm. A larger s generally 
    indicates less confidence in initial parameters, allowing for faster early 
    adaptation.

1. The hyperparameters alpha, beta and tau are the most relevant for performance. If tau is None it receives beta value
2. The hyperparameters are in the range [0,1] considering the data are normalized in the range [0.1]

Example:

    from sklearn.preprocessing import MinMaxScaler
    from evolvingfuzzysystems.eFS import ePL
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    model = ePL(alpha = 0.1, beta = 0.2, lambda1 = 0.5, sigma = 0.1)
    model.fit(X_train, y_train)
    model.evolve(X_val, y_val)
    y_pred = model.predict(y_test)

### exTS

To import the exTS, type:

    from evolvingfuzzysystems.eFS import exTS

Hyperparameters:

    rho: float, possible values are in the interval [0,1], default = 1/2 (0.5)
    This parameter is a forgetting factor for updating the rule's radius (sigma). 
    It controls how much influence new observations have on adapting the spread 
    of a rule, balancing between current data and historical information.

    mu: float or int, must be greater than 0, default = 1/3
    This parameter acts as a threshold for the membership degree (mu) 
    of a data point to a rule. If all membership degrees of existing rules 
    are below mu, it indicates a novel data region, potentially leading to 
    the creation of a new rule.

    epsilon: float, possible values are in the interval [0,1], default = 0.01
    This parameter is a threshold for rule removal based on their relative number 
    of observations. Rules whose proportion of total observations falls 
    below epsilon are considered for pruning, aiming to remove underutilized rules.

    omega: int, must be a positive integer, default = 1000
    This parameter is used to initialize the C matrix 
    (covariance matrix inverse) in the weighted Recursive Least Squares (wRLS) 
    algorithm, which estimates the consequent parameters of each rule. 
    A larger omega implies a larger initial uncertainty in the consequent parameters, 
    allowing for faster early adjustments.

1. The hyperparameters rho, mu and epsilon are the most relevant for performance. 
2. The hyperparameters are in the range [0,1] considering the data are normalized in the range [0.1]

Example:

    from sklearn.preprocessing import MinMaxScaler
    from evolvingfuzzysystems.eFS import exTS
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    model = exTS()
    model.fit(X_train, y_train)
    model.evolve(X_val, y_val)
    y_pred = model.predict(y_test)

### Simpl_eTS

To import the Simpl_eTS, type:

    from evolvingfuzzysystems.eFS import Simpl_eTS

Hyperparameters:

    r: float or int, must be greater than 0, default = 0.1
    This parameter defines the radius for the Cauchy membership functions. 
    It controls the spread of the membership functions; a smaller r leads 
    to more localized rules, while a larger r creates broader rules. 
    It is also used as a threshold for determining if a data point is 
    close enough to an existing rule to update it.

    epsilon: float, possible values are in the interval [0,1], default = 0.01
    This parameter is a threshold for rule removal based on their relative number 
    of observations. Rules whose proportion of total observations falls below 
    epsilon are considered for pruning, aiming to remove underutilized rules.

    omega: int, must be a positive integer, default = 1000
    This parameter is used to initialize the C matrix (covariance matrix inverse) 
    in the weighted Recursive Least Squares (wRLS) algorithm, which estimates 
    the consequent parameters of each rule. A larger omega implies a larger 
    initial uncertainty in the consequent parameters, allowing for faster early 
    adjustments.

1. The hyperparameters r and epsilon are the most relevant for performance. 
2. The hyperparameters are in the range [0,1] considering the data are normalized in the range [0.1]

Example:

    from sklearn.preprocessing import MinMaxScaler
    from evolvingfuzzysystems.eFS import Simpl_eTS
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    model = Simpl_eTS(r = 0.2, epsilon=0.01)
    model.fit(X_train, y_train)
    model.evolve(X_val, y_val)
    y_pred = model.predict(y_test)

### eTS

To import the eTS, type:

    from evolvingfuzzysystems.eFS import eTS

Hyperparameters:

    r: float or int, must be greater than 0, default = 0.1
    This parameter defines the radius for the Gaussian membership functions. 
    It controls the spread of the membership functions; a smaller r leads 
    to more localized rules, while a larger r creates broader rules. 
    It is also used as a threshold in the rule creation logic.

    omega: int, must be a positive integer, default = 1000
    This parameter is used to initialize the C matrix 
    (covariance matrix inverse) in the weighted Recursive Least Squares 
    (wRLS) algorithm, which estimates the consequent parameters of each rule. 
    A larger omega implies a larger initial uncertainty in the consequent parameters,
    allowing for faster early adjustments.

Example:

    from sklearn.preprocessing import MinMaxScaler
    from evolvingfuzzysystems.eFS import eTS
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_test = scaler.transform(X_test)
    model = eTS(r = 0.1)
    model.fit(X_train, y_train)
    model.evolve(X_val, y_val)
    y_pred = model.predict(y_test)


## Extra details

If you want to see how many rules was generated, you can type:

    model.n_rules()

You can see the rules graphically by typing:

    model.plot_rules()

If you want to see all Gaussian fuzzy sets, type:

    model.plot_gaussians()

To see the evolution of the rules along with the training, type:

    model.plot_rules_evolution()

For the eMG model, as it uses covariance matrix to model the distribution of the input vector, if you want to visualize the covariance between two attributes, type:

    model.plot_2d_projections()

These last four function that plots graphics accepts extra arguments:

    grid (boolean): if you want the graphic with grid
    save (boolean): if you want to save the graphic
    format_save (default='eps'): the format you want to save the graphic.
    dpi (integer, default=1200): the resolution to save the graphic

You can learn more about the ePL-KRLS-DISCO and eFSs in the paper: https://doi.org/10.1016/j.asoc.2021.107764.

Code of Conduct

evolvingfuzzysystems is a library developed by Kaike Alves. Please read the Code of Conduct for guidance.

Call for Contributions

The project welcomes your expertise and enthusiasm!

Small improvements or fixes are always appreciated. If you are considering larger contributions to the source code, please contact by email first.

If you think you can contribute to this project regarding the code, speed, etc., please, feel free to contact me and to do so.