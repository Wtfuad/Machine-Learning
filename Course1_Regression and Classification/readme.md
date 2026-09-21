# Course 1: Regression and Classification

This folder contains my notes and practice code for regression and classification. The notebooks start with a simple linear model and build up to multiple features, logistic regression, and regularization.

The focus is on working through the calculations with NumPy, plotting the results, and comparing some of the implementations with scikit-learn. Most topics have their own notebook so they are easy to revisit.

## What's in each week

### Week 1 — Linear regression

Starts with the model `f(x) = wx + b` and looks at how the weight and bias affect predictions. The second notebook puts the pieces together: creating a dataset, calculating cost and gradients, training with gradient descent, and making predictions. It also compares the result with scikit-learn's `LinearRegression`.

### Week 2 — Working with multiple features

Extends linear regression to more than one input feature, using house prices as an example. The notebooks cover:

- Multiple linear regression and vectorized predictions
- Feature scaling with z-score normalization
- Gradient descent with multiple features
- Feature engineering
- Polynomial regression
- Linear regression with scikit-learn

### Week 3 — Classification and regularization

Introduces logistic regression, starting with the sigmoid function and moving through logistic loss, gradient descent, and decision boundaries. The later notebooks explore overfitting and L2 regularization, including Ridge regression and a regularized logistic regression implementation.

The final notebook brings the classification workflow together: creating data, visualizing it, normalizing features, training a model from scratch, checking accuracy and the confusion matrix, and comparing with scikit-learn.

## Folder layout

```text
.
├── week1/    # Model representation and single-feature linear regression
├── week2/    # Multiple features, scaling, and polynomial regression
├── week3/    # Logistic regression, overfitting, and regularization
├── src/      # Reusable NumPy implementations
└── readme.md
```

The `src` folder contains three small modules:

- `linear_regression.py`: predictions, squared-error cost, gradients, and gradient descent for one feature.
- `multiple_linear_regression.py`: the same core operations for multiple features, plus z-score normalization.
- `logistic_regression.py`: sigmoid, probability and class predictions, logistic loss, and gradient descent, with regularized versions as well.

## Running the notebooks

You'll need Python, Jupyter Notebook, NumPy, Matplotlib, and scikit-learn. Install them with:

```bash
python -m pip install notebook numpy matplotlib scikit-learn
```

From this folder, start Jupyter:

```bash
python -m notebook
```

Open a notebook and run the cells from top to bottom. Start with `week1/1_model_representation.ipynb` and follow the numbered files through each week.

These are small learning examples. Changing the learning rate, number of iterations, polynomial degree, or regularization strength is a useful way to see how the models behave. The cost curves and prediction plots help make those changes easier to understand.
