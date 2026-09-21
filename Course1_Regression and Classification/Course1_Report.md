# Course 1 Report

Regression and Classification

My work in Course 1 covered linear regression, logistic regression, and regularization. I used Python notebooks to implement the main calculations with NumPy, plot results with Matplotlib, and compare selected models with scikit-learn. The exercises made the connection between the equations and the code clearer, while also showing where my implementations still need improvement.

# **Linear regression**

I started with the single-feature model f(x) \= wx \+ b, where w controls the slope and b is the intercept. I wrote functions for prediction, squared-error cost, gradients, and gradient descent. For m training examples, the cost was:

$J(w,\ b)\ =\ (1\ /\ 2m)\ Σᵢ\ (wxᵢ\ +\ b\ -\ yᵢ)²$

Gradient descent repeatedly subtracts the learning rate times the gradient from each parameter. I recorded the cost after each update to follow the training process and used the resulting weights to predict new values.

The saved Week 1 results showed a mean squared error of 1580.2475 for my implementation, compared with 34.9414 for scikit-learn. This was a useful reminder that a working training loop does not guarantee a good fit. Checking convergence and adjusting the training settings would be the next step. These are results from that notebook run, rather than evidence of performance on unseen data.

# **Multiple features and feature scaling**

In Week 2, I extended the model to predict house prices from size, bedrooms, floors, and age. The predictions were calculated as X @ w \+ b, using a feature matrix X and a weight vector w. This vectorized form applies the same calculation to all examples without a separate Python loop for each prediction.

I also practised z-score normalization, calculated as (x − μ) / σ for each feature. Putting features on similar scales helps gradient descent when inputs have very different ranges. Feature engineering and polynomial regression introduced ways to represent curved relationships by adding transformed inputs, such as squared terms, while keeping the model linear in its parameters.

# **Logistic regression**

Week 3 moved from predicting numerical values to binary classification. I used a linear score z \= X @ w \+ b and passed it through the sigmoid function to obtain a probability estimate:

$p\ =\ 1\ /\ (1\ +\ exp(-z))$

A threshold of 0.5 converted the probabilities into class labels. At that threshold, the decision boundary occurs where the linear score is zero. I used binary cross-entropy as the cost function, which gives a large penalty to confident incorrect predictions:

$J\ =\ -(1\ /\ m)\ Σᵢ\ [yᵢ\ ln(pᵢ)\ +\ (1\ -\ yᵢ)\ ln(1\ -\ pᵢ)]$

The gradient calculation used the prediction error p − y. In vector form, the weight gradient was X.T @ (p − y) / m, and the bias gradient was the mean of those errors. The code clipped probabilities between 10⁻¹⁵ and 1 − 10⁻¹⁵ before taking logarithms to avoid evaluating log(0).

# **Overfitting and regularization**

The polynomial examples explored how model complexity affects fitting. A model can follow the training points closely and still perform poorly on new data. I studied L2 regularization, which adds λΣⱼwⱼ² / (2m) to the cost and discourages large weights. In my regularized logistic regression code, this added λw / m to the weight gradient, while leaving the bias unpenalized. The notebooks also compared polynomial regression with Ridge regression. Increasing λ strengthens the penalty, although too much regularization can lead to underfitting.

# **Final practice and next steps**

The final notebook brings the classification steps together using 120 generated samples with two features, split evenly between two classes. The training function uses a learning rate of 0.1 and 5,000 iterations. It normalizes the features, records the cost, and includes an accuracy comparison with scikit-learn and a confusion matrix for the library model.

One limitation is that this notebook evaluates predictions on the same data used for training. That checks the fit, but does not establish how well the model generalizes. My next step is to add a separate test set and calculate normalization statistics from the training set only. I also want to revisit the earlier linear regression result and check why its error remained high. Writing the algorithms from scratch gave me a practical foundation for making those improvements.