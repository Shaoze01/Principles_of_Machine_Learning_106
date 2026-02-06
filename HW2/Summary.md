## Lab Report: Linear Regression for Wine Quality Prediction

### 1. Paper Design & Approach

In this experiment, we implemented a Linear Regression model from scratch to predict the citric acid content of Vinho Verde wine samples based on other physicochemical features. The approach consisted of four main stages:

- Implementation of Least Squares:We utilized the closed-form solution for linear regression. While the normal equation $w = (X^T X)^{-1} X^T y$ is theoretically sound, we opted for np.linalg.lstsq (based on SVD) for better numerical stability when computing the weight vector $w$.

- Baseline Model (2 Features):We started with a baseline model using only alcohol and density as predictors, as specified in the requirements.

- Feature Selection (Greedy Approach):Finding the 3rd Feature: We iterated through all remaining features, adding them one by one to the baseline set. We selected the feature that resulted in the lowest L2 Norm error on the training set. Finding the 4th Feature (Exhaustive Pair Search): Instead of simply adding one feature to the previous best set, we performed a more rigorous search. We iterated through all possible pairs of remaining features to add to the baseline (alcohol, density), identifying the optimal 4-feature combination.

- Full Model:Finally, we trained a model using all available predictors to establish a performance ceiling (lower bound on training error).

### 2 and 3. Model Description and Parameter Selection

We used Ordinary Least Squares (OLS) for all regression tasks. The objective was to minimize the sum of squared residuals:$$J(w) = ||Xw - y||_2^2$$

#### Results Table
| Model | Features | L2 Norm Error |
|-------|----------|-------|
| Model 1 | alcohol, density | 1.7707 |
| Model 2 | alcohol, density, volatile_acidity | 1.3634 |
| Model 3 | alcohol, density, ixed_acidity, volatile_acidity | 1.2416 |
| Full Model | all features | 1.0624 |



### 4. Discussion

(1) Numerical Stability: I learned that even if a matrix is mathematically Full Rank (Rank 10), it can be "ill-conditioned". This leads to overflow or divide by zero errors when using np.linalg.inv(), necessitating the use of more robust solvers like np.linalg.lstsq().


(2) Data Handling: The transition from Pandas DataFrame  to NumPy ndarrays  is a critical workflow step for implementing efficient optimization models in Python.



## Lab Report: k-Nearest Neighbor Classification

### 1. Design & Approach

In this experiment, we implemented a k-Nearest Neighbor (k-NN) classifier from scratch and evaluated it on the Lenses (multiclass) and Credit Approval (binary) datasets. The implementation followed a modular design:

- Imputation: We handled missing values (?) by filling continuous features with the mean and categorical features with the mode.

- Encoding: Categorical variables were transformed using One-Hot Encoding.

Normalization: We applied Z-score standardization ($z = \frac{x - \mu}{\sigma}$) to all numerical features to strictly adhere to the Euclidean distance assumptions.

- Metric Alignment: To satisfy the requirement "Distance = 1 if values disagree" for categorical attributes, we scaled the One-Hot encoded vectors by a factor of $\frac{1}{\sqrt{2}}$. This ensures that the Euclidean distance between two different categories ($\sqrt{1^2 + 1^2} = \sqrt{2}$) is normalized to $1$.

#### k-NN Implementation:

- We utilized a vectorized approach using np.linalg.norm with broadcasting to compute the distance matrix between test and training samples efficiently.

- Predictions were made using majority voting (np.bincount and np.argmax).


### 2 and 3. Model Description and Parameter Selection

We evaluated the k-NN model using the L2 (Euclidean) Distance metric.

Hyperparameter Selection:We tested four different values for the number of neighbors: $k \in \{1, 3, 5, 7\}$.

- Odd values were chosen to avoid tie-breaking situations in binary classification.

- Range: This range covers the spectrum from a highly complex model ($k=1$, low bias/high variance) to a smoother model ($k=7$, higher bias/lower variance).

#### Data Splitting:

- Lenses: Used the provided fixed training/testing files.

- Credit Approval: Preprocessed the training and testing sets separately but fitted the standardization parameters (mean/std) on the training set to avoid data leakage.


#### Results Table

| Dataset | k=1 | k=3 | k=5 | k=7 |
|---------|-----|-----|-----|-----|
| Lenses | 1 | 1 | 0.5 | 0.8333 |
| Credit Approval | 0.9529 | 0.8768 | 0.8587 | 0.8551 |

### 4. What did you learn from this exercise?

Metric Sensitivity: I learned that KNN is not a "plug-and-play" algorithm; it is highly sensitive to how distance is defined. Mixed-type data (continuous and categorical) requires careful weighting (like the $\sqrt{2}$ scaling we discussed) to ensure consistency(Distance = 1 if they are not accurate).




## Lab Report: Naive Bayes for Spam Detection

### 1. Design & Approach

In this experiment, we implemented a Gaussian Naive Bayes classifier from scratch to detect spam emails using the UCI Spambase dataset. The approach consisted of three main stages:

#### Model Implementation: 

We built a class-based structure GaussianNaiveBayes adhering to the fit and predict paradigm.

- - Training (fit): We calculated the prior probabilities $P(C)$ and the feature statistics (Mean $\mu_{c,i}$ and Variance $\sigma^2_{c,i}$) for both "Spam" and "Non-Spam" classes using maximum likelihood estimation. An epsilon ($1e-9$) was added to the variance to prevent division-by-zero errors.

- - Prediction (predict): We implemented the decision rule using Log-Probabilities to ensure numerical stability. instead of calculating the product $P(C) \prod P(x_i|C)$, we computed the sum $\log P(C) + \sum \log P(x_i|C)$, transforming the problem into a summation of log-likelihoods derived from the Gaussian Probability Density Function (PDF).

#### Evaluation Pipeline:

The data was split into training (80%) and testing (20%) sets.

We implemented a custom compute_metrics function to calculate Accuracy, Precision, Recall, and F1-Score from scratch by counting TP, TN, FP, and FN.

- Feature Analysis:

We quantified the "discriminative power" of each feature by calculating the difference in means between the two classes, normalized by the Pooled Standard Deviation. This allowed us to rank features by how well they separate the classes.



### 2. Model Description and Parameter Selection

- Model Type: Gaussian Naive Bayes.

- Assumption: We assumed that the continuous features (word frequencies, run lengths) follow a Normal (Gaussian) distribution within each class.

- Hyperparameters: epsilon = 1e-9: A smoothing parameter added to the variance to ensure numerical stability.

- Metric Selection: We focused on Precision and Recall alongside Accuracy, as false positives (flagging a normal email as spam) are particularly costly in spam detection systems.

### 3. Results Table
| Metric | Value |
|--------|-------|
| Accuracy | 0.8217 |
| Precision | 0.7233 |
| Recall | 0.9385 |
| F1-Score | 0.8170 |
 

#### Top 5 Discriminative Features
1. word_freq_your
2. word_freq_remove
3. word_freq_000
4. char_freq_$
5. word_freq_you


#### 4. Reflection

Implementing Naive Bayes highlighted the trade-off between theoretical simplicity and practical engineering.

Numerical Engineering: The shift from raw probabilities to log-probabilities was not just a mathematical trick but a necessity. Without it, the likelihoods for 57 features would vanish to zero (underflow), rendering the classifier useless.

Independence Assumption: Despite the clearly false assumption that words appear independently (e.g., "credit" and "card" are correlated), the model performed surprisingly well. This reinforces the concept that for classification, we only need to estimate the decision boundary correctly, not the true underlying probability density.


