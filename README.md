# Comparative Analysis of Methods for Option Pricing

## UCL Year 2 Summer Research Programme

In this repository I investigate **European option pricing using analytical, numerical and machine-learning methods**. My focus is on the mathematical structure of the pricing problem and on whether machine-learning models can learn the relationship between market parameters and option prices.

The main pricing problem I consider is

\[
(S,K,T,r,\sigma) \longrightarrow V,
\]

where \(S\) is the underlying price, \(K\) the strike, \(T\) the time to maturity, \(r\) the risk-free rate, \(\sigma\) the volatility and \(V\) the option value.

I use this mapping as the basis for comparing **Black–Scholes, Monte Carlo, Random Forest, XGBoost, CatBoost and LSTM** models.

---

## What I am trying to answer

Black–Scholes gives a closed-form solution for European vanilla options, so a natural question is:

> **Why use machine learning when the option price can already be calculated analytically?**

The reason is that the Black–Scholes framework is built on assumptions such as constant volatility, continuous trading and an idealised market. Real option prices can reflect changing volatility, liquidity, market expectations and other effects that are not explicitly represented by the basic model.

My aim in this repository is therefore not simply to replace Black–Scholes. I want to investigate whether a model trained on market data can learn a useful approximation to option prices when the relationship between the inputs and the observed price is more complicated than the Black–Scholes assumptions imply.

---

# Mathematical foundation

## Geometric Brownian Motion

I begin with the standard model for the underlying asset:

\[
dS_t=\mu S_t\,dt+\sigma S_t\,dW_t.
\]

Under the risk-neutral measure \(\mathbb Q\), the drift becomes the risk-free rate:

\[
dS_t=rS_t\,dt+\sigma S_t\,dW_t^{\mathbb Q}.
\]

The terminal price can therefore be written as

\[
S_T=S_0\exp\left[
\left(r-\frac{1}{2}\sigma^2\right)T+\sigma\sqrt{T}Z
\right],
\qquad Z\sim N(0,1).
\]

This representation is particularly useful because it connects the analytical Black–Scholes model directly to the Monte Carlo simulation I use in this repository.

---

## Risk-neutral valuation

For a payoff \(\Phi(S_T)\), the risk-neutral price is

\[
V_0=e^{-rT}\mathbb E^{\mathbb Q}[\Phi(S_T)].
\]

The three main approaches in my analysis can be viewed as different ways of evaluating this relationship:

- **Black–Scholes** evaluates the pricing problem analytically.
- **Monte Carlo** estimates the expectation numerically.
- **Machine learning** learns an approximation to the resulting pricing function from data.

This gives a common mathematical framework for comparing the methods rather than treating them as unrelated algorithms.

---

# Black–Scholes

For a European call option,

\[
C=S\Phi(d_1)-Ke^{-rT}\Phi(d_2),
\]

where

\[
d_1=
\frac{\ln(S/K)+(r+\frac{1}{2}\sigma^2)T}
{\sigma\sqrt{T}}
\]

and

\[
d_2=d_1-\sigma\sqrt{T}.
\]

For a European put,

\[
P=Ke^{-rT}\Phi(-d_2)-S\Phi(-d_1).
\]

I use Black–Scholes as the main analytical benchmark throughout the project.

Implementation:

[`models/BlackScholes.py`](models/BlackScholes.py)

---

# Monte Carlo

I also implement Monte Carlo pricing by simulating the terminal stock price under the risk-neutral dynamics.

For simulated paths \(S_T^{(i)}\), the price estimator is

\[
\widehat V_N=
\frac{e^{-rT}}{N}
\sum_{i=1}^{N}\Phi(S_T^{(i)}).
\]

As \(N\) increases, the estimator converges towards the risk-neutral expectation under the usual Monte Carlo assumptions.

This gives me a useful numerical benchmark against the closed-form Black–Scholes value.

Implementation:

[`models/MonteCarlo.py`](models/MonteCarlo.py)

---

# Machine Learning

## Learning the option-pricing function

I treat option pricing as a supervised regression problem. The input vector is

\[
X=(S,K,T,r,\sigma),
\]

and the target is the option price

\[
y=V.
\]

The objective is to learn a function

\[
f_\theta(X)\approx V,
\]

where \(f_\theta\) is the machine-learning model.

This is a nonlinear function-approximation problem. The models do not need to be given the Black–Scholes formula explicitly; instead, they infer relationships from the training observations.

A neural network is one possible function-approximation approach, while tree-based methods provide a different way of modelling nonlinear interactions between the input variables.

---

## Data

I use two different sources of target prices so that I can separate **learning a known theoretical function** from **learning observed market prices**.

### Synthetic training data — `synthetic_training_data.csv`

The synthetic dataset combines:

- historical stock prices;
- historical realised volatility; and
- Black–Scholes prices.

The target therefore has the form

\[
y=f_{BS}(S,K,T,r,\sigma).
\]

This creates a controlled experiment: if a model performs well, it demonstrates that the model can approximate the Black–Scholes pricing surface.

However, there is an important limitation. If the labels are generated by Black–Scholes, the model is ultimately learning a sophisticated approximation to a **Black–Scholes calculator**. It cannot learn market effects that were never present in the labels.

### Real training data — `real_training_data.csv`

I also train models using **actual market option prices**. This changes the learning problem because the target is now an observed market price rather than a value generated directly from the Black–Scholes equation.

### Real test data — `real_test_data.csv`

I hold out approximately **20% of the real observations** for testing. The final evaluation therefore measures performance on data that were not used to fit the models.

---

## Market data

For the real-data experiments I use U.S. equity option-chain data obtained through `yfinance`.

The main equities considered include:

- Apple — **AAPL**
- Microsoft — **MSFT**
- NVIDIA — **NVDA**
- Tesla — **TSLA**
- Amazon — **AMZN**

For consistency with the analytical framework, I treat the observed contracts as **European options** when comparing them with Black–Scholes and Monte Carlo.

The data are stratified across **ticker and maturity**, reducing the risk that the training set is dominated by particular assets or expiry ranges.

---

# Models I implement

## Random Forest

Random Forest models the pricing function using an ensemble of decision trees.

A tree repeatedly partitions the feature space using threshold decisions. For example:

```text
                  Is S > threshold?
                     /        \
                   Yes         No
                   /             \
          Is T > threshold?     ...
             /       \
           Yes        No
            ↓          ↓
        Prediction  Prediction
```

A Random Forest combines the predictions from many trees. This gives a flexible nonlinear model without requiring me to specify a functional form for the option-pricing surface.

I use Random Forest as one of the main tree-based benchmarks.

---

## XGBoost

XGBoost uses gradient-boosted decision trees. Instead of treating each tree as an independent estimator, the trees are built sequentially so that later trees improve the current prediction.

This is useful for the pricing problem because the relationship between \(S\), \(K\), \(T\), \(r\), \(\sigma\) and \(V\) is strongly nonlinear.

I compare XGBoost directly with Random Forest and CatBoost, both in predictive accuracy and computational cost.

---

## CatBoost

CatBoost is another gradient-boosting approach based on decision trees. It is designed to provide strong predictive performance while handling categorical variables effectively and reducing some preprocessing requirements.

I use CatBoost as a third tree-based approach so that I can compare different ensemble strategies on the same option-pricing problem.

---

## LSTM

I also investigate a Long Short-Term Memory network.

An LSTM is designed to retain information through a sequence using three main gates:

- **Forget gate** — determines what information to remove from the cell state.
- **Input gate** — determines what new information to store.
- **Output gate** — determines what information is used for the prediction.

The motivation for using an LSTM is that financial variables have temporal structure. However, there is an important issue with the available option-chain data: observations from Yahoo Finance are not naturally arranged as a clean, evenly spaced chronological time series.

This makes the LSTM a poor architectural fit for the data I have available. This is also important when interpreting the results: a poor LSTM result does not imply that neural networks are inherently unsuitable for option pricing. It indicates that the particular sequential architecture is not well matched to this dataset.

In my experiments, this mismatch contributes to the LSTM performing substantially worse than the tree-based models.

---

# Synthetic vs. real pricing

This is one of the main mathematical ideas behind the repository.

### Synthetic data

With synthetic Black–Scholes labels,

\[
y=f_{BS}(X).
\]

The ML model is approximating a known mathematical function. This is useful because I know what the model is trying to learn.

### Real data

With observed market prices,

\[
y=V_{market},
\]

and the relationship is no longer simply the Black–Scholes function. The data can contain noise and market effects that are not represented in the basic theoretical model.

This leads to two different questions:

| Dataset | Question |
|---|---|
| Synthetic | Can the model learn the Black–Scholes pricing function? |
| Real market | Can the model learn a useful approximation to observed option prices? |

This distinction is important when interpreting model performance.

---

# Evaluation

I compare the models using **MAE, RMSE and \(R^2\)** on unseen observations.

## Mean Absolute Error

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\widehat y_i|.
\]

MAE measures the average absolute pricing error.

## Root Mean Squared Error

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\widehat y_i)^2}.
\]

RMSE gives greater weight to large errors.

## \(R^2\)

\[
R^2=1-
\frac{\sum_i(y_i-\widehat y_i)^2}
{\sum_i(y_i-\bar y)^2}.
\]

This measures how much of the variation in the target is explained relative to a constant-mean benchmark.

Implementation:

[`evaluation/evaluation.py`](evaluation/evaluation.py)

---

# Results

## Synthetic data

When I train on Black–Scholes-generated labels, the models are effectively learning to reproduce the Black–Scholes relationship. This can lead to very low pricing errors, but it should not be interpreted as evidence that the model has learned additional market information.

The synthetic experiment is therefore mainly a test of **function approximation**.

## Real market data

When I train on actual option prices, the models generally produce lower errors and higher \(R^2\) on the real test set than the corresponding models trained on synthetic data.

This is consistent with the idea that the real-data models can learn relationships contained in observed market prices that are not encoded in the Black–Scholes-generated labels.

The result needs to be interpreted carefully because the two experiments have different target-generating processes.

---

## LSTM

The LSTM performs significantly worse than the tree-based models in my experiments.

The most important explanation is the structure of the data. LSTMs are designed to exploit sequential information, while the option-chain observations I use are not a clean evenly spaced time series.

This makes the LSTM result an example of why **model architecture and data structure need to be considered together**.

---

## Random Forest, XGBoost and CatBoost

The three tree-based models perform relatively similarly, and all substantially outperform the LSTM in the experiments.

I therefore also consider computational cost rather than choosing a model purely from its error metric. In my experiments the qualitative ordering was approximately:

```text
XGBoost  →  Random Forest  →  CatBoost  →  LSTM
least computational cost                         most computational cost
```

The exact cost depends on hyperparameters, hardware and implementation, so this ordering is specific to the experiments in this repository rather than a universal ranking of the algorithms.

---

# What I learned from the comparison

The most useful conclusion for me is that the different methods are solving closely related mathematical problems in different ways.

### Black–Scholes

Uses a strong mathematical model of the underlying and produces a closed-form price under its assumptions.

### Monte Carlo

Uses the same risk-neutral framework but estimates the expectation numerically rather than evaluating it analytically.

### Machine learning

Does not impose the Black–Scholes functional form. Instead, it estimates the mapping from observed inputs to prices from data.

This makes the comparison more interesting than simply asking which model has the lowest error. The assumptions used to generate the target data are just as important as the algorithm used to fit the data.

---

# Repository structure

```text
UCL-Summer-Research-Project-Year-2/
│
├── data/
│   ├── generate_datasets.py
│   ├── attempt2.py
│   ├── synthetic_training_data.csv
│   ├── real_training_data.csv
│   └── real_test_data.csv
│
├── models/
│   ├── BlackScholes.py
│   ├── MonteCarlo.py
│   ├── Random Forest/
│   │   └── RandomForest.py
│   ├── XGBoost/
│   │   └── XGBoost.py
│   ├── CatBoost/
│   │   └── CatBoost.py
│   └── LSTM/
│       └── LSTM.py
│
├── evaluation/
│   └── evaluation.py
│
├── outputs/
│   └── filesplitter.py
│
└── README.md
```

The code in this repository is **my implementation** of the work documented here: the data generation and preparation, pricing implementations, machine-learning models, training pipeline and evaluation code.

---

# References

I used the following papers, reports and resources when researching the mathematical and machine-learning aspects of option pricing:

1. **Neural networks and option pricing** — discussion of neural-network approaches to reproducing option-pricing calculations.  
   https://arxiv.org/html/2510.01446v1#S5

2. **Stanford CS230 project report** — neural-network modelling and architecture considerations.  
   https://cs230.stanford.edu/projects_fall_2019/reports/26260984.pdf

3. **Machine learning for option pricing** — research on machine-learning approaches to derivative pricing.  
   https://arxiv.org/pdf/2307.07657

4. **Applied machine-learning research** — reference used during the wider literature review.  
   https://www.sciencedirect.com/science/article/pii/S0957417420306187

5. **Option Pricing via Machine Learning** — practical quantitative-finance treatment of ML option pricing.  
   https://www.tidy-finance.org/r/option-pricing-via-machine-learning.html

6. **Saeed Bidi — option_pricing** — reference repository used while researching ML approaches to option pricing.  
   https://github.com/saeedbidi/option_pricing

---

# UCL Summer Research Programme

This repository represents **my work on the option-pricing component of the UCL Year 2 Summer Research Programme**. The research and code presented here are my own.

The overall programme project was broader than this repository. Other parts of the group project included topics such as **Antithetic Variates as a variance-reduction technique** and **Put–Call Parity / no-arbitrage consistency checks**. Those topics were part of the wider project but are not presented here as work contained in this repository.

My focus in this repository is the mathematical and computational comparison of **Black–Scholes, Monte Carlo and machine-learning approaches to European option pricing**.

**Inika Prakash**  
UCL Year 2 Summer Research Programme
