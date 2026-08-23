# Comparative Analysis of Methods for Option Pricing

### UCL Year 2 Summer Research Programme

This repository contains the work from my **UCL Year 2 Summer Research Programme** group project, *A Comparative Analysis of Methods for Option Pricing*.

The project investigated different approaches to pricing **European vanilla options**, bringing together stochastic calculus, mathematical finance, numerical methods, and machine learning. Rather than treating the methods as isolated techniques, we used the same pricing problem to compare analytical, numerical, and data-driven approaches.

My **primary contribution was the machine-learning component**, including the design of the supervised-learning pipeline, data preparation, model development, training, prediction, and evaluation. I also contributed to the wider pricing framework and the implementation and validation of the classical methods.

---

## Project overview

The project considered three broad approaches to option pricing:

| Approach | Methods | Purpose |
|---|---|---|
| **Analytical** | Black–Scholes | Closed-form benchmark under the model assumptions |
| **Numerical** | Monte Carlo, Antithetic Variates | Simulation-based approximation of the risk-neutral expectation |
| **Machine Learning** | Random Forest, XGBoost, CatBoost, LSTM | Learn the mapping from market parameters to option prices |

The overall workflow was:

```text
Brownian Motion / Geometric Brownian Motion
                    ↓
          Risk-Neutral Framework
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
  Black–Scholes            Monte Carlo
  Analytical Price       + Antithetic Variates
        │                       │
        └───────────┬───────────┘
                    ↓
          Validation / Benchmarking
                    ↓
          Machine Learning Pipeline
                    ↓
     Synthetic + Real Market Data
                    ↓
 Random Forest / XGBoost / CatBoost / LSTM
                    ↓
        Evaluation on Unseen Data
```

The central machine-learning problem was to approximate the pricing function

\[
(S,K,T,r,\sigma)\longrightarrow V,
\]

where:

- \(S\) = underlying asset price;
- \(K\) = strike price;
- \(T\) = time to maturity;
- \(r\) = risk-free interest rate;
- \(\sigma\) = volatility; and
- \(V\) = option price.

This formulation allowed us to investigate whether flexible machine-learning models could learn the complex nonlinear relationships between market parameters and option prices.

---

## My contribution: machine learning

My main responsibility within the group was the **machine-learning side of the project**.

I first spent time understanding the mathematical structure of the pricing problem before implementing the ML pipeline. This was important because the goal was not simply to train a model to minimise an error metric; we wanted to understand what the models were learning and how their predictions compared with established pricing methods.

My work covered the main stages of the pipeline:

```text
Data preparation
      ↓
Feature construction / preprocessing
      ↓
Train-test organisation
      ↓
Model training
      ↓
Prediction
      ↓
Evaluation
      ↓
Comparison across models and datasets
```

The models I worked with were:

- **Random Forest**
- **XGBoost**
- **CatBoost**
- **LSTM**

I investigated these models as different ways of approximating the option-pricing function rather than assuming that a more complex model would automatically produce a better result.

---

## Synthetic vs. real market data

A major part of the ML analysis was comparing performance under two different data-generating settings.

### 1. Synthetic data

Synthetic option data were generated using the **Black–Scholes framework**. This gave us a controlled environment in which the underlying pricing function was known.

Conceptually,

\[
X_i=(S_i,K_i,T_i,r_i,\sigma_i),
\qquad
V_i=f_{BS}(X_i).
\]

This allowed us to ask a clean question:

> **Can a machine-learning model learn a known theoretical pricing function?**

Because the target prices came from the Black–Scholes model, this experiment provided a useful benchmark for the approximation capabilities of the ML models.

### 2. Real U.S. equity options

We also worked with **real U.S. equity option-chain data from major U.S. equities**.

This introduced a much less idealised setting. Real market prices can reflect effects that are not fully captured by the basic Black–Scholes assumptions, including changing implied volatility, bid–ask spreads, liquidity, dividends, and other market effects.

This comparison was important because strong performance on synthetic Black–Scholes data does not necessarily imply equally strong performance on real market observations.

---

## Stratified data pipeline

One of the important parts of the machine-learning setup was making sure that the training data were not dominated by particular assets or maturities.

The data pipeline was **stratified across ticker and maturity**, allowing the models to be trained on a more balanced representation of the underlying market conditions and expiry structure.

This helped make the evaluation more meaningful by reducing the risk that performance was driven primarily by a particular ticker or maturity bucket.

The models were subsequently evaluated on **unseen data**, allowing us to assess out-of-sample predictive performance rather than simply measuring how well the models fitted their training observations.

---

## Machine-learning models

### Random Forest

Random Forest provides an ensemble of decision trees, with predictions aggregated across the individual trees.

It gives a useful nonlinear baseline for the pricing problem and allows the relationship between the input variables and option price to be modelled without imposing a specific functional form.

Implementation:

[`models/Random Forest/RandomForest.py`](models/Random%20Forest/RandomForest.py)

### XGBoost

XGBoost uses gradient-boosted decision trees to construct an additive approximation to the pricing function.

This was particularly useful for investigating nonlinear interactions between variables such as underlying price, strike, maturity, and volatility.

Implementation:

[`models/XGBoost/XGBoost.py`](models/XGBoost/XGBoost.py)

### CatBoost

CatBoost provided another gradient-boosting approach for modelling the nonlinear option-pricing relationship.

I used it as a comparison against the other tree-based models, allowing us to investigate whether different boosting methodologies produced materially different predictive behaviour.

Implementation:

[`models/CatBoost/CatBoost.py`](models/CatBoost/CatBoost.py)

### LSTM

I also investigated an LSTM-based neural-network architecture.

The LSTM experiment was intended as an investigation of whether a neural architecture could learn the nonlinear pricing relationship from the available features. The input variables should not be interpreted as a conventional time series of consecutive observations; the experiment was instead focused on function approximation.

Implementation:

[`models/LSTM/LSTM.py`](models/LSTM/LSTM.py)

---

## Classical pricing methods

Although my primary contribution was machine learning, the ML work was developed within the wider theoretical and computational pricing framework of the project.

### Black–Scholes

For a European call, the Black–Scholes price is

\[
C=S\Phi(d_1)-Ke^{-rT}\Phi(d_2),
\]

where

\[
d_1=\frac{\ln(S/K)+(r+\frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}
\]

and

\[
d_2=d_1-\sigma\sqrt{T}.
\]

The corresponding European put price is

\[
P=Ke^{-rT}\Phi(-d_2)-S\Phi(-d_1).
\]

The Black–Scholes implementation provided the analytical benchmark against which the numerical and machine-learning approaches could be compared.

Implementation:

[`models/BlackScholes.py`](models/BlackScholes.py)

### Monte Carlo

Under the risk-neutral measure, the terminal asset price can be simulated using

\[
S_T=S_0\exp\left[
\left(r-\frac{1}{2}\sigma^2\right)T
+\sigma\sqrt{T}Z
\right],
\qquad Z\sim N(0,1).
\]

For a payoff \(\Phi(S_T)\), Monte Carlo estimates the option value using

\[
\widehat V_N=
\frac{e^{-rT}}{N}
\sum_{i=1}^{N}\Phi(S_T^{(i)}).
\]

This provides a numerical approximation to the same risk-neutral expectation represented analytically by Black–Scholes.

Implementation:

[`models/MonteCarlo.py`](models/MonteCarlo.py)

### Antithetic Variates

We also investigated **Antithetic Variates** as a variance-reduction technique for Monte Carlo simulation.

Instead of using only a simulated draw \(Z\), the method pairs it with \(-Z\). Because the two draws are negatively correlated, averaging the corresponding payoffs can reduce the variance of the estimator without requiring the same proportional increase in independent simulations.

This provided an additional comparison between straightforward Monte Carlo and a variance-reduced numerical approach.

---

## Put–Call Parity and validation

The project also included validation against fundamental no-arbitrage relationships.

For European calls and puts with the same underlying, strike, maturity, and interest rate, Put–Call Parity gives

\[
C-P=S-Ke^{-rT}.
\]

We used this relationship as a consistency check across the pricing framework.

More generally, the project was concerned not only with numerical prediction error but also with whether the resulting prices were consistent with the mathematical and financial structure expected of European option prices.

---

## Evaluation

The machine-learning models were benchmarked on unseen data using standard regression metrics.

### Root Mean Squared Error

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\widehat y_i)^2}.
\]

RMSE places greater weight on larger prediction errors.

### Mean Absolute Error

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\widehat y_i|.
\]

MAE gives the average absolute difference between the predicted and observed prices.

### Coefficient of determination

\[
R^2=1-
\frac{\sum_i(y_i-\widehat y_i)^2}
{\sum_i(y_i-\bar y)^2}.
\]

Together, these metrics provided a quantitative basis for comparing the different ML approaches and assessing their out-of-sample performance.

Implementation:

[`evaluation/evaluation.py`](evaluation/evaluation.py)

---

## Why compare synthetic and real data?

The distinction between the two datasets is central to the project.

With synthetic Black–Scholes data, the target relationship is known and generated from a controlled mathematical model. If an ML model performs poorly here, this gives us information about its ability to approximate a relatively well-defined function.

With real option-chain data, the problem becomes more realistic. Market prices can depart from the assumptions of the Black–Scholes model, and the relationship between the observed market parameters and prices contains effects that are not captured by a simple theoretical pricing formula.

This means the two experiments answer different questions:

| Dataset | Main question |
|---|---|
| **Synthetic** | Can the ML model approximate a known theoretical pricing function? |
| **Real market data** | How well does the model generalise to observed option prices? |

This comparison was particularly useful for separating **function-approximation capability** from the effects of **model misspecification and real market structure**.

---

## Mathematical foundation

The project was grounded in the standard continuous-time model of asset prices.

Under geometric Brownian motion,

\[
dS_t=\mu S_t\,dt+\sigma S_t\,dW_t.
\]

Under the risk-neutral measure,

\[
dS_t=rS_t\,dt+\sigma S_t\,dW_t^{\mathbb Q}.
\]

This leads to the risk-neutral valuation expression

\[
V_0=e^{-rT}\mathbb{E}^{\mathbb Q}[\Phi(S_T)].
\]

This expectation provides a useful way of viewing the whole comparison:

- **Black–Scholes** evaluates the expectation analytically under the model assumptions.
- **Monte Carlo** estimates the expectation numerically.
- **Antithetic Variates** improve the Monte Carlo estimator through variance reduction.
- **Machine learning** learns an approximation to the resulting relationship between the market parameters and the option price.

The approaches are therefore connected through the same underlying pricing problem.

---

## Repository structure

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
├── .gitignore
├── .gitattributes
└── README.md
```

The repository also contains supporting model-training artefacts and generated datasets.

---

## Project workflow

The overall experimental workflow was:

```text
1. Establish the theoretical pricing framework
                ↓
2. Implement Black–Scholes pricing
                ↓
3. Implement Monte Carlo simulation
                ↓
4. Investigate Antithetic Variates
                ↓
5. Validate pricing relationships such as Put–Call Parity
                ↓
6. Prepare synthetic and real option datasets
                ↓
7. Stratify data across ticker and maturity
                ↓
8. Train Random Forest / XGBoost / CatBoost / LSTM
                ↓
9. Evaluate on unseen data
                ↓
10. Compare analytical, numerical and ML approaches
```

---

## Key takeaways

The project gave me the opportunity to work across several areas that are closely connected in quantitative finance:

- **Quantitative Finance** — European derivatives and option pricing
- **Stochastic Calculus** — Brownian motion, geometric Brownian motion and risk-neutral dynamics
- **Mathematical Finance** — Black–Scholes and no-arbitrage relationships
- **Numerical Methods** — Monte Carlo simulation and variance reduction
- **Machine Learning** — supervised learning for nonlinear function approximation
- **Python** — data processing, modelling and evaluation
- **Statistical Modelling** — out-of-sample model comparison
- **Data Analysis** — working with synthetic and real U.S. equity option data

More importantly, the project helped me understand the relationship between a mathematically derived pricing model and a machine-learning approximation of the same problem. My main focus was on building and evaluating that ML approximation while keeping the underlying financial and mathematical structure in view.

---

## Future directions

There are several directions I would be interested in exploring further:

- comparing performance across different moneyness and maturity regimes;
- analysing model errors by ticker and maturity rather than only aggregate metrics;
- investigating implied volatility as the prediction target rather than option price;
- incorporating additional market variables into the learning problem;
- testing alternative volatility and pricing models;
- evaluating computational cost alongside predictive accuracy; and
- investigating whether the learned pricing surfaces preserve additional no-arbitrage constraints.

---

## Project status

**Completed — UCL Year 2 Summer Research Programme group project.**

My primary contribution was the **machine-learning component**, while the project as a whole covered analytical pricing, numerical simulation, validation, and machine-learning approaches to European option pricing.

---

## Author / Project Team

This was a **group research project** completed as part of the UCL Year 2 Summer Research Programme.

**Inika Prakash**  
Primary contribution: Machine Learning / Computational Modelling

UCL — Year 2 Summer Research Programme
