# Comparative Analysis of Methods for Option Pricing

### UCL Year 2 Summer Research Programme

This repository contains **my code and implementation work** for a UCL Year 2 Summer Research Programme group project: **A Comparative Analysis of Methods for Option Pricing**.

The research was completed as a group, but the **code contained in this repository was written and developed by me**. My primary contribution to the wider project was the **machine-learning and computational modelling component**, including the data pipelines, model implementations, training, prediction and evaluation framework. The repository is therefore intended to document my implementation of the project rather than imply that I independently carried out every part of the group's research.

The project compared analytical, numerical and machine-learning approaches to pricing **European vanilla options**, combining stochastic calculus, mathematical finance, numerical methods and supervised learning.

> **Important note:** The real market options used in this project are treated as European-style options for the purposes of comparison with Black–Scholes and Monte Carlo. The market data itself comes from U.S. equity option chains.

---

## Project objective

The central machine-learning problem can be expressed as

\[
(S,K,T,r,\sigma)\longrightarrow V,
\]

where:

- \(S\) = current underlying stock price;
- \(K\) = strike price;
- \(T\) = time remaining until expiration;
- \(r\) = risk-free interest rate;
- \(\sigma\) = volatility; and
- \(V\) = theoretical or observed option price.

The goal was to build machine-learning models that take market and contract parameters as inputs and learn to produce an option price.

We then compared the machine-learning approaches with established pricing methods, particularly **Black–Scholes** and **Monte Carlo**, using unseen option data and standard regression metrics.

---

## Why use machine learning for option pricing?

Black–Scholes gives an elegant analytical solution, but it relies on strong assumptions, including constant volatility and an idealised market environment. Real option prices can reflect effects that are difficult to represent using the basic model, including changing implied volatility, liquidity, bid–ask spreads, dividends, market expectations and changes in demand around events such as earnings announcements.

This motivates the machine-learning question:

> **Can a data-driven model learn relationships in observed option prices that are not captured by a simple analytical pricing formula?**

Machine learning does not require us to specify a closed-form relationship between every input and the option price. Instead, the model learns an approximation from examples.

A neural network, for example, can approximate a broad class of continuous input-output relationships given sufficient data and an appropriate architecture. In this project, the same general idea was applied to the option-pricing function.

For background on neural-network approaches to option pricing, see the references at the end of this README.

---

## Project structure

The wider research project considered three categories of pricing methods:

| Approach | Methods | Role in the comparison |
|---|---|---|
| **Analytical** | Black–Scholes | Theoretical benchmark |
| **Numerical** | Monte Carlo, Antithetic Variates | Simulation-based pricing |
| **Machine Learning** | LSTM, Random Forest, XGBoost, CatBoost | Data-driven approximation |

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
       LSTM / Random Forest / XGBoost
                    ↓
                 CatBoost
                    ↓
        Evaluation on Unseen Data
```

---

# Machine Learning

## My primary contribution

The machine-learning component was my **primary contribution to the group project**, and the code in this repository for the data preparation, model development, training, prediction and evaluation was **implemented by me**.

My workflow was:

```text
Data collection / generation
          ↓
Data cleaning and preparation
          ↓
Feature construction
          ↓
Stratification across ticker and maturity
          ↓
Train / test organisation
          ↓
Model training
          ↓
Prediction
          ↓
Evaluation
          ↓
Comparison of models
```

I worked with both synthetic and real market data so that the models could be tested in a controlled theoretical setting and against actual observed option prices.

---

## Datasets

The ML experiments use three main datasets.

### `synthetic_training_data.csv`

The synthetic training set combines:

- historical stock prices;
- historical realised volatility; and
- Black–Scholes pricing.

The target is therefore generated from the Black–Scholes framework. This creates a controlled learning problem where the underlying pricing relationship is known.

Conceptually,

\[
X=(S,K,T,r,\sigma),
\qquad
y=f_{BS}(X).
\]

This allows us to test whether the ML models can learn to reproduce a known theoretical pricing surface.

### `real_training_data.csv`

This dataset contains **actual market option prices** from U.S. equity option chains.

### `real_test_data.csv`

This contains actual market option prices held out for testing. Approximately **20% of the real observations were held out** so that the models could be evaluated on unseen data.

The distinction between these datasets is important because a model trained entirely on Black–Scholes-generated labels is effectively learning to reproduce the Black–Scholes pricing relationship. Training with real market prices instead gives the model an opportunity to learn patterns that are present in observed market data but absent from the simplified theoretical model.

---

## Equity universe

The real-data analysis focused on major U.S. equities, including:

- **Apple Inc. (AAPL)**
- **Microsoft Corporation (MSFT)**
- **NVIDIA Corporation (NVDA)**
- **Tesla, Inc. (TSLA)**
- **Amazon.com, Inc. (AMZN)**

The data were obtained using **Yahoo Finance / `yfinance`**. For the purposes of this project, the observed equity options were treated as European-style contracts so that they could be compared consistently with the Black–Scholes and Monte Carlo frameworks.

---

## Stratified data pipeline

A key part of the ML setup was ensuring that the training data were not dominated by particular assets or expiry dates.

The data pipeline was stratified across:

- **ticker**; and
- **maturity**.

This was intended to expose the models to a balanced range of market conditions and expiry structures rather than allowing a small number of assets or maturities to dominate the training set.

The models were then evaluated on unseen observations to measure out-of-sample performance.

---

# Models

## LSTM — Long Short-Term Memory

LSTMs are recurrent neural networks designed to retain information over sequences. They use three principal gates to control the information flowing through the cell:

1. **Forget gate** — determines which existing information should be discarded.
2. **Input gate** — determines which new information should be stored.
3. **Output gate** — determines which information is used to produce the current output.

The motivation for considering an LSTM was that financial data naturally contain temporal structure. However, there is an important limitation in this project.

An LSTM is most naturally suited to observations arranged in chronological sequences with meaningful and sufficiently regular temporal spacing. The option-chain observations obtained from Yahoo Finance are **not a clean, evenly spaced time series** of historical observations. Consequently, the LSTM was not an ideal architectural match for the available option-chain data.

This helps explain its weaker performance relative to the tree-based models.

The LSTM implementation in this repository was developed as part of my ML contribution.

---

## Random Forest

Random Forest is an ensemble learning method based on multiple decision trees.

A decision tree can be thought of as a sequence of questions that partitions the feature space:

```text
                 Is S > threshold?
                    /        \
                  Yes         No
                  /             \
          Is T > threshold?   ...
             /      \
           Yes       No
           ↓          ↓
       Prediction  Prediction
```

Each tree learns its own set of splits from the training data. The Random Forest combines the predictions of many trees to produce a more robust estimate.

Increasing the number of trees can generally improve stability, but it also increases computational cost. This creates a practical trade-off between predictive performance and computation.

The Random Forest implementation was written by me as part of the ML component.

---

## XGBoost

XGBoost uses gradient-boosted decision trees. Rather than constructing many independent trees and averaging them in the same way as Random Forest, the model builds trees sequentially, with later trees attempting to improve the current prediction.

This makes XGBoost particularly useful for learning nonlinear interactions between variables such as:

\[
S,\ K,\ T,\ r,\ \sigma.
\]

I also considered transformations motivated by the mathematical structure of option pricing, such as relative moneyness and maturity-related quantities.

The XGBoost implementation was written by me as part of the ML component.

---

## CatBoost

CatBoost is a gradient-boosting framework based on decision trees. It is designed to provide strong performance while supporting categorical features and reducing the need for extensive preprocessing in many applications.

In this project, CatBoost provided another tree-based model against which Random Forest and XGBoost could be compared.

The CatBoost implementation was written by me as part of the ML component.

---

## Neural-network approaches and MLPs

The project literature review also considered **feedforward multi-layer perceptrons (MLPs)** as an alternative neural-network approach.

An MLP can learn nonlinear relationships between input features and a continuous target. A typical architecture consists of an input layer, one or more hidden layers, and an output layer:

```text
(S, K, T, r, σ)
       ↓
  Input layer
       ↓
  Hidden layer(s)
       ↓
  Hidden layer(s)
       ↓
  Option price
```

A relevant approach in the literature uses `MLPRegressor` and selects architectural and optimisation parameters such as the number of hidden layers, neurons, regularisation strength and learning rate through hyperparameter search.

The references below were used to inform the ML research and comparison. The repository's implemented models should be distinguished from methods discussed only in the literature review.

---

# Analytical and numerical pricing

Although my primary contribution was machine learning, the ML experiments were designed around the wider pricing framework developed for the group project.

## Black–Scholes

For a European call,

\[
C=S\Phi(d_1)-Ke^{-rT}\Phi(d_2),
\]

where

\[
d_1=\frac{\ln(S/K)+(r+\frac12\sigma^2)T}{\sigma\sqrt{T}}
\]

and

\[
d_2=d_1-\sigma\sqrt{T}.
\]

For a European put,

\[
P=Ke^{-rT}\Phi(-d_2)-S\Phi(-d_1).
\]

Black–Scholes provided the principal analytical benchmark.

Implementation:

[`models/BlackScholes.py`](models/BlackScholes.py)

---

## Monte Carlo

Under the risk-neutral measure,

\[
dS_t=rS_tdt+\sigma S_tdW_t^{\mathbb Q}.
\]

Therefore,

\[
S_T=S_0\exp\left[
\left(r-\frac12\sigma^2\right)T+\sigma\sqrt{T}Z
\right],
\qquad Z\sim N(0,1).
\]

The option price can then be estimated using

\[
\widehat V_N=\frac{e^{-rT}}{N}
\sum_{i=1}^{N}\Phi(S_T^{(i)}).
\]

Monte Carlo therefore provides a numerical approximation to the same risk-neutral expectation represented analytically by Black–Scholes.

Implementation:

[`models/MonteCarlo.py`](models/MonteCarlo.py)

---

## Antithetic Variates

We also implemented **Antithetic Variates** as a variance-reduction technique for Monte Carlo.

For each simulated standard-normal draw \(Z\), the corresponding antithetic draw \(-Z\) is also used. The two paths are negatively correlated, so averaging their payoffs can reduce estimator variance.

This allowed us to compare ordinary Monte Carlo with a variance-reduced simulation approach.

---

## Put–Call Parity

For European calls and puts with the same underlying, strike, maturity and interest rate,

\[
C-P=S-Ke^{-rT}.
\]

Put–Call Parity was used as a no-arbitrage consistency check across the pricing framework.

---

# Evaluation

The models were evaluated using three main regression metrics.

### Mean Absolute Error — MAE

\[
MAE=\frac1n\sum_{i=1}^{n}|y_i-\widehat y_i|.
\]

MAE represents the average absolute difference between the true option price and the model prediction, measured in price units.

### Root Mean Squared Error — RMSE

\[
RMSE=\sqrt{\frac1n\sum_{i=1}^{n}(y_i-\widehat y_i)^2}.
\]

RMSE penalises larger errors more strongly than MAE.

### \(R^2\) score

\[
R^2=1-\frac{\sum_i(y_i-\widehat y_i)^2}
{\sum_i(y_i-\bar y)^2}.
\]

\(R^2\) measures the proportion of variation in the target explained by the model relative to a constant-mean benchmark. It should not be interpreted literally as the percentage of market behaviour that a model "understands".

Implementation:

[`evaluation/evaluation.py`](evaluation/evaluation.py)

---

# Results and interpretation

The central comparison was between models trained using synthetic Black–Scholes-derived labels and models trained using actual market option prices.

### Synthetic training data

A model trained on Black–Scholes-generated labels is effectively learning to reproduce the Black–Scholes pricing function. This can produce very strong approximation performance, but it also means that the ML model inherits the assumptions embedded in the synthetic data.

In other words:

> If the training labels are produced by Black–Scholes, the machine-learning model is learning a sophisticated approximation to a Black–Scholes calculator.

This makes synthetic data useful for testing function approximation, but it does not demonstrate that ML has discovered information beyond the Black–Scholes model.

### Real market training data

Training against actual market option prices allows the models to learn relationships present in the observed market data. This can capture effects that a constant-volatility Black–Scholes specification does not directly model.

In our experiments, the real-data models generally produced **lower errors and higher \(R^2\)** than the corresponding synthetic-data models when evaluated on the real market test set.

This result should be interpreted carefully: the comparison is not simply "ML versus Black–Scholes". The models are being trained against different target-generating processes, so the result also reflects the difference between synthetic theoretical labels and noisy real market observations.

---

## LSTM results

The LSTM performed substantially worse than the tree-based models in the experiments.

The main reason is the mismatch between the architecture and the structure of the available data. LSTMs are designed to exploit meaningful sequential information, whereas the option-chain observations used here do not form a clean, evenly spaced chronological sequence.

There is therefore a useful methodological lesson from this result:

> **A more sophisticated neural architecture is not automatically a better model if its inductive bias does not match the data.**

The LSTM result should consequently not be interpreted as evidence that neural networks are intrinsically unsuitable for option pricing. A properly constructed time-series dataset or a feedforward architecture such as an MLP would represent a different experiment.

---

## Tree-based model comparison

The **Random Forest, XGBoost and CatBoost** models produced broadly similar performance in the project, with the three tree-based approaches generally outperforming the LSTM.

The comparison therefore also becomes a question of computational cost and implementation complexity rather than simply selecting the model with the smallest error.

A useful qualitative ordering from the project was:

```text
Lower computational cost                         Higher computational cost

XGBoost  →  Random Forest  →  CatBoost  →  LSTM
```

The exact cost depends on hyperparameters, hardware and implementation, so this ordering should be treated as an empirical project-level observation rather than a universal property of the algorithms.

---

# Synthetic vs. real data: trade-offs

## Synthetic data

### Advantages

- Controlled data-generating process.
- Known theoretical pricing function.
- Useful for testing whether a model can approximate Black–Scholes.
- Large datasets can be generated without depending on market-data availability.
- Makes it easier to isolate model approximation error.

### Limitation

The model can simply learn the assumptions of the Black–Scholes framework because those assumptions are embedded in the target labels.

## Real option data

### Advantages

- Represents actual observed market prices.
- Can contain patterns not captured by the basic Black–Scholes model.
- Allows the ML models to learn from real market behaviour.
- Does not have the same synthetic-label limitation.

### Limitations

- Market data contain noise.
- Observations can be affected by liquidity and bid–ask spreads.
- The available data are not a clean evenly spaced time series, limiting the usefulness of sequence-based architectures such as LSTM.
- Market prices do not necessarily satisfy the assumptions required by the Black–Scholes model.

---

# Why this comparison matters

The project is not simply asking which algorithm produces the lowest RMSE.

It is asking how different representations of the same pricing problem behave:

\[
\boxed{
\text{Analytical}\rightarrow\text{Numerical}\rightarrow\text{Data-driven}
}
\]

**Black–Scholes** gives a mathematically elegant closed-form solution under restrictive assumptions.

**Monte Carlo** relaxes the need for a closed-form solution by estimating the risk-neutral expectation numerically.

**Antithetic Variates** improve the efficiency of that numerical estimation.

**Machine learning** instead learns an approximation from examples, potentially allowing the model to capture relationships present in market data that are not explicitly represented in the Black–Scholes formula.

This makes the comparison useful from both a quantitative-finance and machine-learning perspective.

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
├── .gitignore
├── .gitattributes
└── README.md
```

**All code in this repository was implemented by me.** The repository is my implementation/codebase from the group research project; the group aspect refers to the research project itself and does not mean that the code here was jointly authored.

---

# Reproducibility / implementation note

The project was developed as a research codebase rather than a packaged Python library. Some scripts may contain local development paths or assumptions about the available datasets, so paths may need to be adjusted when reproducing the experiments on another machine.

The main Python ecosystem used includes tools such as:

- Python
- NumPy
- pandas
- SciPy
- scikit-learn
- XGBoost
- CatBoost
- TensorFlow / Keras
- `yfinance`

---

# References and further reading

The following references informed the mathematical, machine-learning and option-pricing aspects of the project. They are included as **research references**, not as claims that the referenced implementations were copied into this repository.

1. **Neural networks for option pricing** — discussion of neural-network approaches to financial pricing and replication of Monte Carlo calculations.  
   urlarXiv — Neural Network / Option Pricing referencehttps://arxiv.org/html/2510.01446v1#S5

2. **Stanford CS230 project report** — neural-network approaches and model architecture considerations.  
   urlStanford CS230 project reporthttps://cs230.stanford.edu/projects_fall_2019/reports/26260984.pdf

3. **Machine learning for option pricing** — research on ML approaches to option pricing.  
   urlarXiv — Option Pricing / Machine Learninghttps://arxiv.org/pdf/2307.07657

4. **Applied machine learning research** — reference used during the wider literature review.  
   urlScienceDirect articlehttps://www.sciencedirect.com/science/article/pii/S0957417420306187

5. **Option Pricing via Machine Learning** — practical quantitative-finance treatment of machine-learning approaches to option pricing.  
   urlTidy Finance — Option Pricing via Machine Learninghttps://www.tidy-finance.org/r/option-pricing-via-machine-learning.html

6. **Option pricing machine-learning repository** — used as a reference point during the project.  
   urlSaeed Bidi — option_pricinghttps://github.com/saeedbidi/option_pricing

---

# Project status

**Completed — UCL Year 2 Summer Research Programme.**

### Group project

**A Comparative Analysis of Methods for Option Pricing**

### My primary contribution

**Machine Learning / Computational Modelling**

This included the implementation of the ML pipeline, model development, data preparation, training, prediction and evaluation contained in this repository.

**Author: Inika Prakash**  
UCL — Year 2 Summer Research Programme
