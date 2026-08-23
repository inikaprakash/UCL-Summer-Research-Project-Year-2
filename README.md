# UCL Summer Research Project — Year 2

## Option Pricing: From Stochastic Calculus to Machine Learning

This repository contains my UCL Year 2 Summer Research Project on **option pricing**. I use the project to work through the problem from first principles — starting with probability and stochastic calculus, deriving the Black–Scholes framework, implementing Monte Carlo pricing, and then investigating whether machine-learning models can learn the resulting pricing surface.

The main question I am exploring is:

> **How well can different computational methods approximate option prices, and what do I learn by comparing them with the mathematical model that generates the prices?**

I deliberately treat machine learning as the final part of the project rather than the starting point. Before asking a model to learn a pricing function, I want to understand the probability, stochastic calculus and no-arbitrage arguments that give the function its structure.

---

## Project overview

The project follows the chain:

```text
Geometric Brownian Motion
        ↓
     Itô's Lemma
        ↓
 Risk-Neutral Valuation
        ↓
Black–Scholes Formula / PDE
        ↓
   Monte Carlo Pricing
        ↓
 Synthetic Data Generation
        ↓
 Machine-Learning Models
        ↓
 Evaluation and Comparison
```

At a high level, I treat option pricing as a function-approximation problem. For a European option,

\[
V=f(S,K,T,r,\sigma),
\]

where \(S\) is the current underlying price, \(K\) is the strike, \(T\) is time to maturity, \(r\) is the continuously compounded risk-free rate, \(\sigma\) is volatility, and \(V\) is the option value.

Under Black–Scholes assumptions, this function is not arbitrary. It is constrained by the underlying stochastic process and by no-arbitrage. This gives me a useful theoretical benchmark against which I can evaluate numerical and machine-learning approaches.

---

## 1. The mathematical starting point

I model the underlying asset using geometric Brownian motion:

\[
dS_t=\mu S_t\,dt+\sigma S_t\,dW_t.
\]

For pricing, I work under the risk-neutral measure \(\mathbb Q\), giving

\[
dS_t=rS_t\,dt+\sigma S_t\,dW_t^{\mathbb Q}.
\]

Applying Itô's lemma to \(\log S_t\) gives

\[
d\log S_t=\left(r-\frac12\sigma^2\right)dt+\sigma dW_t^{\mathbb Q}.
\]

Therefore,

\[
S_T=S_0\exp\left[\left(r-\frac12\sigma^2\right)T+\sigma\sqrt{T}Z\right],\qquad Z\sim N(0,1).
\]

This result is important throughout the repository because it provides the distribution from which I can simulate terminal asset prices and derive the Black–Scholes valuation formula.

---

## 2. Risk-neutral valuation

Once the risk-neutral dynamics are established, the value of a derivative with terminal payoff \(\Phi(S_T)\) is

\[
V_0=e^{-rT}\mathbb E^{\mathbb Q}[\Phi(S_T)].
\]

For a European call,

\[
C_0=e^{-rT}\mathbb E^{\mathbb Q}[(S_T-K)^+],
\]

while for a European put,

\[
P_0=e^{-rT}\mathbb E^{\mathbb Q}[(K-S_T)^+].
\]

This expectation is the common mathematical object behind several parts of my implementation:

1. **Black–Scholes** evaluates it analytically.
2. **Monte Carlo** approximates it numerically by simulation.
3. **Machine learning** attempts to approximate the resulting input–output relationship from data.

This is why I view these approaches as different computational treatments of the same underlying pricing problem rather than as completely separate models.

---

## 3. Black–Scholes pricing

For a European call, the closed-form Black–Scholes price is

\[
C=S\Phi(d_1)-Ke^{-rT}\Phi(d_2),
\]

where

\[
d_1=\frac{\log(S/K)+(r+\frac12\sigma^2)T}{\sigma\sqrt{T}},
\]

and

\[
d_2=d_1-\sigma\sqrt{T}.
\]

For a European put,

\[
P=Ke^{-rT}\Phi(-d_2)-S\Phi(-d_1).
\]

Here \(\Phi\) is the standard normal cumulative distribution function.

The implementation is in [`models/BlackScholes.py`](models/BlackScholes.py). It calculates prices for the real test data and reports MAE, RMSE, MAPE and \(R^2\).

I use Black–Scholes as the principal theoretical benchmark for the rest of the project.

---

## 4. The Black–Scholes PDE

I also consider the PDE formulation of the same pricing problem.

For an option value \(V(S,t)\), Itô's lemma gives

\[
dV=\left(V_t+\mu SV_S+\frac12\sigma^2S^2V_{SS}\right)dt+\sigma SV_SdW_t.
\]

Constructing the delta-hedged portfolio

\[
\Pi=V-\Delta S
\]

and choosing \(\Delta=V_S\) removes the stochastic component. Applying the no-arbitrage condition gives the Black–Scholes PDE:

\[
\boxed{V_t+\frac12\sigma^2S^2V_{SS}+rSV_S-rV=0}
\]

with the appropriate terminal payoff condition.

One of the main mathematical ideas I want to demonstrate is therefore the connection

\[
\boxed{\text{SDE}\leftrightarrow\text{Itô calculus}\leftrightarrow\text{no-arbitrage}\leftrightarrow\text{PDE}\leftrightarrow\text{risk-neutral expectation}}
\]

These are different representations of the same underlying pricing framework.

---

## 5. Monte Carlo pricing

Monte Carlo gives me a numerical way of evaluating the risk-neutral expectation without using the closed-form Black–Scholes formula.

I generate independent standard normal variables

\[
Z_1,\ldots,Z_N\overset{iid}{\sim}N(0,1)
\]

and simulate

\[
S_T^{(i)}=S\exp\left[\left(r-\frac12\sigma^2\right)T+\sigma\sqrt{T}Z_i\right].
\]

The Monte Carlo estimator is then

\[
\widehat V_N=\frac{e^{-rT}}{N}\sum_{i=1}^{N}\Phi(S_T^{(i)}).
\]

By the law of large numbers,

\[
\widehat V_N\rightarrow V\qquad\text{as }N\rightarrow\infty.
\]

The variance decreases at the usual Monte Carlo rate,

\[
\operatorname{Var}(\widehat V_N)=O(N^{-1}),
\]

so the standard error scales as \(O(N^{-1/2})\).

My implementation is in [`models/MonteCarlo.py`](models/MonteCarlo.py). The current implementation uses 10,000 simulations with a fixed random seed.

The important point is that Monte Carlo is **not a different pricing model** in this setting. It is a numerical approximation to the same risk-neutral expectation used by the analytical solution.

---

## 6. Synthetic data: learning a known pricing function

The synthetic experiment lets me ask a particularly controlled machine-learning question:

> **Can a machine-learning model learn a pricing function when I already know the mathematical function generating the targets?**

The data-generation process uses market-style inputs and the Black–Scholes formula to produce target prices. Conceptually,

\[
X_i=(S_i,K_i,T_i,r_i,\sigma_i),\qquad y_i=f_{BS}(X_i).
\]

The ML task is therefore

\[
\boxed{\widehat f\approx f_{BS}}.
\]

This gives me something that is difficult to obtain with purely historical data: a known ground-truth pricing function.

The main data-generation code is in [`data/generate_datasets.py`](data/generate_datasets.py). The repository also contains the generated synthetic training data in `data/synthetic_training_data.csv`.

---

## 7. Real option data

The real-data experiment is deliberately different from the synthetic experiment.

The real data contain market-derived quantities such as

\[
(S,K,T,r,\sigma_{imp},\text{option type},V_{market}).
\]

In actual markets, implied volatility is generally not constant. It can vary with strike and maturity:

\[
\sigma_{imp}=\sigma_{imp}(K,T).
\]

This gives rise to the implied-volatility surface, including effects such as volatility smiles and skews.

As a result, observed market prices do not have to lie exactly on the constant-volatility Black–Scholes surface.

There are also practical effects that are not captured by the basic model, including dividends, bid–ask spreads, liquidity effects, stochastic volatility, jumps, market microstructure effects, and noise in observed prices.

This makes the synthetic-versus-real comparison particularly useful. A model that performs well on synthetic Black–Scholes data but less well on real observations may be revealing model misspecification rather than simply having poor predictive ability.

The repository contains the real training and test datasets in `data/real_training_data.csv` and `data/real_test_data.csv`.

---

## 8. Machine-learning models

I compare several different model classes. I am interested not only in which model gives the lowest prediction error, but also in what kind of approximation each model is making.

### Random Forest

A decision tree partitions the input space into regions and produces predictions based on those regions. A random forest averages predictions from many trees:

\[
\widehat f(x)=\frac1B\sum_{b=1}^{B}f_b(x).
\]

My implementation is in [`models/Random Forest/RandomForest.py`](models/Random%20Forest/RandomForest.py).

### XGBoost

XGBoost constructs an additive approximation by repeatedly adding trees that improve the current model:

\[
\widehat f_M(x)=\sum_{m=1}^{M}\eta h_m(x).
\]

I also use mathematically motivated transformations such as log-moneyness, \(\log(S/K)\), and the square-root maturity transformation, \(\sqrt{T}\).

The implementation is in [`models/XGBoost/XGBoost.py`](models/XGBoost/XGBoost.py).

### CatBoost

CatBoost is another gradient-boosted tree method that I use as a flexible nonlinear approximation to the option-pricing surface.

The implementation is in [`models/CatBoost/CatBoost.py`](models/CatBoost/CatBoost.py).

### LSTM

I also experiment with an LSTM architecture. The network standardises the pricing variables and passes them through recurrent layers followed by dense layers.

There is an important limitation to how I interpret this experiment: the six pricing variables are reshaped as a sequence, but they are **not six consecutive time observations**. Therefore, I treat this as an experiment in nonlinear function approximation using an LSTM architecture rather than evidence that the network has learned temporal market dynamics.

The implementation is in [`models/LSTM/LSTM.py`](models/LSTM/LSTM.py).

---

## 9. Why the feature transformations matter

The Black–Scholes formula itself suggests useful coordinates.

In

\[
d_1=\frac{\log(S/K)+(r+\frac12\sigma^2)T}{\sigma\sqrt T},
\]

two quantities immediately stand out:

\[
\log(S/K)
\]

and

\[
\sigma\sqrt T.
\]

The first captures relative moneyness, while the second reflects the natural scale of Brownian uncertainty over the time horizon.

I therefore view feature engineering as more than a generic preprocessing step. In this project it is an attempt to expose structure already present in the mathematical pricing model.

---

## 10. Evaluation

I evaluate the models using standard regression metrics.

### Mean Absolute Error

\[
MAE=\frac1n\sum_{i=1}^{n}|y_i-\widehat y_i|.
\]

### Root Mean Squared Error

\[
RMSE=\sqrt{\frac1n\sum_{i=1}^{n}(y_i-\widehat y_i)^2}.
\]

RMSE places more weight on larger errors than MAE.

### Mean Absolute Percentage Error

\[
MAPE=\frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i-\widehat y_i}{y_i}\right|.
\]

### Coefficient of determination

\[
R^2=1-\frac{\sum_i(y_i-\widehat y_i)^2}{\sum_i(y_i-\bar y)^2}.
\]

The evaluation code is in [`evaluation/evaluation.py`](evaluation/evaluation.py).

I use these metrics as a starting point, but I do not regard predictive accuracy alone as sufficient for evaluating an option-pricing model.

---

## 11. Financial and mathematical consistency

A learned pricing surface should ideally respect structural relationships implied by no-arbitrage and option payoffs.

For example, European calls and puts satisfy put–call parity:

\[
C-P=S-Ke^{-rT}.
\]

For a European call, the price should also increase with the underlying price and, under the standard assumptions, volatility. Other useful checks include sensible behaviour with respect to strike and maturity and the absence of obvious arbitrage opportunities.

This is an important distinction in my project:

> **A model can have good regression metrics while still producing an economically or mathematically inconsistent pricing surface.**

For that reason, I am interested in both pointwise prediction error and the structure of the function being learned.

---

## 12. Repository structure

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

The repository also contains CatBoost training artefacts under `catboost_info/`.

---

## 13. How I approach the comparison

My comparison is built around three levels of increasing distance from the theoretical model:

### Level 1 — Analytical benchmark

Black–Scholes gives a closed-form price under its assumptions.

### Level 2 — Numerical approximation

Monte Carlo approximates the same risk-neutral expectation through simulation.

### Level 3 — Data-driven approximation

Random Forest, XGBoost, CatBoost and LSTM learn an approximation to the pricing relationship from data.

This hierarchy lets me separate several questions that are often mixed together:

- How accurately can I calculate the theoretical price?
- How accurately can I approximate it numerically?
- How accurately can a machine-learning model reproduce it?
- How does that behaviour change when I move from synthetic data to real market data?
- Does a model that predicts well also preserve the mathematical structure expected of an option-pricing surface?

---

## 14. What I am trying to learn from the project

The project is not simply a competition between algorithms.

I am using option pricing as a setting in which I can connect several areas of mathematics and computation that are often studied separately:

\[
\boxed{\text{Probability}\rightarrow\text{Stochastic Calculus}\rightarrow\text{Mathematical Finance}\rightarrow\text{Numerical Methods}\rightarrow\text{Machine Learning}}
\]

The Black–Scholes model gives me a mathematically understood benchmark. Monte Carlo shows how that benchmark can be recovered computationally. Synthetic data then let me test whether ML models can learn a known pricing function. Real data finally introduce model error and market structure that the idealised Black–Scholes assumptions do not capture.

That progression is the main reason I chose this problem: it lets me investigate machine learning without losing sight of the mathematical structure of the problem being learned.

---

## 15. Current limitations and next steps

There are several directions I would like to develop further:

- compare the models across a wider range of strikes, maturities and volatility regimes;
- examine error as a function of moneyness and maturity rather than only aggregate metrics;
- test no-arbitrage constraints directly on the learned surfaces;
- investigate implied-volatility rather than price as the prediction target;
- experiment with alternative volatility models;
- improve the treatment of real market data, including bid–ask information and dividends;
- compare computational cost as well as predictive accuracy; and
- investigate whether architectures designed specifically for tabular function approximation are more appropriate than the LSTM formulation used here.

I also want to distinguish more clearly between **interpolation**, where the model predicts within the region represented by the training data, and **extrapolation**, where the model is asked to price options outside that region. This is particularly important for financial applications because a low test error does not automatically imply robust behaviour away from the observed data.

---

## 16. Running the project

The repository is organised as a collection of Python scripts rather than a packaged Python project. The scripts currently use common scientific Python and machine-learning libraries, including NumPy, pandas, SciPy, scikit-learn, XGBoost, CatBoost and TensorFlow/Keras.

A typical workflow is:

```text
1. Generate / inspect the datasets
        ↓
2. Run the Black–Scholes benchmark
        ↓
3. Run Monte Carlo
        ↓
4. Train the ML models
        ↓
5. Evaluate predictions
        ↓
6. Compare accuracy and structural behaviour
```

Some scripts contain local filesystem paths from my development environment, so these may need to be updated before running the project on another machine.

---

## Project status

**Year 2 UCL Summer Research Project — in development.**

The repository is intended to document both the mathematical reasoning and the computational experiments behind my investigation of option pricing.

---

## Author

**Inika Prakash**  
UCL — Year 2 Summer Research Project
