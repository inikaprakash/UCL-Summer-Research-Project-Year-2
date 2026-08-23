# UCL Summer Research Project — Year 2

## Mathematical study of option pricing: Black–Scholes, Monte Carlo and machine learning

This repository contains the computational work for a UCL Year 2/3 Mathematics summer research project on **European option pricing**. The central question is not simply which model gives the smallest prediction error, but:

> **How much of the mathematical structure of option pricing can be recovered, approximated, or replaced by data-driven models?**

The project compares classical mathematical finance with four supervised-learning approaches, using both **synthetic data generated from the Black–Scholes model** and **real option-chain data**.

The mathematical progression is

\[
\text{stochastic model}
\;\longrightarrow\;
\text{closed-form pricing}
\;\longrightarrow\;
\text{Monte Carlo approximation}
\;\longrightarrow\;
\text{non-parametric ML approximation}
\;\longrightarrow\;
\text{out-of-sample evaluation}.
\]

---

## 1. Problem formulation

For a European option with current underlying price \(S\), strike \(K\), time to maturity \(T\), continuously compounded risk-free rate \(r\), and volatility \(\sigma\), the project studies the pricing map

\[
(S,K,T,r,\sigma,\text{option type})\mapsto V.
\]

For a call, the payoff at maturity is

\[
C_T=(S_T-K)^+=\max(S_T-K,0),
\]

while for a put

\[
P_T=(K-S_T)^+=\max(K-S_T,0).
\]

The computational models therefore attempt to approximate

\[
V=f(S,K,T,r,\sigma),
\]

with calls and puts treated separately.

A particularly useful dimensionless variable is **log-moneyness**

\[
m=\log\frac{S}{K},
\]

which measures relative rather than absolute moneyness. The XGBoost implementation explicitly includes \(\log(S/K)\) and \(\sqrt T\) as engineered features.

---

## 2. Black–Scholes theory

The theoretical starting point is the Black–Scholes model. Under the risk-neutral measure \(\mathbb Q\), the stock price follows geometric Brownian motion

\[
dS_t=rS_t\,dt+\sigma S_t\,dW_t^{\mathbb Q}.
\]

Applying Itô's lemma to \(\log S_t\) gives

\[
d\log S_t=\left(r-\frac12\sigma^2\right)dt+\sigma\,dW_t^{\mathbb Q},
\]

hence

\[
S_T=S_0\exp\left[\left(r-\frac12\sigma^2\right)T+\sigma\sqrt T Z\right],
\qquad Z\sim N(0,1).
\]

Risk-neutral valuation gives

\[
V_0=e^{-rT}\mathbb E^{\mathbb Q}[\Phi(S_T)],
\]

where \(\Phi\) is the payoff.

For a European call,

\[
C=S\Phi(d_1)-Ke^{-rT}\Phi(d_2),
\]

and for a European put,

\[
P=Ke^{-rT}\Phi(-d_2)-S\Phi(-d_1),
\]

where

\[
d_1=\frac{\log(S/K)+(r+\tfrac12\sigma^2)T}{\sigma\sqrt T},
\qquad d_2=d_1-\sigma\sqrt T.
\]

Here \(\Phi\) is the standard normal CDF.

This formula exposes the structure the ML models are trying to approximate. In particular, it explains why spot, strike, maturity, interest rates and volatility matter, and why the transformations \(\log(S/K)\) and \(\sqrt T\) are natural features.

The implementation in `models/BlackScholes.py` evaluates the closed-form solution and computes MAE, RMSE, MAPE and \(R^2\) against the test prices.

---

## 3. Monte Carlo as numerical integration

Monte Carlo pricing starts from the same risk-neutral expectation:

\[
V_0=e^{-rT}\mathbb E^{\mathbb Q}[\Phi(S_T)].
\]

Generate independent samples

\[
Z_1,\ldots,Z_N\overset{iid}{\sim}N(0,1)
\]

and simulate

\[
S_T^{(i)}=S\exp\left[\left(r-\frac12\sigma^2\right)T+\sigma\sqrt T Z_i\right].
\]

The expectation is approximated by

\[
\widehat V_N=e^{-rT}\frac1N\sum_{i=1}^N\Phi(S_T^{(i)}).
\]

By the law of large numbers,

\[
\widehat V_N\to V
\quad\text{as }N\to\infty,
\]

while the standard Monte Carlo error is typically of order

\[
O(N^{-1/2}).
\]

The implementation uses \(N=10,000\) simulations with a fixed seed. This gives a numerical approximation to the same mathematical quantity that Black–Scholes evaluates analytically.

See `models/MonteCarlo.py`.

---

## 4. Synthetic data: learning a known mathematical function

The synthetic dataset is deliberately generated from Black–Scholes.

Historical prices for MSFT, AAPL, GOOGL, AMZN, META, NVDA and TSLA are downloaded. Log returns are calculated as

\[
r_t^{(\mathrm{log})}=\log\frac{S_t}{S_{t-1}},
\]

and a 30-day annualised realised-volatility estimate is formed using

\[
\hat\sigma_t=\operatorname{sd}_{30}(r^{(\mathrm{log})})\sqrt{252}.
\]

For each observation the code varies maturity over

\[
\{7,30,90,240,365\}\text{ days}
\]

and strike over

\[
K\in\{0.80,0.90,0.95,1.00,1.05,1.10,1.20\}S.
\]

Both calls and puts are generated, with the Black–Scholes value used as the target.

This creates a clean mathematical experiment:

\[
\boxed{\text{Can ML rediscover a function whose generating mechanism is known?}}
\]

A model trained on synthetic data is therefore primarily approximating the Black–Scholes pricing surface, rather than predicting a genuinely unknown market process.

See `data/generate_datasets.py`.

---

## 5. Real option data and domain shift

The real dataset is obtained from option chains and contains quantities of the form

\[
(S,K,T,r,\sigma,\text{option type},V_{\mathrm{market}}).
\]

For these observations, \(\sigma\) is taken from the option chain's implied-volatility field and the observed option price is the target.

This changes the mathematical problem. Real markets are not generated exactly by constant-volatility geometric Brownian motion. The observed pricing surface can reflect

- stochastic and time-varying volatility;
- volatility smiles and skews;
- bid–ask spreads and liquidity;
- dividends and market conventions;
- model misspecification;
- noisy observed prices.

The project therefore studies the distribution shift

\[
\text{synthetic distribution}\longrightarrow\text{real market distribution}.
\]

A model that performs very well on synthetic data but deteriorates on real data may have learned the mathematical structure of Black–Scholes without learning the structure of actual market prices.

---

## 6. Machine-learning models

Four regression families are compared.

### Random Forest

A random forest approximates the pricing map through an ensemble of decision trees. Each tree partitions feature space and produces approximately piecewise-constant predictions:

\[
\hat f(x)\approx f(x),
\qquad x=(S,K,T,r,\sigma,\text{maturity days}).
\]

The implementation uses 500 trees with maximum depth 20.

See `models/Random Forest/RandomForest.py`.

### XGBoost

XGBoost constructs an additive tree model of the form

\[
\hat f(x)=\sum_{m=1}^{M}\eta h_m(x),
\]

where successive trees are fitted to reduce the loss. The implementation trains on

\[
\log(1+V)
\]

and converts predictions back using

\[
\hat V=\exp(\widehat{\log(1+V)})-1.
\]

It also adds the mathematically motivated features

\[
\log(S/K),\qquad \sqrt T.
\]

See `models/XGBoost/XGBoost.py`.

### CatBoost

CatBoost is a gradient-boosted decision-tree method used here as a flexible nonlinear approximation to

\[
(S,K,T,r,\sigma,\text{maturity days})\mapsto V.
\]

Models are trained separately by ticker and option type, using both synthetic and real training regimes.

See `models/CatBoost/CatBoost.py`.

### LSTM

An LSTM is a recurrent neural network whose hidden state is updated through nonlinear gated maps, schematically

\[
(h_t,c_t)=F(x_t,h_{t-1},c_{t-1}).
\]

The implementation standardises the six features and feeds them through two LSTM layers followed by dense layers.

**Important methodological caveat:** the current implementation does not feed a genuine temporal price sequence to the LSTM. The six pricing variables are reshaped as a sequence of length six. Thus this experiment tests nonlinear function approximation with an LSTM architecture; it does not establish that LSTMs exploit temporal market dynamics.

See `models/LSTM/LSTM.py`.

---

## 7. Classical mathematics vs learned structure

| Approach | Mathematical object | Main assumption |
|---|---|---|
| Black–Scholes | Closed-form risk-neutral expectation | GBM, constant \(\sigma\) |
| Monte Carlo | Same expectation by sampling | Same stochastic model; numerical approximation |
| Random Forest | Nonlinear pricing surface | Piecewise tree approximation |
| XGBoost | Additive tree approximation | Gradient-boosted nonlinear regression |
| CatBoost | Additive boosted-tree approximation | Flexible nonlinear regression |
| LSTM | Learned nonlinear map | Neural representation of feature interactions |

Black–Scholes and Monte Carlo have an explicit probabilistic interpretation. The ML models primarily learn the input-output relationship from examples.

The research question is therefore not merely **which algorithm wins**, but what mathematical structure is retained when an explicit pricing model is replaced by statistical approximation.

---

## 8. Evaluation mathematics

For observed prices \(y_i\) and predictions \(\hat y_i\), the project uses standard regression metrics.

### Mean Absolute Error

\[
\operatorname{MAE}=\frac1n\sum_{i=1}^{n}|y_i-\hat y_i|.
\]

### Root Mean Squared Error

\[
\operatorname{RMSE}=\sqrt{\frac1n\sum_{i=1}^{n}(y_i-\hat y_i)^2}.
\]

RMSE penalises large pricing errors more strongly than MAE.

### Coefficient of determination

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}.
\]

A high \(R^2\) means the predictions explain much of the variation in the evaluated sample; it is not by itself evidence of economic usefulness.

### Domain shift

The evaluation compares real- and synthetic-trained ML models through

\[
\Delta RMSE=RMSE_{real}-RMSE_{synthetic}.
\]

This quantifies how much performance changes when the data-generating regime changes.

See `evaluation/evaluation.py`.

---

## 9. Mathematical extensions

For a Year 2/3 mathematics project, the most valuable next steps are arguably stronger mathematical diagnostics rather than simply adding more algorithms.

### Derive the Black–Scholes PDE

A self-financing delta-hedged portfolio leads to

\[
\frac{\partial V}{\partial t}
+\frac12\sigma^2S^2\frac{\partial^2V}{\partial S^2}
+rS\frac{\partial V}{\partial S}
-rV=0.
\]

This connects stochastic calculus, PDEs and no-arbitrage pricing.

### Study the Greeks

Differentiate the pricing map to obtain

\[
\Delta,\ \Gamma,\ \Theta,\ \mathrm{Vega},\ \rho.
\]

A strong ML extension is to test whether a learned surface reproduces these derivatives.

### Enforce put–call parity

For European options,

\[
C-P=S-Ke^{-rT}.
\]

This is an exact structural constraint and can be used to test whether predictions are financially coherent.

### Test convexity and monotonicity

A valid European call surface should satisfy structural conditions such as monotonicity in \(S\) and convexity in \(K\) under standard assumptions. These constraints are more informative than a single global RMSE.

### Study the implied-volatility surface

Rather than assuming a constant \(\sigma\), investigate

\[
\sigma_{imp}=\sigma_{imp}(K,T).
\]

The resulting smile/skew structure gives a direct empirical measure of where Black–Scholes fails.

### Quantify Monte Carlo error

For the estimator

\[
\widehat V_N=e^{-rT}\frac1N\sum_{i=1}^{N}\Phi(S_T^{(i)}),
\]

its variance is

\[
\operatorname{Var}(\widehat V_N)
=\frac{e^{-2rT}}{N}\operatorname{Var}(\Phi(S_T)).
\]

This can be compared empirically with the predicted \(N^{-1/2}\) convergence rate.

---

## 10. Repository structure

```text
.
├── data/
│   ├── generate_datasets.py
│   ├── attempt2.py
│   ├── real_training_data.csv
│   ├── real_test_data.csv
│   └── synthetic_training_data.csv
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
│   └── generated predictions and evaluation tables
│
└── README.md
```

---

## 11. Running the project

Main Python dependencies:

```bash
pip install numpy pandas scipy scikit-learn yfinance catboost xgboost tensorflow joblib
```

Intended workflow:

```text
Generate datasets
      ↓
Black–Scholes benchmark
      ↓
Monte Carlo benchmark
      ↓
Random Forest / XGBoost / CatBoost / LSTM
      ↓
Evaluate on held-out real options
      ↓
Compare synthetic vs real performance
      ↓
Test mathematical constraints
```

The current scripts contain a local Windows `BASE_DIR` path. For use on another machine, change this path or refactor it to a project-relative path.

---

## 12. Interpretation

The central research questions are:

1. **Can ML recover known theory?** On synthetic data, the target is generated directly from Black–Scholes, so performance measures approximation of a known nonlinear pricing function.

2. **Can ML generalise beyond the theoretical model?** On real option data, deviations from synthetic performance expose model misspecification and distribution shift.

3. **Does low prediction error imply a good financial model?** Not necessarily. A model can have low RMSE while violating put–call parity, monotonicity, convexity or other structural properties.

For option pricing, **mathematical consistency is part of model quality**.

The natural direction of the project is therefore from prediction accuracy towards **structure-aware modelling**: learning from data while preserving the mathematical properties of a valid option-pricing surface.

---

## Background

The project draws on stochastic calculus, Itô's lemma, geometric Brownian motion, risk-neutral pricing, martingale measures, the Black–Scholes PDE, Monte Carlo estimation, regression, ensemble methods, neural networks and no-arbitrage theory.

This repository is a computational research project around those ideas, not a production trading or pricing system.
