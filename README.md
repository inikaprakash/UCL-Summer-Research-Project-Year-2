# UCL Summer Research Project — Year 2

## Option Pricing through Probability, Stochastic Calculus, Numerical Methods and Machine Learning

I use this project to study option pricing by starting with the mathematics and then asking how different computational methods reproduce it.

The central object is the pricing function

\[
V=V(S,K,T,r,\sigma),
\]

where \(S\) is the underlying price, \(K\) the strike, \(T\) the time to maturity, \(r\) the continuously compounded risk-free rate and \(\sigma\) the volatility.

For a European call and put, the terminal payoffs are

\[
C_T=(S_T-K)^+,
\qquad
P_T=(K-S_T)^+.
\]

The computational question is therefore a function-approximation problem, but the function being approximated has a rich mathematical structure. I want to understand where that structure comes from before comparing algorithms that attempt to learn it.

The project follows the chain

\[
\boxed{
\text{SDE}
\rightarrow
\text{Itô calculus}
\rightarrow
\text{risk-neutral valuation}
\rightarrow
\text{Black--Scholes PDE/formula}
\rightarrow
\text{Monte Carlo}
\rightarrow
\text{machine-learning approximation}
}
\]

---

## 1. The pricing problem

For a European option, I can regard pricing as a map

\[
f:(S,K,T,r,\sigma)\mapsto V.
\]

The important point is that this is not an arbitrary multivariable function. Under the Black--Scholes assumptions it is constrained by probability, stochastic calculus and no-arbitrage.

A particularly natural coordinate is **log-moneyness**

\[
m=\log\frac{S}{K}.
\]

The quantity \(\sigma\sqrt T\) also appears naturally because Brownian motion satisfies the scaling relation

\[
W_T\overset{d}=\sqrt T Z,
\qquad Z\sim N(0,1).
\]

This is why transformations such as \(\log(S/K)\) and \(\sqrt T\) are mathematically motivated features rather than arbitrary preprocessing.

---

## 2. Geometric Brownian motion

The starting point is the stochastic differential equation

\[
dS_t=\mu S_t\,dt+\sigma S_t\,dW_t.
\]

Under the physical measure, \(\mu\) represents the expected rate of return. For pricing, I work under the risk-neutral measure \(\mathbb Q\), where the drift becomes the risk-free rate:

\[
dS_t=rS_t\,dt+\sigma S_t\,dW_t^{\mathbb Q}.
\]

This SDE is the mathematical model underlying the code in `models/BlackScholes.py` and `models/MonteCarlo.py`.

### Applying Itô's lemma

Set

\[
X_t=\log S_t.
\]

Itô's lemma gives

\[
dX_t
=\frac{1}{S_t}dS_t
-\frac{1}{2S_t^2}(dS_t)^2.
\]

Since

\[
(dW_t)^2=dt,
\]

we have

\[
(dS_t)^2=\sigma^2S_t^2dt.
\]

Therefore

\[
d\log S_t
=\left(r-\frac12\sigma^2\right)dt
+\sigma dW_t^{\mathbb Q}.
\]

Integrating from \(0\) to \(T\),

\[
\log S_T
=
\log S_0
+\left(r-\frac12\sigma^2\right)T
+\sigma W_T^{\mathbb Q}.
\]

Since

\[
W_T^{\mathbb Q}=\sqrt T Z,
\qquad Z\sim N(0,1),
\]

I obtain

\[
S_T
=S_0\exp\left[
\left(r-\frac12\sigma^2\right)T
+\sigma\sqrt T Z
\right].
\]

This is the reason the terminal stock price is lognormally distributed in the Black--Scholes model.

---

## 3. Risk-neutral valuation

Once the risk-neutral dynamics are known, the price of a derivative with payoff \(\Phi(S_T)\) is

\[
V_0=e^{-rT}\mathbb E^{\mathbb Q}[\Phi(S_T)].
\]

For a European call,

\[
C_0=e^{-rT}\mathbb E^{\mathbb Q}[(S_T-K)^+],
\]

and for a European put,

\[
P_0=e^{-rT}\mathbb E^{\mathbb Q}[(K-S_T)^+].
\]

This expectation is the central mathematical object behind the project.

There are then several ways of approaching it:

- evaluate it analytically;
- approximate it numerically using Monte Carlo;
- learn the resulting input-output map from data.

The first two approaches retain an explicit probabilistic interpretation. The machine-learning models instead try to approximate the resulting pricing surface statistically.

---

## 4. Deriving the Black--Scholes formula

Evaluating the risk-neutral expectation gives the Black--Scholes formula

\[
C=S\Phi(d_1)-Ke^{-rT}\Phi(d_2),
\]

where

\[
d_1
=
\frac{\log(S/K)+(r+\frac12\sigma^2)T}
{\sigma\sqrt T},
\]

and

\[
d_2=d_1-\sigma\sqrt T.
\]

For a put,

\[
P=Ke^{-rT}\Phi(-d_2)-S\Phi(-d_1).
\]

Here \(\Phi\) is the standard normal CDF.

The appearance of the normal CDF is a direct consequence of the normal random variable in the expression for \(\log S_T\). The pricing formula is therefore ultimately an integration problem involving the lognormal distribution of \(S_T\).

The code in `models/BlackScholes.py` implements this closed-form expression and uses it as the main theoretical benchmark.

---

## 5. The Black--Scholes PDE

The same price can be obtained from a PDE rather than directly from the risk-neutral expectation.

Suppose the option value is \(V(S,t)\). Applying Itô's lemma,

\[
dV
=
\left(
V_t+\mu SV_S+\frac12\sigma^2S^2V_{SS}
\right)dt
+\sigma SV_SdW_t.
\]

Construct the delta-hedged portfolio

\[
\Pi=V-\Delta S.
\]

Choosing

\[
\Delta=V_S
\]

eliminates the stochastic term. The resulting portfolio is locally riskless, so no-arbitrage requires it to earn the risk-free rate.

This gives the Black--Scholes PDE

\[
\boxed{
V_t+\frac12\sigma^2S^2V_{SS}+rSV_S-rV=0
}
\]

with terminal condition determined by the payoff.

This gives one of the central mathematical connections in the project:

\[
\boxed{
\text{stochastic differential equations}
\leftrightarrow
\text{Itô calculus}
\leftrightarrow
\text{no-arbitrage}
\leftrightarrow
\text{PDEs}
}
\]

The risk-neutral expectation and the PDE are two mathematical descriptions of the same pricing problem.

---

## 6. Monte Carlo as numerical probability

Rather than evaluate

\[
V_0=e^{-rT}\mathbb E^{\mathbb Q}[\Phi(S_T)]
\]

analytically, Monte Carlo approximates the expectation by sampling.

I generate

\[
Z_1,\ldots,Z_N\overset{iid}{\sim}N(0,1)
\]

and calculate

\[
S_T^{(i)}
=S\exp\left[
\left(r-\frac12\sigma^2\right)T
+\sigma\sqrt T Z_i
\right].
\]

The estimator is

\[
\widehat V_N
=
\frac{e^{-rT}}{N}
\sum_{i=1}^N\Phi(S_T^{(i)}).
\]

The law of large numbers gives

\[
\widehat V_N\longrightarrow V
\qquad\text{as }N\to\infty.
\]

More precisely,

\[
\operatorname{Var}(\widehat V_N)
=
\frac{e^{-2rT}}{N}
\operatorname{Var}(\Phi(S_T)),
\]

so the standard error scales like

\[
O(N^{-1/2}).
\]

The implementation uses 10,000 simulations with a fixed seed.

The important mathematical point is that Monte Carlo is **not a different pricing model** here. It is a numerical approximation to the same risk-neutral expectation that leads to the Black--Scholes price.

See `models/MonteCarlo.py`.

---

## 7. Synthetic data: learning a known function

The synthetic experiment asks a very controlled question:

> Can a machine-learning model learn a pricing function whose generating mathematical model is already known?

Historical prices are used to construct log returns

\[
r_t=\log\frac{S_t}{S_{t-1}},
\]

and a 30-day annualised realised-volatility estimate

\[
\widehat\sigma_t
=\operatorname{sd}(r_{t-29},\ldots,r_t)\sqrt{252}.
\]

For each observation, the code varies the maturity and strike. The strike grid is expressed relative to spot, so the resulting options cover different levels of moneyness.

The Black--Scholes formula is then used to generate the target price:

\[
X_i=(S_i,K_i,T_i,r_i,\sigma_i),
\qquad
y_i=f_{BS}(X_i).
\]

The machine-learning problem is therefore

\[
\boxed{\widehat f\approx f_{BS}}.
\]

This is essentially a nonlinear function-approximation experiment. Since the target is generated by Black--Scholes, there is a known mathematical function against which the learned approximation can be compared.

See `data/generate_datasets.py`.

---

## 8. Real option data

The real-data experiment is mathematically different.

The observed data contain quantities such as

\[
(S,K,T,r,\sigma_{imp},\text{option type},V_{market}).
\]

In real markets, the volatility required to reproduce observed prices is generally not a single constant. Instead, it can depend on strike and maturity:

\[
\sigma_{imp}=\sigma_{imp}(K,T).
\]

This produces the implied-volatility surface and phenomena such as volatility smiles and skews.

Consequently, a real market pricing surface need not satisfy the constant-volatility Black--Scholes model exactly.

Other effects can also appear, including dividends, liquidity, bid--ask spreads, stochastic volatility, jumps and noise in observed prices.

This makes the synthetic-to-real comparison particularly interesting. If an ML model performs very well on synthetic data but deteriorates on real data, the difference is not necessarily an ML failure. It may indicate that the real data contain structure that is absent from the Black--Scholes model.

---

## 9. Machine learning as approximation theory

I interpret each ML model as constructing an approximation

\[
\widehat f(x)\approx f(x).
\]

The interesting mathematical question is what kind of approximation each model produces.

### Random Forest

A decision tree partitions feature space into regions

\[
\mathcal X=R_1\cup\cdots\cup R_m
\]

and produces an approximately constant prediction on each region:

\[
\widehat f(x)=c_j,
\qquad x\in R_j.
\]

A random forest averages many such trees,

\[
\widehat f(x)=\frac1B\sum_{b=1}^{B}f_b(x).
\]

The implementation uses 500 trees with maximum depth 20.

See `models/Random Forest/RandomForest.py`.

### XGBoost

XGBoost builds an additive approximation

\[
\widehat f_M(x)=\sum_{m=1}^{M}\eta h_m(x),
\]

where successive trees are chosen to improve the current approximation.

I find it useful to interpret this as iterative optimisation in a function space: rather than selecting one complicated function immediately, the model repeatedly adds corrections to the current approximation.

The implementation also introduces

\[
\log(S/K),\qquad \sqrt T
\]

as features and trains on

\[
y'=\log(1+V),
\]

before transforming predictions back with

\[
\widehat V=e^{\widehat y'}-1.
\]

See `models/XGBoost/XGBoost.py`.

### CatBoost

CatBoost is another gradient-boosted tree method. Here it is used as a flexible nonlinear approximation to the pricing surface.

The underlying problem remains

\[
(S,K,T,r,\sigma,\text{maturity})\mapsto V.
\]

See `models/CatBoost/CatBoost.py`.

### LSTM

An LSTM uses a recurrent state and nonlinear gates. Schematically,

\[
(h_t,c_t)=F(x_t,h_{t-1},c_{t-1}).
\]

The implementation standardises the pricing variables and passes them through two LSTM layers followed by dense layers.

There is an important mathematical caveat here: the six pricing variables are reshaped as a sequence of length six, but they are not six consecutive time observations. Therefore I interpret this experiment as nonlinear function approximation using an LSTM architecture, rather than as evidence that the model has learned temporal market dynamics.

See `models/LSTM/LSTM.py`.

---

## 10. Why the feature transformations matter

The structure of the Black--Scholes formula itself suggests useful coordinates.

Recall

\[
d_1
=
\frac{\log(S/K)+(r+\frac12\sigma^2)T}
{\sigma\sqrt T}.
\]

This immediately highlights

\[
\log(S/K)
\]

and

\[
\sigma\sqrt T.
\]

The first measures relative moneyness. The second is the natural scale of stochastic variation over the time horizon.

Thus feature engineering can be viewed as an attempt to expose the geometry already present in the mathematical model.

This is particularly important for the ML models: if the theoretical pricing surface has natural coordinates, giving a model access to those coordinates can make the approximation problem substantially more natural.

---

## 11. Evaluation: accuracy is only the first test

For observed prices \(y_i\) and predictions \(\widehat y_i\), I use standard regression metrics.

### Mean Absolute Error

\[
MAE=\frac1n\sum_{i=1}^n|y_i-\widehat y_i|.
\]

### Root Mean Squared Error

\[
RMSE=\sqrt{\frac1n\sum_{i=1}^n(y_i-\widehat y_i)^2}.
\]

RMSE penalises large errors more heavily than MAE.

### Coefficient of determination

\[
R^2
=1-
\frac{\sum_i(y_i-\widehat y_i)^2}
{\sum_i(y_i-\bar y)^2}.
\]

These metrics measure pointwise predictive accuracy, but they do not tell me whether a learned pricing surface has the correct mathematical structure.

That distinction matters enormously in option pricing.

---

## 12. Mathematical consistency of the learned surface

A good approximation should not merely produce numbers close to observed prices. It should ideally preserve structural properties implied by no-arbitrage and the payoff.

### Put--call parity

For European options,

\[
C-P=S-Ke^{-rT}.
\]

This is an exact relationship.

### Monotonicity

For a European call,

\[
\frac{\partial C}{\partial S}\geq0,
\qquad
\frac{\partial C}{\partial K}\leq0.
\]

Increasing the underlying should not decrease the value of a call, while increasing the strike should not increase it.

### Convexity

Under the standard assumptions,

\[
\frac{\partial^2C}{\partial S^2}\geq0.
\]

This quantity is Gamma.

These conditions suggest a much stronger evaluation framework for ML models. Instead of only asking

\[
|V-\widehat V|\text{ small?}
\]

I can ask whether

\[
\widehat V
\]

has approximately the same derivatives, monotonicity and convexity as the theoretical pricing surface.

---

## 13. Greeks and local structure

The Greeks are derivatives of the pricing function:

\[
\Delta=\frac{\partial V}{\partial S},
\qquad
\Gamma=\frac{\partial^2V}{\partial S^2},
\]

and

\[
\Theta=\frac{\partial V}{\partial t},
\qquad
\mathrm{Vega}=\frac{\partial V}{\partial\sigma},
\qquad
\rho=\frac{\partial V}{\partial r}.
\]

This gives a more demanding test of a learned approximation.

A model could have a very small price RMSE while producing poor local derivatives. In that case it has approximated the values without accurately recovering the local geometry of the pricing surface.

A natural extension is therefore to compare

\[
\frac{\partial \widehat V}{\partial S}
\quad\text{with}\quad
\frac{\partial V_{BS}}{\partial S}
\]

and similarly for the other Greeks.

For a neural network this can be approached through automatic differentiation; for tree models, finite differences or local perturbations can be used.

---

## 14. Synthetic data versus real data

The two datasets correspond to two different mathematical settings.

For synthetic data,

\[
y=f_{BS}(x).
\]

The generating function is known.

For real market data, a useful schematic model is

\[
y=f_{market}(x)+\varepsilon,
\]

where both the underlying pricing mechanism and the noise are unknown.

Therefore the comparison allows me to distinguish, at least conceptually, between

\[
\boxed{\text{function-approximation error}}
\]

and effects associated with

\[
\boxed{\text{model misspecification, market structure and noise}.}
\]

This is why the synthetic experiment is important: it gives a controlled mathematical benchmark before moving to the much less structured real-data problem.

---

## 15. Further mathematical directions

### Feynman--Kac

A natural theoretical extension is the Feynman--Kac theorem, which gives a precise connection between the Black--Scholes PDE and the risk-neutral expectation.

This unifies the probabilistic and analytical viewpoints:

\[
\text{PDE solution}
\longleftrightarrow
\text{conditional expectation under }\mathbb Q.
\]

### Implied volatility

Instead of assuming a constant volatility, define \(\sigma_{imp}\) implicitly by

\[
C_{market}
=
C_{BS}(S,K,T,r,\sigma_{imp}).
\]

This produces the surface

\[
(K,T)\mapsto\sigma_{imp}(K,T).
\]

Studying its smile and skew gives a direct way of investigating where the constant-volatility model fails.

### Monte Carlo convergence

I can experimentally vary \(N\) and test whether the error behaves like

\[
N^{-1/2}.
\]

This connects the empirical implementation directly to the theoretical variance calculation.

### Structure-preserving machine learning

A particularly interesting extension is to include mathematical constraints directly in the learning objective.

Instead of minimising only

\[
\sum_i(V_i-\widehat V_i)^2,
\]

one could include penalties for violating put--call parity, monotonicity or convexity.

The question then becomes:

> Can I learn the pricing surface from data while preserving the mathematics that makes it a valid pricing surface?

That is a much more interesting problem than simply minimising prediction error.

---

## 16. Repository structure

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

## 17. Running the project

The main dependencies are:

```bash
pip install numpy pandas scipy scikit-learn yfinance catboost xgboost tensorflow joblib
```

The intended workflow is

```text
Generate data
      ↓
Black--Scholes analytical benchmark
      ↓
Monte Carlo numerical benchmark
      ↓
Random Forest / XGBoost / CatBoost / LSTM
      ↓
Out-of-sample evaluation
      ↓
Synthetic vs real comparison
      ↓
Mathematical consistency checks
```

The current scripts contain a local Windows `BASE_DIR`, so this path needs to be changed when running the project on another machine.

---

## 18. Perspective

The main idea behind the project is that option pricing is fundamentally a mathematical problem before it is a machine-learning problem.

The Black--Scholes price can be written as

\[
V_0=e^{-rT}\mathbb E^{\mathbb Q}[\Phi(S_T)],
\]

but this single expression connects several areas of mathematics:

\[
\boxed{
\text{probability}
\;+
\text{stochastic processes}
\;+
\text{Itô calculus}
\;+
\text{PDEs}
\;+
\text{numerical analysis}
\;+
\text{function approximation}
}
\]

Black--Scholes evaluates the pricing problem analytically. Monte Carlo approximates the same expectation through sampling. The machine-learning models attempt to approximate the resulting pricing surface from examples.

The interesting question is therefore not simply which model gives the smallest RMSE. It is whether the approximation captures the **mathematical structure** of the object being approximated.

The code is my way of experimenting with those ideas: starting from an explicit mathematical model, constructing numerical approximations to it, and then asking how much of that structure can be recovered when the explicit formula is replaced by a learned function.