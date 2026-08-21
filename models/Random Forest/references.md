Machine Learning:

A neural network is a flexible algorithm that can learn any continuous relationship between the value of a target variable (the output) and the values of features (the inputs) when a huge volume of data is available. We will replicate the Monte Carlo calculations as accurately as possible. 

https://arxiv.org/html/2510.01446v1#S5
Neural Networks: Neural networks are particularly well-suited for capturing complex, non-linear relationships in financial data. In this study, a feedforward multi-layer perceptron (MLP) was implemented using scikit-learn’s MLPRegressor. Rather than fixing the architecture manually, the model configuration, including the number of hidden layers, neurons per layer, regularization strength, and learning rate was selected through randomized hyperparameter search.  
Random Forests: Random Forests is a highly popular, versatile supervised machine learning algorithm used for both classification (predictive categories) and regression (predictive numbers). It operates by building a "forest" of multiple independent decision trees and combining their outputs for a final, robust result 
CatBoost: CatBoost (Categorical Boosting) is a high-performance open-source machine learning library. It builds decision trees using gradient boosting and natively processes categorical and text data without requiring extensive preprocessing (like one-hot encoding), making it ideal for real-world datasets and fast model development 
https://cs230.stanford.edu/projects_fall_2019/reports/26260984.pdf
LSTM
MLP1
MLP2
https://github.com/saeedbidi/option_pricing
XGboost 
Random Forest
https://arxiv.org/pdf/2307.07657

https://www.sciencedirect.com/science/article/pii/S0957417420306187
https://www.tidy-finance.org/r/option-pricing-via-machine-learning.html
Use yfinance but treat as european options, 
price: 
Apple Inc. (AAPL)
Microsoft Corporation (MSFT)
NVIDIA Corporation (NVDA)
Tesla, Inc. (TSLA)
Amazon.com, Inc. (AMZN) 
using : 
Black-Scholes
Monte Carlo
LSTM
Random Forest
XGBoost
CatBoost and compare
Lstm: the training labels are synthetic Black-Scholes prices while the test labels are real market prices. That mismatch explains most of the remaining large errors, especially for TSLA and GOOGL long-dated contracts. 
New training set:
synthetic_training_data.csv
    Historical stock prices
    +
    Historical realized volatility
    +
    Black-Scholes pricing

real_training_data.csv
    Actual market option prices

real_test_data.csv
    Actual market option prices
    Held out 20%



What is our goal? 

To build a machine learning model that takes in current market data like stock price, strike price and time left until expiration and outputs a theoretical value for that option
By using different models on our training data, we evaluate which model is the best and most accurate and determining a theoretical price

What is the point?

We talk about the black scholes equation quite a lot throughout our project so why do we need machine learning?
Although using the Black-Scholes equation would be much easier, the equation is built with many assumptions like constant volatility and does not consider ‘human behaviour’ such as the change in demand when we are anticipating the released of earning reports etc 
When training machine learning models it is able to take these factors into account when making predictions making it more accurate








Evaluation of each model

LSTM (Long Short-Term Memory) Model 


Standard neural networks have short-term memory. This is a problem because the model will not be able to recall previous stock prices
The LSTM model has a working memory so it overcomes this issue 
It works by using 3 gates 

The forget gate: it takes in new data and decides what old data in the long-term memory should be deleted 
The input gate: it decides what data is ‘important enough’ to be written into long-term memory
The output gate: combines data in the long term memory with today’s data which helps it decide its final prediction 

In order to use LSTM, the data points need to be in sequential chronological order with even time intervals. Current market data from Yahoo Finance is not given in even time intervals, making this model unsuitable 

The Random Forest Model 

The Random Forest model consists of  many decision trees, each decision tree acts like a flow chart where each branch represent a yes or no question (don't forget to add a visual picture of a decision tree)
When training this model all you have to decide is how many trees you want to use
In general, using more trees increases accuracy but it also increases computational cost so this has to be balanced 
The computer then takes the training data and decides the unique chain of questions for each tree
When given current data the model can predict a value (like option price)

Using Synthetic data: 
Pros: 



Using real option data: 

Pros: 
The model will be able to pick up on real world phenomena such as human behaviour that the black scholes equation cannot capture allowing us to predict option prices more accurately 
We do not have to worry about the time interval problem we had with LSTM



How are we going to compare our models? 

3 metrics: 
Mean Absolute Error (MAE): the average difference between the true price and the model’s prediction (in dollars)
R2score: tells you the percentage of the market’s behaviour the model ‘understands’
Root Mean Square Error (RMSE)








From the graphs there are 3 main things we can notice: 

When we train out models using real data our models have lower errors and higher R^2 score meaning they predict options prices more accurately. Since our synthetic data is generated using the black-scholes formula, our model is trained to be a ‘black-scholes calculator’, meaning that it would reintroduce the assumptions from the black scholes formula 

The LSTM model performed significantly poorer compared to the other models, the main reason for this model is designed to be trained with data in chronological data over even time intervals which we could not easily access 

The performance of the 3 models CatBoost, XGBoost and Random Forest are quite similar and they all outperform the black-scholes formula and montecarlo, so we consider its computational cost

Computational cost 
XGBoost (least expensive) - the trees don’t have to be symmetric 
Random Forest 
CatBoost 
LSTM (most expensive)

(we could perhaps also consider how hard they were to code??)

1 https://arxiv.org/html/2510.01446v1#S5 2 https://cs230.stanford.edu/projects_fall_2019/reports/26260984.pdf 3 https://arxiv.org/pdf/2307.07657 4 https://www.sciencedirect.com/science/article/pii/S0957417420306187 5 https://www.tidy-finance.org/r/option-pricing-via-machine-learning.html 6 https://github.com/saeedbidi/option_pricing
