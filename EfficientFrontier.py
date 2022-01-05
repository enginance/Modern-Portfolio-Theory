##############################################################################################################
# ███████╗███████╗███████╗██╗░█████╗░██╗███████╗███╗░░██╗████████╗
# ██╔════╝██╔════╝██╔════╝██║██╔══██╗██║██╔════╝████╗░██║╚══██╔══╝
# █████╗░░█████╗░░█████╗░░██║██║░░╚═╝██║█████╗░░██╔██╗██║░░░██║░░░
# ██╔══╝░░██╔══╝░░██╔══╝░░██║██║░░██╗██║██╔══╝░░██║╚████║░░░██║░░░
# ███████╗██║░░░░░██║░░░░░██║╚█████╔╝██║███████╗██║░╚███║░░░██║░░░
# ╚══════╝╚═╝░░░░░╚═╝░░░░░╚═╝░╚════╝░╚═╝╚══════╝╚═╝░░╚══╝░░░╚═╝░░░

# ███████╗██████╗░░█████╗░███╗░░██╗████████╗██╗███████╗██████╗░
# ██╔════╝██╔══██╗██╔══██╗████╗░██║╚══██╔══╝██║██╔════╝██╔══██╗
# █████╗░░██████╔╝██║░░██║██╔██╗██║░░░██║░░░██║█████╗░░██████╔╝
# ██╔══╝░░██╔══██╗██║░░██║██║╚████║░░░██║░░░██║██╔══╝░░██╔══██╗
# ██║░░░░░██║░░██║╚█████╔╝██║░╚███║░░░██║░░░██║███████╗██║░░██║
# ╚═╝░░░░░╚═╝░░╚═╝░╚════╝░╚═╝░░╚══╝░░░╚═╝░░░╚═╝╚══════╝╚═╝░░╚═╝
##############################################################################################################

##############################################################################################################
# Portfolio Optimization with Python using Efficient Frontier with Practical Examples
##############################################################################################################

# Portfolio optimization is the process of creating a portfolio of assets, for which your investment has the maximum return and minimum risk.
# Modern Portfolio Theory (MPT), or also known as mean-variance analysis is a mathematical process which allows the user to maximize returns for a given risk level.
# It was formulated by H. Markowitz and while it is not the only optimization technique known, it is the most widely used.

# Efficient frontier is a graph with ‘returns’ on the Y-axis and ‘volatility’ on the X-axis. 
# It shows the set of optimal portfolios that offer the highest expected return for a given risk level or the lowest risk for a given level of expected return.


# In this example, we are considering a portfolio made up of stocks from just 2 companies, Tesla and Facebook.


##############################################################################################################
# Step 1: Pull the stock price data
##############################################################################################################
# Load Packages
import numpy as np
import pandas as pd
from pandas_datareader import data
import matplotlib.pyplot as plt

# The following is for JupiterNotebook
# %matplotlib inline


# Read Data
test = data.DataReader(['TSLA', 'FB'], 'yahoo', start='2018/01/01', end='2019/12/31')
print(test.head())

# The ‘Adj Close’ column. This colum gives us the closing price of company’s stock on the given day.
# Closing price
test = test['Adj Close']
print(test.head())


##############################################################################################################
# Step 2: Calculate percentage change in stock prices
##############################################################################################################
# Next, we calculate the percentage change in stock prices of tesla everyday. You will notice that that we take the log of percentage change.
# Why take log? The reason for this is that log of the returns is time additive. Then, log(r13) = log(r12) + log(r23)

# It is common practice in portfolio optimization to take log of returns for calculations of covariance and correlation.

# Log of percentage change for Tesla
tesla = test['TSLA'].pct_change().apply(lambda x: np.log(1+x))
print(tesla.head())

##############################################################################################################
# Variance
##############################################################################################################

# Variance
var_tesla = tesla.var()
print(var_tesla)
# 0.0011483734269334596

# Log of Percentage change for Facebook
fb = test['FB'].pct_change().apply(lambda x: np.log(1+x))
print(fb.head())

# Variance
var_fb = fb.var()
print(var_fb)
#> .00045697258417022536


# But volatility for the annual standard deviation. 
# What we get from square root of variance is the daily standard deviation. 
# To convert it to annual standard deviation we multiply the variance by 250. 
# 250 is used because there are 250 trading days in a year.

# In US the trading days in a year are 250 (this may be different in UK)
trading_days_year = 250

##############################################################################################################
# Volatility
##############################################################################################################

# Volatility
tesla_vol = np.sqrt(var_tesla * trading_days_year)
fb_vol = np.sqrt(var_fb * trading_days_year)
print(tesla_vol, fb_vol)
#> .5358109337568289  .33799873674698305

# Volatility of both stocks
test.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(trading_days_year)).plot(kind='bar')
plt.show()

##############################################################################################################
# Covariance
##############################################################################################################

# Covariance measures the directional relationship between the returns on two assets. 
# A positive covariance means that returns of the two assets move together while a negative covariance means they move inversely. 
# Risk and volatility can be reduced in a portfolio by pairing assets that have a negative covariance.

# Log of Percentage change
test1 = test.pct_change().apply(lambda x: np.log(1+x))
print(test1.head())

# Covariance
test1['TSLA'].cov(test1['FB'])
#> .00018261623156030972
print(test1['TSLA'].cov(test1['FB']))


##############################################################################################################
# Correlation
##############################################################################################################
# Correlation, in the finance and investment industries, is a statistic that measures the degree to which two securities move in relation to each other. 
# Correlations are used in advanced portfolio management, computed as the correlation coefficient, which has a value that must fall between -1.0 and +1.0.

# A correlation of -1 means negative relation, i.e, if correlation between Asset A and Asset B is -1, if Asset A increases, Asset B decreases. 
# A correlation of +1 means positive relation, i.e, if correlation between Asset A and Asset B is 1, if Asset A increases, Asset B increases.

# Correlation
test1['TSLA'].corr(test1['FB'])
#> .2520883272466132
print(test1['TSLA'].corr(test1['FB']))


##############################################################################################################
# Expected returns
##############################################################################################################
# Expected returns of an asset are simply the mean of percentage change in its stock prices. 
# So, the value of expected return we obtain here are daily expected returns. 
# For an yearly expected return value, you will need to resample the data year-wise, as you will see further.

# For expected returns, you need to define weights for the assets choosen.
# In simpler terms, this means you need to decide what percentage of your total money to you want to hold in each company’s stock.
# Usually this decision is done by using the optimization techniques we will discuss later but for now we will consider random weights for Tesla and Facebook.

# First, let’s compute the log of percentage change.
test2 = test.pct_change().apply(lambda x: np.log(1+x))
print(test2.head())

# Let’s define an array of random weights for the purpose of calculation. 
# These weights will represent the percentage allocation of investments between these two stocks. 
# They must add up to 1.

# Define weights for allocation
w = [0.2, 0.8]
# Expected returns of the single assets
e_r_ind = test2.mean()
print(e_r_ind)

# Total expected return
e_r = (e_r_ind*w).sum()
print(e_r)
#> .0003027691524101118




##############################################################################################################
# Building an optimal risky portfolio
##############################################################################################################
# We will be using stocks from 4 companies, namely, Apple, Nike, Google and Amazon for a period of 5 years. 
# You will learn to calculate the weights of assets for each one. 
# Then, we will calculate the expected returns, minimum variance portfolio, optimal risky portfolio and efficient frontier. 
# You will also learn a new term called Sharpe Ratio.

# Import data
df = data.DataReader(['AAPL', 'NKE', 'GOOGL', 'AMZN'], 'yahoo', start='2015/01/01', end='2019/12/31')
print(df.head())

# Closing price
df = df['Adj Close']
print(df.head())


##############################################################################################################
# Covariance and Correlation matrix
##############################################################################################################
# The first step is to obtain a covariance and correlation matrix to understand how different assets behave with respect to each other. 
# When we had a 2 asset portfolio, we directly plugged in the names of the assets into .cov() and .corr() functions. 
# In this case, we will need a matrix for better visualisation. 
# This is also achieved by using the same 2 functions on our dataframe df.

# Log of percentage change and covariance matrix
cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
print(cov_matrix)

# Correlation matrix
corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
print(corr_matrix)
# As you can see, an asset always has a perfectly positive correlation of 1 with itself.


##############################################################################################################
# Portfolio variance
##############################################################################################################
# The formula for calculating portfolio variance differs from the usual formula of variance. It looks like this: 
# $$\sigma^2(Rp) = \sum{i=1}^{n} \sum_{j=1}^{n} w_i w_j COV(R_i, R_j) $$

# The simplest way to do this complex calculation is defining a list of weights and multiplying this list horizontally and vertically with our covariance matrix. 
# For this purpose, let’s define a random list of weights for all 4 assets. 
# Remember that sum of weights should always be 1.

# Randomly weighted portfolio's variance
w = {'AAPL': 0.1, 'NKE': 0.2, 'GOOGL': 0.5, 'AMZN': 0.2}
port_var = cov_matrix.mul(w, axis=0).mul(w, axis=1).sum().sum()
print(port_var)
# 0.00016069523003596587

# Thus we have found the portfolio variance. But for truly optimizing the portfolio, we cant plug in random weights. 
# We will need to calculate it according to what gives us maximum expected returns.

##############################################################################################################
# Portfolio expected returns
##############################################################################################################
# The mean of returns (given by change in prices of asset stock prices) give us the expected returns of that asset. 
# The sum of all individual expected returns further multiplied by the weight of assets give us expected return for the portfolio.

# Note that we use the resample() function to get yearly returns. The argument to function, ‘Y’, denotes yearly. 
# If we dont perform resampling, we will get daily returns, like you saw earlier in the ‘Fundamental Terms’ section.

# Yearly returns for individual companies
ind_er = df.resample('Y').last().pct_change().mean()
print(ind_er)

# Portfolio returns
w = [0.1, 0.2, 0.5, 0.2]
port_er = (w*ind_er).sum()
print(port_er)
# 0.27071990038443955


##############################################################################################################
# Plotting the efficient frontier
##############################################################################################################
# Efficient frontier is a graph with ‘returns’ on the Y-axis and ‘volatility’ on the X-axis. 
# It shows us the maximum return we can get for a set level of volatility, or conversely, the volatility that we need to accept for certain level of returns.

# But first, lets take a look at the volatiltilty and returns of individual assets for a better understanding.

# In US the trading days in a year are 250 (this may be different in UK)
# trading_days_year = 250

# Volatility is given by the annual standard deviation. We multiply by 250 because there are 250 trading days/year.
ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(trading_days_year))
print(ann_sd)

# Creating a table for visualising returns and volatility of assets
assets = pd.concat([ind_er, ann_sd], axis=1) 
assets.columns = ['Returns', 'Volatility']
print(assets)

# Next, to plot the graph of efficient frontier, we need run a loop. 
# In each iteration, the loop considers different weights for assets and calculates the return and volatility of that particular portfolio combination.
# We run this loop a 1000 times. To get random numbers for weights, we use the np.random.random() function. 
# But remember that the sum of weights must be 1, so we divide those weights by their cumulative sum.

p_ret = [] # Define an empty array for portfolio returns
p_vol = [] # Define an empty array for portfolio volatility
p_weights = [] # Define an empty array for asset weights

num_assets = len(df.columns)
num_portfolios = 10000

for portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights = weights/np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum() # Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    ann_sd = sd*np.sqrt(trading_days_year) # Multiply by 250. Annual standard deviation = volatility
    p_vol.append(ann_sd)


data = {'Returns':p_ret, 'Volatility':p_vol}

for counter, symbol in enumerate(df.columns.tolist()):
    #print(counter, symbol)
    data[symbol+' weight'] = [w[counter] for w in p_weights]

portfolios  = pd.DataFrame(data)
print(portfolios.head()) # Dataframe of the 10000 portfolios created

# You can see that there are a number of portfolios with different weights, returns and volatility. 
# Plotting the returns and volatility from this dataframe will give us the efficient frontier for our portfolio.

# Plot efficient frontier
portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])
plt.show()

# Each point on the line (left edge) represents an optimal portfolio of stocks that maximises the returns for any given level of risk.
# On this graph, you can also see the combination of weights that will give you all possible combinations: 
#   Minimum volatility (left most point)
#   Maximum returns (top most point)

min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
# idxmin() gives us the minimum value in the column specified.                               
print(min_vol_port)
# The values at the minimum index found are as follows
print(portfolios.iloc[min_vol_port])

##############################################################################################################
# plotting the minimum volatility portfolio
##############################################################################################################
plt.subplots(figsize=[10,10])
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
plt.show()

# The red star denotes the most efficient portfolio with minimum volatility.


##############################################################################################################
# Sharpe ratio
##############################################################################################################

# The question arises that how do we find this optimal risky portfolio and finally optimize our portfolio to the maximum? 
# This is done by using a parameter called the Sharpe Ratio. 
# The ratio is the average return earned in excess of the risk-free rate per unit of volatility or total risk.
# The risk-free rate of return is the return on an investment with zero risk, meaning it’s the return investors could expect for taking no risk.

# We define as an assumption the risk-free rate to be 1% or 0.01.


# Finding the optimal portfolio
rf = 0.01 # risk factor
optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
print(optimal_risky_port)

# Plotting optimal portfolio
plt.subplots(figsize=(10, 10))
plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)
plt.show()

# The green star represents the optimal risky portfolio.
