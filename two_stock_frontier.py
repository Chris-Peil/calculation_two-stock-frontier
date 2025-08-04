# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:53:01 2022

@author: Chris
"""

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


def covariance(x, y):
    # Finding the mean of the series x and y
    mean_x = sum(x)/float(len(x))
    mean_y = sum(y)/float(len(y))
    
    # Subtracting mean from the individual elements
    sub_x = [i - mean_x for i in x]
    sub_y = [i - mean_y for i in y]
    
    numerator = sum([sub_x[i]*sub_y[i] for i in range(len(sub_x))])
    denominator = len(x)-1
    cov = numerator/denominator
    
    return cov


def correlation(x, y):
    # Finding the mean of the series x and y
    mean_x = sum(x)/float(len(x))
    mean_y = sum(y)/float(len(y))
    
    # Subtracting mean from the individual elements
    sub_x = [i-mean_x for i in x]
    sub_y = [i-mean_y for i in y]
    
    # covariance for x and y
    numerator = sum([sub_x[i]*sub_y[i] for i in range(len(sub_x))])
    
    # Standard Deviation of x and y
    std_deviation_x = sum([sub_x[i]**2.0 for i in range(len(sub_x))])
    std_deviation_y = sum([sub_y[i]**2.0 for i in range(len(sub_y))])
    
    # squaring by 0.5 to find the square root
    denominator = (std_deviation_x*std_deviation_y)**0.5 # short but equivalent to (std_deviation_x**0.5) * (std_deviation_y**0.5)
    cor = numerator/denominator
    return cor


#This function only works for two or more stocks
def data_extraction(data):
    #Extracting column specific data from yahoo finance
    close_data = data['Adj Close']
    
    
    #Initializing data and its shape
    close_array = np.array(close_data)
    data_position_x = close_array.shape[0] -1
    data_selection_y = close_array.shape[1] - 1

    storage = []
    toggle_num = 0

    #Calculating returns as a percentage
    while data_position_x > 0:
        first_num = close_array[data_position_x, data_selection_y]
        second_num = close_array[data_position_x -1, data_selection_y]
        toggle_num = ((first_num/second_num)-1) *100
        storage.append(toggle_num)
        first_num = second_num
        second_num = toggle_num
        
        data_position_x -= 1
        
        #Check if current array is complete
        if data_position_x == 0:
            data_selection_y -= 1
            
            #Check if all arrays are complete
            if data_selection_y != -1:
                data_position_x = close_array.shape[0] -1        

    #Resize array into stock specific data                
    close_returns = np.array(storage)
    close_returns = np.resize(close_returns,(close_array.shape[1], close_array.shape[0]-1))
    
    
    return(close_returns)


def stat_calc(close_returns):
    #Initializing data and its shape
    data_parameters = close_returns.shape
    stats_position_x = data_parameters[0] -1
    
    stock_position = 0  #Hard coded in, accessing column data will fix this
    stats_mean = []
    stats_vari = []
    stats_std = []

    #Calculates and prints statistical data
    while stats_position_x > -1:
        stats_data = close_returns[stats_position_x, :]
        print(stocks[stock_position]) 
    
        #Returns
        mean_num = stats_data.mean()
        print("Mean:", mean_num)
        stats_mean.append(mean_num)
        
        #Variance
        vari_num = stats_data.var()
        print('Variance:', vari_num)
        stats_vari.append(vari_num)
        
        #Risk
        std_num = stats_data.std()
        print("Standard Deviation:", std_num)
        stats_std.append(std_num)
    
        print('')
        stats_position_x -= 1
        stock_position += 1
    

    covar_num = covariance(close_returns[0], close_returns[1])
    print('Covariance:', covar_num)
    correl_num = correlation(close_returns[0], close_returns[1])
    print('Correlation:', correl_num)
    print('')
    
    #Combines all data and returns as a single variable
    stats_values = [stats_mean, stats_vari, stats_std, covar_num, correl_num]
    return(stats_values)    



def data_plot(stats_values):
    #Checks total amount of stocks
    if len(stocks) == 2:
        #Initialization of stock weight combinations
        weight_array_up = list(range(0,11,1))
        weight_array_down = list(range(10,-1,-1))

        mean_results = []
        vari_results = []
        std_results = []


        #Statistical calculations for dual stock comparison
        array_counter = 0
        while array_counter < len(weight_array_up):
            mean_calc = (stats_values[0][0] * (weight_array_up[array_counter]/10) + 
                         stats_values[0][1] * (weight_array_down[array_counter]/10))
    
            mean_results.append(mean_calc)
            array_counter += 1


        array_counter = 0
        while array_counter < len(weight_array_up):
            vari_calc = (((weight_array_up[array_counter]/10)**2) * stats_values[1][0] + 
                         (((weight_array_down[array_counter]/10)**2) * stats_values[1][1]) + 
                         2 * (weight_array_up[array_counter]/10) * (weight_array_down[array_counter]/10) 
                         * stats_values[3])
    
            vari_results.append(vari_calc)
            array_counter += 1


        array_counter = 0
        while array_counter < len(vari_results):
            std_calc = (vari_results[array_counter]**(1/2))
    
            std_results.append(std_calc)
            array_counter += 1
    
        #Plot efficient frontier calculations
        print('Plotting graph...')
        print("")
        plt.xlabel('Risk (Standard Deviation)')
        plt.ylabel('% Returns (Mean)')
        plt.scatter(std_results, mean_results)
        plt.plot(std_results, mean_results)
        plt.show()    
    
        data_result = [stocks[0], weight_array_up, mean_results, std_results, weight_array_down, stocks[1]]
        return(data_result)


    else:
        print('No graphical data avaliable.')
        print("")
        return



def portfolio_calculations(data, weights):
    print("Portfolio analysis:")
    #daily returns as percentage
    daily_change = data["Close"].pct_change()
    #print(daily_change.describe())

    #portfolio return
    portfolio_returns = (daily_change * weights).sum(axis = 1)
    
    #total cumulative returns for the portfolio
    cumulative_returns = (portfolio_returns + 1).cumprod()
    plt.xlabel('Time')
    plt.ylabel('Returns %')
    plt.plot(cumulative_returns)
    plt.show() 

    #standard deviation I.E  daily risk/volatility
    risk = np.std(portfolio_returns)
    print("Daily risk", risk)
    
    #annual volatility
    annual_risk = np.std(portfolio_returns) * np.sqrt(252)
    print("Annual risk", annual_risk)
    
    '''
    #Correlation matrix
    correlation = daily_change.corr()
    print("Correlation:\n", correlation)
    '''
    
    #Sharpe ratio (higher is better)
    sharpe = (np.mean(portfolio_returns)/np.std(portfolio_returns)) * np.sqrt(252)
    print("Sharpe Ratio:", sharpe, " - Higher is better")
    print("")
    
    #Monte Carlo Simulation - random portfolio weights
    print("Monte Carlo Simulation:")
    mc_weights = []
    mc_returns = []
    mc_risk = []
    mc_sharpe = []
       
    count = 500
    for num in range(0, count):
        random_weights = np.random.uniform(size = len(daily_change.columns))
        random_weights = random_weights/np.sum(random_weights)
        mc_weights.append(random_weights)
        
        #returns
        mean_returns = (daily_change.mean() * random_weights).sum()*252
        mc_returns.append(mean_returns)
        
        #volatility
        mc_portfolio_returns = (daily_change * random_weights).sum(axis = 1)
        mc_annual_std = np.std(mc_portfolio_returns) * np.sqrt(252)
        mc_risk.append(mc_annual_std)
        
        #Sharpe Ratio
        sharpe_ratio = (np.mean(mc_portfolio_returns)/np.std(mc_portfolio_returns)) * np.sqrt(252)
        mc_sharpe.append(sharpe_ratio)
        
    #highest sharpe index value
    max_index = np.argmax(mc_sharpe)
    
    #Max sharpe ratio
    calc_sharpe = mc_sharpe[max_index]
    print("Simulated sharpe: ", calc_sharpe)
    
    #Optimal weights
    calc_weights = mc_weights[max_index]
    print("Simulated Weighting:")
    print(stocks, "\n", calc_weights)
    
    return

   
##############################################################################



##############################################################################



##############################################################################
'''
Efficient frontier for two stocks
Portfolio information for two or more stocks
Check and reorder the stocks with the data output from yf.download 
(ALPHABETICAL ORDER)
'''
stocks = ["IOO.AX", "IOZ.AX"]


#Commsec pocket example 
#stock = ["ETHI.AX", "IEM.AX", "IOO.AX", "IOZ.AX", "IXJ.AX", "NDQ.AX", "SYI.AX"]
#weights = [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1]

'''
For a complete stock comparison, individual weights must equal 1.0
For a portfolio stock comparison, individual weights must equal the portfolio weight
'''
weights = [0.5, 0.5]

'''
#Date format is YYYY-MM-DD
start_date = '2011-01-03'
finish_date = '2022-06-26'
'''

# valid intervals: 1d,5d,1wk,1mo,3mo
time_interval = "1d"
# valid periods: 5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd
time_period = "5y"
    
data = yf.download(stocks, period = time_period, interval = time_interval)


close_returns = data_extraction(data)
stats_values = stat_calc(close_returns)
data_result = data_plot(stats_values) 
portfolio_calculations(data, weights)

