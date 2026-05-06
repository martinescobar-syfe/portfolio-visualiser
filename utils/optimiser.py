
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

def create_constraints(min_weight, max_weight, fixed_weights=None):
    """create optimization constraints"""
    constraints = []
    constraints.append({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    constraints.append({'type': 'ineq', 'fun': lambda weights: weights - min_weight})
    constraints.append({'type': 'ineq', 'fun': lambda weights: max_weight - weights})
    
    return constraints

def negative_sharpe_ratio(weights, returns, covar, rf):
    portfolio_return = np.dot(returns.T, weights)
    weights_values = weights.values.flatten() if isinstance(weights, pd.DataFrame) else weights
    portfolio_volatility = np.sqrt(np.dot(weights_values.T, np.dot(covar, weights_values)))
    return -(portfolio_return - rf) / portfolio_volatility

def get_maxSharpe_wgt(ret, covar, rf, constraints, initial_weights=None, bounds=None):
    TOLERANCE = 1e-11
    if initial_weights is None:
        initial_weights = [1. / len(ret)] * len(ret)
    if bounds is None:
        bounds = [(0, 1)] * len(ret)
    
    result = minimize(negative_sharpe_ratio, initial_weights, args=(ret, covar, rf), 
                      method='SLSQP', constraints=constraints, bounds=bounds, tol=TOLERANCE)

    optimized_weights = result.x
    return optimized_weights

def _allocation_risk(weights, covariances):
    portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]
    return portfolio_risk

def _risk_budget_objective_error(weights, args):
    covariances = args[0]
    weights = np.matrix(weights)
    portfolio_risk = _allocation_risk(weights, covariances)
    return portfolio_risk

def min_vol(ret_data, covar_ann, constraints):
    initial_weights = [1. / len(ret_data.columns)] * len(ret_data.columns)
    
    optimize_result = minimize(fun=_risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covar_ann],
                               method='SLSQP',
                               constraints=constraints,
                               tol=1e-11,
                               options={'disp': False})

    weights = optimize_result.x
    return weights

def optimize_portfolio(returns_data, start_date, end_date, constraints_dict, method='max_sharpe', 
                       shrink_covar=True, rebal_months=[1], annualized=261, halflife=3.5, 
                       fixed_weights=None, min_weight=0.0, max_weight=1.0):
    """optimization for max sharpe, min vol, block opt"""
    
    ret_data_all = returns_data.loc[:end_date]
    ret_data_filtered = returns_data.loc[start_date:end_date]
    first_date = ret_data_all.iloc[0].name
    tickers= list(ret_data_all.columns)

    weights = pd.DataFrame(0.0, index=ret_data_filtered.index, columns=ret_data_filtered.columns)
    weights.index = pd.to_datetime(weights.index, format='%Y%m%d')
    reb_flag = pd.DataFrame(False, index=weights.index, columns=['reb_flag'])
    
    alpha = 1 - math.exp(math.log(0.5) / (halflife* annualized))
    span = (2 / alpha) - 1
    
    # separate fixed and free tickers
    free_tickers = tickers.copy()
    fixed_weight_sum = 0
    
    if fixed_weights:
        for ticker, weight in fixed_weights.items():
            if ticker in free_tickers:
                free_tickers.remove(ticker)
                fixed_weight_sum += weight
    
    remaining_budget = 1.0 - fixed_weight_sum
    
    for i in range(0, len(ret_data_filtered)):
        if (weights.index[i].month in rebal_months and weights.index[i-1].month != weights.index[i].month) or i==0:
            # rebal date
            if fixed_weights and remaining_budget < 1e-6:
                for t in tickers:
                    weights.loc[weights.index[i], t] = fixed_weights.get(t, 0.0)
                reb_flag.iloc[i] = True
            else:
                curr_date = ret_data_filtered.iloc[i].name
                if free_tickers:
                    ret_free = ret_data_all[free_tickers]
                    
                    if shrink_covar:
                        shrink_covar_matrix_model = LedoitWolf().fit(ret_free.loc[first_date:curr_date])
                        covar_ann = shrink_covar_matrix_model.covariance_ * annualized
                    else:
                        exp_cov_matrix = ret_free.loc[first_date:curr_date].ewm(span=span).cov(pairwise=True).iloc[-len(free_tickers):]
                        covar_ann = exp_cov_matrix * annualized
                    
                    mean_ret = (ret_free.loc[first_date:curr_date].mean() * annualized)
                    bounds = [(min_weight, max_weight)] * len(free_tickers)
                    std_constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
                    
                    if method == 'max_sharpe':
                        w_opt = get_maxSharpe_wgt(mean_ret, covar_ann, rf=0, constraints=std_constraints, bounds=bounds)
                    elif method == 'min_vol':
                        w_opt = min_vol(ret_free, covar_ann, constraints=std_constraints)
                    elif method == 'equal_weight':
                        w_opt = np.array([1.0/len(free_tickers)] * len(free_tickers))
                    else:
                        w_opt = np.array([1.0/len(free_tickers)] * len(free_tickers))
                    
                    # scale to remaining budget
                    w_opt = w_opt * remaining_budget
                    
                    for idx, ticker in enumerate(free_tickers):
                        weights.loc[weights.index[i], ticker] = w_opt[idx]
                
                if fixed_weights:
                    for ticker, weight in fixed_weights.items():
                        if ticker in tickers:
                            weights.loc[weights.index[i], ticker] = weight
                
                reb_flag.iloc[i] = True
            
        else:
            # drift
            weights.iloc[i] = weights.iloc[i-1] * (1+ ret_data_filtered.iloc[i].fillna(0))
            weights_sum = weights.iloc[i].sum()
            if weights_sum > 0:
                weights.iloc[i] /= weights_sum
            reb_flag.iloc[i] = False
    
    weights = weights.rename(columns={c: c+'_wgt' for c in weights.columns})
    weights = pd.concat([reb_flag, weights], axis=1)
    return weights

def calc_port_perf(returns_data, weights_data, name_prefix='Port'):
    """calc portfolio perf from returns and weights"""
    weights_only = weights_data[[c for c in weights_data.columns if c.endswith('_wgt')]]
    weights_only.columns = [c.replace('_wgt', '') for c in weights_only.columns]
    
    common_idx = returns_data.index.intersection(weights_only.index)
    returns_data = returns_data.loc[common_idx]
    weights_only = weights_only.loc[common_idx]
    
    port_ret = (returns_data * weights_only).sum(axis=1)
    port_ret.name = f'{name_prefix}_port_ret'
    port_cum_ret = (1 + port_ret).cumprod()
    port_cum_ret.name = f'{name_prefix}_cum_ret'
    
    return pd.concat([port_ret, port_cum_ret], axis=1)
