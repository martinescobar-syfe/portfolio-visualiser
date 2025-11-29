# @title 1. Load libraries and functions
!pip install reportlab
!pip install kaleido
!pip install PyPDF2

import pandas as pd
import numpy as np
import math, os, io, time, pytz, re, copy
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
import statsmodels.api as sm
from scipy.stats import norm
from scipy.optimize import minimize, fsolve
import itertools
import plotly.graph_objects as go
import yfinance as yf
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib import colors
import statsmodels.formula.api as smf
from dateutil.parser import parse
from datetime import datetime, timedelta
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Javascript
from ipywidgets import VBox, HBox, Label, HTML
# Compatibility layer for Jupyter and Colab
import jupyter_colab_compat as jcc
jcc.setup_environment()

from contextlib import redirect_stdout
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from IPython.display import FileLink
from matplotlib.backends.backend_pdf import PdfPages
import PyPDF2
from PyPDF2 import PdfMerger, PdfReader, PdfWriter


import sys
sys.path.append(jcc.get_base_path())
import port_cons as pc

import warnings
warnings.filterwarnings("ignore")

def convert_to_datetime(input_str, parserinfo=None):
    return parse(input_str, parserinfo=parserinfo)

TOLERANCE = 1e-10

def _allocation_risk(weights, covariances):

    portfolio_risk = np.sqrt((weights * covariances * weights.T))[0, 0]

    return portfolio_risk

def _assets_risk_contribution_to_allocation_risk(weights, covariances):

    portfolio_risk = _allocation_risk(weights, covariances)

    assets_risk_contribution = np.multiply(weights.T, covariances * weights.T) \
        / portfolio_risk

    return assets_risk_contribution

def _risk_budget_objective_error(weights, args):
    covariances = args[0]
    assets_risk_budget = args[1]
    weights = np.matrix(weights)

    portfolio_risk = _allocation_risk(weights, covariances)

    assets_risk_contribution = \
        _assets_risk_contribution_to_allocation_risk(weights, covariances)

    assets_risk_target = \
        np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))

    error = sum(np.absolute(assets_risk_contribution - assets_risk_target.T))[0, 0]
    return error

def _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},{'type': 'ineq', 'fun': lambda x: x})

    optimize_result = minimize(fun=_risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances, assets_risk_budget],
                               method='SLSQP',
                               constraints=constraints,
                               tol=TOLERANCE,
                               options={'disp': False})

    weights = optimize_result.x
    print(optimize_result.message)
    return weights

def rebal_wgt_riskparity(returns_data, start_date, end_date, rebal_months=[4,10], halflife=3.5, annualized=252, shrink_covar=False):
    num_of_assets = len(returns_data.columns)

    ret_data_filtered = returns_data.loc[start_date:end_date]
    first_date = returns_data.index[0]
    tickers = list(returns_data.columns)

    weights = pd.DataFrame(0, index=ret_data_filtered.index, columns=ret_data_filtered.columns)
    weights.index = pd.to_datetime(weights.index, format='%Y-%m-%d')
    reb_flag = pd.DataFrame(0, index=weights.index, columns=['reb_flag'])

    alpha = 1 - math.exp(math.log(0.5) / (halflife * annualized))
    span = (2 / alpha) - 1

    for i in range(len(ret_data_filtered)):
        curr_date = ret_data_filtered.index[i]
        curr_date_format = curr_date.strftime('%Y-%m-%d')

        if (weights.index[i].month in rebal_months and weights.index[i-1].month != weights.index[i].month) or i==0:

            if shrink_covar:
                shrink_covar_matrix_model = LedoitWolf().fit(returns_data.loc[first_date:curr_date])
                covar_ann = pd.DataFrame(shrink_covar_matrix_model.covariance_ * annualized, index=tickers, columns=tickers)
                covar_ann.columns = tickers
                covar_ann.index = tickers
            else:
                exp_cov_matrix = returns_data.loc[first_date:curr_date].ewm(span=span).cov(pairwise=True).iloc[-num_of_assets:]
                covar_ann = exp_cov_matrix * annualized
                covar_ann.columns = tickers
                covar_ann.index = tickers

            valid_covar_ann = covar_ann.dropna(how='all')
            valid_covar_ann = valid_covar_ann.dropna(axis=1, how='all')
            valid_num_assets = len(valid_covar_ann)
            valid_tickers = valid_covar_ann.columns

            assets_risk_budget = np.ones([valid_num_assets]) / valid_num_assets
            initial_weights = np.ones([valid_num_assets]) / valid_num_assets

            # print(valid_covar_ann)
            new_wgt = _get_risk_parity_weights(valid_covar_ann.values, assets_risk_budget, initial_weights)
            new_wgt = pd.DataFrame(new_wgt).T
            new_wgt.columns = valid_tickers

            for ticker in valid_tickers:
                weights.at[curr_date_format, ticker] = new_wgt[ticker].values

            reb_flag.loc[curr_date_format] = True

        else:
            weights.iloc[i] = weights.iloc[i-1] * (1+ ret_data_filtered.iloc[i].fillna(0))
            weights_sum = weights.iloc[i].sum()
            weights.iloc[i] /= weights_sum
            reb_flag.iloc[i] = False

    weights = weights.rename(columns={c: c + '_wgt' for c in weights.columns})
    weights = pd.concat([reb_flag, weights], axis=1)

    return weights

def combine_backtest_data(portfolio_names):
    # combined portfolio returns
    all_port_ret = {}

    for portfolio in portfolio_names:
        perf_df = globals()[f"{portfolio}_perf"]
        ret_col = [col for col in perf_df.columns if col.endswith('_port_ret')]
        all_port_ret[portfolio] = perf_df[ret_col]

    df_all_port_ret = pd.concat(all_port_ret.values(), keys=all_port_ret.keys(), axis=1)
    df_all_port_ret.columns = [col[0] for col in df_all_port_ret.columns]

    # combined portfolio weights
    all_port_weights = {}

    for portfolio in portfolio_names:
        wgt_df = globals()[f"{portfolio}_wgt"]
        # wgt_col = [col for col in wgt_df.columns if col.endswith('_wgt')]
        # all_port_weights[portfolio] = wgt_df[wgt_col]
        all_port_weights[portfolio] = wgt_df

    df_all_port_weights = pd.concat(all_port_weights.values(), keys=all_port_weights.keys(), axis=1)

    # combined portfolio backtest daata
    all_port_bt = {}

    for portfolio in portfolio_names:
        bt_df = globals()[f"{portfolio}_perf"]
        all_port_bt[portfolio] = bt_df

    df_all_port_bt = pd.concat(all_port_bt.values(), keys=all_port_bt.keys(), axis=1)
    return df_all_port_ret, df_all_port_weights, df_all_port_bt


def print_arial(text):
    display(HTML(f"<div style='font-family: Arial, sans-serif'>{text}</div>"))

def print_arial_bold(text):
    display(HTML(f"<div style='font-family: Arial, sans-serif; font-weight: bold;'>{text}</div>"))

def merge_pdfs(existing_pdf, merged_pdf, output_pdf):
    # Open the existing PDFs
    with open(existing_pdf, "rb") as pdf1, open(merged_pdf, "rb") as pdf2:
        reader1 = PyPDF2.PdfReader(pdf1)
        reader2 = PyPDF2.PdfReader(pdf2)
        writer = PyPDF2.PdfWriter()

        # Add all pages from the first PDF
        for page in reader1.pages:
            writer.add_page(page)

        # Add all pages from the second PDF
        for page in reader2.pages:
            writer.add_page(page)

        # Save the merged PDF
        with open(output_pdf, "wb") as output:
            writer.write(output)

def plot_pie_chart_topX(series, title, topX=10, width=1000, height=600, colors=None):
    # Filter for values greater than 0
    series = series[series > 0]

    # Proceed only if there are values to plot
    if not series.empty:
        top_x = series.sort_values(ascending=False).head(topX)
        other_sum = series.sum() - top_x.sum()

        # Filter labels and values based on positive values
        labels = list(top_x.index)
        values = list(top_x)

        # Add 'Others' only if other_sum is greater than 0
        if other_sum > 0.01:
            labels.append('Others')
            values.append(other_sum)

        # Create the Pie trace data
        pie_data = {
            'labels': labels,
            'values': values,
            'textinfo': 'label+percent',
            'hole': 0.3,
            'textposition': 'outside',
        }

        # Add marker with custom colors if provided
        if colors:
            pie_data['marker'] = dict(colors=colors)

        fig = go.Figure(data=[go.Pie(**pie_data)])  # Unpack pie_data into go.Pie
        fig.update_layout(title={'text': f'<b>{title}</b>','font': dict(size=75)}, width=width, height=height, showlegend=False, font=dict(family="Arial", size=75, color='black'))
        return fig
    else:
        print("No values greater than 0 to plot.")
        return None

def plot_bar_chart_topX(series, title, topX=10, width=1000, height=600, colors=None):


    # Filter for values greater than 0
    series = series[series > 0]

    # Proceed only if there are values to plot
    if not series.empty:
        top_x = series.sort_values(ascending=False).head(topX)
        other_sum = series.sum() - top_x.sum()

        # Filter labels and values based on positive values
        labels = list(top_x.index)
        values = list(top_x)

        # Add 'Others' only if other_sum is greater than 0
        if other_sum > 0:
            labels.append('Others')
            values.append(other_sum)

        # Calculate percentages and format labels
        total_sum = sum(values)
        text_labels = [f'{round(value / total_sum * 100, 2)}%' for value in values]  # Only percentage

        # Create the Bar trace data
        bar_data = {
            'x': labels,
            'y': values,
            'text': text_labels,
            'textposition': 'auto',
            'textangle': 0
        }

        # Add marker with the first custom color if provided
        if colors:
            bar_data['marker'] = dict(color=colors[0])  # Use the first color from the list

        fig = go.Figure(data=[go.Bar(**bar_data)])  # Unpack bar_data into go.Bar
        fig.update_layout(title={'text': f'<b>{title}</b>','font': dict(size=75)}, width=width, height=height, showlegend=False, yaxis_tickformat=".0%", font=dict(family="Arial", size=75, color='black'))
        return fig
    else:
        print("No values greater than 0 to plot.")
        return None

########################################################################################################################

# read asseet class, sector, country data from csv
data_path = jcc.get_data_path()
df_sec_list = pd.read_csv(data_path + 'PV_sec_list.csv')
df_sec_list['Remarks'] = df_sec_list['Remarks'].fillna('N/A')
df_asset_class = pd.read_csv(data_path + 'PV_asset_class.csv', index_col=0)
df_sector = pd.read_csv(data_path + 'PV_sector.csv', index_col=0)
df_country = pd.read_csv(data_path + 'PV_country.csv', index_col=0)
df_fi_metrics = pd.read_csv(data_path + 'PV_FI_metrics.csv', index_col=0)
df_fx = pd.read_csv(data_path + 'PV_daily_ret.csv',header=0, index_col='Date', parse_dates=['Date'],dayfirst=True)[['USDSGD', 'USDHKD']]/100

# @title 2. Input parameters

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


syfe_port_mapping = pd.DataFrame({
'MB_port_name': ['CASH_PLUS', 'CASH_PLUS_USD', 'CHINA_GROWTH', 'CORE_BALANCED', 'CORE_DEFENSIVE', 'CORE_GROWTH', 'DISRUPTIVE_TECHNOLOGY', 'ESG_AND_CLEAN_ENERGY',
                  'GLOBAL_EQUITY_100', 'HEALTHCARE_INNOVATION', 'INCOME_ENHANCE', 'INCOME_PRESERVE', 'REIT', 'REIT_RISK_MANAGED',
                  'SRS_CASH_PLUS', 'SRS_GLOBAL_EQUITY_100', 'SRS_INCOME_ENHANCE', 'SRS_INCOME_PRESERVE', 'DOWNSIDE_PROTECTED', 'CASH_PLUS_GUARANTEED_SGD', 'CUSTOM_USD'],

'internal_port_name': ['Cash SGD flexi', 'Cash USD flexi', 'China Growth', 'Core Balanced', 'Core Defensive', 'Core Growth', 'Disruptive Technology',
                        'ESG & Clean Energy', 'Core Equity100', 'Healthcare Innovation', 'Income Enhance', 'Income Preserve', 'REIT 100', 'REIT Risk Managed',
                        'SRS Cash SGD flexi', 'SRS Equity100', 'SRS Income Enhance', 'SRS Income Preserve', 'Downside protected', 'Cash SGD guaranteed', 'Custom USD']
})
out = widgets.Output()

#####################################################################################################################
# Define a function to display securities for a given category
def display_securities_df(category):
    with out:
        out.clear_output(wait=True)  # Clear previous output
        filtered_df = df_sec_list[df_sec_list['Category'] == category][['Ticker', 'Name', 'Exchange','Remarks']].sort_values(by='Ticker', ascending=True)
        filtered_df = filtered_df.fillna('')
        display(filtered_df.reset_index(drop=True))

# Create buttons for each category
syfeportfolio_button = widgets.Button(description="Syfe portfolio")
equity_button = widgets.Button(description="Equities")
fixed_income_button = widgets.Button(description="Fixed income")
commodities_button = widgets.Button(description="Commodities")
multi_asset_button = widgets.Button(description="Multi-asset")
index_button = widgets.Button(description="Index")

# Assign the display_securities_df function to each button's on_click event
syfeportfolio_button.on_click(lambda b: display_securities_df("Syfe portfolio"))
equity_button.on_click(lambda b: display_securities_df("Equities"))
fixed_income_button.on_click(lambda b: display_securities_df("Fixed income"))
commodities_button.on_click(lambda b: display_securities_df("Commodities"))
multi_asset_button.on_click(lambda b: display_securities_df("Multi-asset"))
index_button.on_click(lambda b: display_securities_df("Index"))

# Display the buttons and output widget
print_arial_bold('See full list of securities')
display(widgets.HBox([syfeportfolio_button, equity_button, fixed_income_button, commodities_button, multi_asset_button,index_button]))
display(out)

#####################################################################################################################
# @title Alternatively, check if a security is in the list

etf_ticker_input = widgets.Text(
    value='',
    placeholder='Enter ETF ticker',
    description='ETF Ticker:',
    disabled=False
)

# Create an output widget to display the result
output_check = widgets.Output()

# Define a function to check if the ETF ticker is in df_sec_list
def check_security(b):
  with output_check:
    clear_output(wait=True)
    ticker = etf_ticker_input.value  # Convert to uppercase for case-insensitivity
    if ticker in df_sec_list['Ticker'].values:
      print(f"{ticker} is in the list.")
    else:
      print(f"{ticker} is not in the list.")

# Create a button to trigger the check
print('\n')
print_arial_bold('Alternatively, check if a security is in the list')
check_button = widgets.Button(description="Check Security")
check_button.on_click(check_security)

# Display the widgets
display(etf_ticker_input, check_button, output_check)

#####################################################################################################################

bbg_last_update_date = pd.read_csv(data_path + 'PV_daily_ret.csv', index_col=0, usecols=[0], parse_dates=['Date'],dayfirst=True).index[-1]
bbg_last_update_date = bbg_last_update_date.strftime('%Y-%m-%d')

print('\n')
print_arial_bold('Choose data source')
print_arial('• Bloomberg: Provides total return data, but not updated in real-time. Last updated: ' + str(bbg_last_update_date))
print_arial('• Yahoo Finance: Provides live data (as of last close), but includes only price returns.')

data_source_widget = widgets.RadioButtons(
       options=['Bloomberg', 'Yahoo Finance'],
       description='Data Source:',
       disabled=False)
display(data_source_widget)

#####################################################################################################################
# @title Input data parameters

# Widget for start date
start_date_widget = widgets.DatePicker(
    description='Start Date:',
    layout=widgets.Layout(width='300px'),  # Adjust width as needed
    style={'description_width': 'initial'}  # Allow description to take full width
)

# Widget for end date
end_date_widget = widgets.DatePicker(
    description='End Date:',
    layout=widgets.Layout(width='300px'),  # Adjust width as needed
    style={'description_width': 'initial'}  # Allow description to take full width
)

# Widget for benchmark tickers
bm_tickers_widget = widgets.Text(
    value='',
    description='Benchmark Tickers:',
    disabled=False,
    layout=widgets.Layout(width='300px'),  # Adjust width as needed
    style={'description_width': 'initial'}  # Allow description to take full width
)

# Widget for backtest offset days
backtest_offset_days_widget = widgets.IntText(
    value=0,
    description='Observation window (Days):',
    disabled=False,
    layout=widgets.Layout(width='300px'),  # Adjust width as needed
    style={'description_width': 'initial'}  # Allow description to take full width
)

# Widget for currency option
currency_dropdown = widgets.Dropdown(
    options=['USD', 'SGD', 'HKD'],
    value='USD',
    description='Currency:',
    disabled=False,
    layout=widgets.Layout(width='300px'),
    style={'description_width': 'initial'}
)
# Display the widgets
print_arial_bold('Input data parameters')
print_arial("<br>".join([
    "•   Start date: format - DD/MM/YYYY",
    "•   End date: format - DD/MM/YYYY",
    "•   Benchmark ticker format - ETF1, ETF2, ETF3, ... | e.g. ACWI, URTH, SPY",
    "•   Observation window (days): the number of days after the specified start date used to initialize the optimization process - \
    this ensures enough data is available before the backtest begins (i.e. if lookback window is 365 days and start date is Jan 1 2010, the actual backtest will begin in Jan 1 2011)"]))

print('\n')
print_arial("❗ If no start and end date is specified, the backtest will begin from the earliest common inception until latest available data")

display(start_date_widget)
display(end_date_widget)
display(bm_tickers_widget)
display(backtest_offset_days_widget)
display(currency_dropdown)

# @title 2. Input parameters

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


syfe_port_mapping = pd.DataFrame({
'MB_port_name': ['CASH_PLUS', 'CASH_PLUS_USD', 'CHINA_GROWTH', 'CORE_BALANCED', 'CORE_DEFENSIVE', 'CORE_GROWTH', 'DISRUPTIVE_TECHNOLOGY', 'ESG_AND_CLEAN_ENERGY',
                  'GLOBAL_EQUITY_100', 'HEALTHCARE_INNOVATION', 'INCOME_ENHANCE', 'INCOME_PRESERVE', 'REIT', 'REIT_RISK_MANAGED',
                  'SRS_CASH_PLUS', 'SRS_GLOBAL_EQUITY_100', 'SRS_INCOME_ENHANCE', 'SRS_INCOME_PRESERVE', 'DOWNSIDE_PROTECTED', 'CASH_PLUS_GUARANTEED_SGD', 'CUSTOM_USD'],

'internal_port_name': ['Cash SGD flexi', 'Cash USD flexi', 'China Growth', 'Core Balanced', 'Core Defensive', 'Core Growth', 'Disruptive Technology',
                        'ESG & Clean Energy', 'Core Equity100', 'Healthcare Innovation', 'Income Enhance', 'Income Preserve', 'REIT 100', 'REIT Risk Managed',
                        'SRS Cash SGD flexi', 'SRS Equity100', 'SRS Income Enhance', 'SRS Income Preserve', 'Downside protected', 'Cash SGD guaranteed', 'Custom USD']
})
out = widgets.Output()

#####################################################################################################################
# Define a function to display securities for a given category
def display_securities_df(category):
    with out:
        out.clear_output(wait=True)  # Clear previous output
        filtered_df = df_sec_list[df_sec_list['Category'] == category][['Ticker', 'Name', 'Exchange','Remarks']].sort_values(by='Ticker', ascending=True)
        filtered_df = filtered_df.fillna('')
        display(filtered_df.reset_index(drop=True))

# Create buttons for each category
syfeportfolio_button = widgets.Button(description="Syfe portfolio")
equity_button = widgets.Button(description="Equities")
fixed_income_button = widgets.Button(description="Fixed income")
commodities_button = widgets.Button(description="Commodities")
multi_asset_button = widgets.Button(description="Multi-asset")
index_button = widgets.Button(description="Index")

# Assign the display_securities_df function to each button's on_click event
syfeportfolio_button.on_click(lambda b: display_securities_df("Syfe portfolio"))
equity_button.on_click(lambda b: display_securities_df("Equities"))
fixed_income_button.on_click(lambda b: display_securities_df("Fixed income"))
commodities_button.on_click(lambda b: display_securities_df("Commodities"))
multi_asset_button.on_click(lambda b: display_securities_df("Multi-asset"))
index_button.on_click(lambda b: display_securities_df("Index"))

# Display the buttons and output widget
print_arial_bold('See full list of securities')
display(widgets.HBox([syfeportfolio_button, equity_button, fixed_income_button, commodities_button, multi_asset_button,index_button]))
display(out)

#####################################################################################################################
# @title Alternatively, check if a security is in the list

etf_ticker_input = widgets.Text(
    value='',
    placeholder='Enter ETF ticker',
    description='ETF Ticker:',
    disabled=False
)

# Create an output widget to display the result
output_check = widgets.Output()

# Define a function to check if the ETF ticker is in df_sec_list
def check_security(b):
  with output_check:
    clear_output(wait=True)
    ticker = etf_ticker_input.value  # Convert to uppercase for case-insensitivity
    if ticker in df_sec_list['Ticker'].values:
      print(f"{ticker} is in the list.")
    else:
      print(f"{ticker} is not in the list.")

# Create a button to trigger the check
print('\n')
print_arial_bold('Alternatively, check if a security is in the list')
check_button = widgets.Button(description="Check Security")
check_button.on_click(check_security)

# Display the widgets
display(etf_ticker_input, check_button, output_check)

#####################################################################################################################

bbg_last_update_date = pd.read_csv(data_path + 'PV_daily_ret.csv', index_col=0, usecols=[0], parse_dates=['Date'],dayfirst=True).index[-1]
bbg_last_update_date = bbg_last_update_date.strftime('%Y-%m-%d')

print('\n')
print_arial_bold('Choose data source')
print_arial('• Bloomberg: Provides total return data, but not updated in real-time. Last updated: ' + str(bbg_last_update_date))
print_arial('• Yahoo Finance: Provides live data (as of last close), but includes only price returns.')

data_source_widget = widgets.RadioButtons(
       options=['Bloomberg', 'Yahoo Finance'],
       description='Data Source:',
       disabled=False)
display(data_source_widget)

#####################################################################################################################
# @title Input data parameters

# Widget for start date
start_date_widget = widgets.DatePicker(
    description='Start Date:',
    layout=widgets.Layout(width='300px'),  # Adjust width as needed
    style={'description_width': 'initial'}  # Allow description to take full width
)

# Widget for end date
end_date_widget = widgets.DatePicker(
    description='End Date:',
    layout=widgets.Layout(width='300px'),  # Adjust width as needed
    style={'description_width': 'initial'}  # Allow description to take full width
)

# Widget for benchmark tickers
bm_tickers_widget = widgets.Text(
    value='',
    description='Benchmark Tickers:',
    disabled=False,
    layout=widgets.Layout(width='300px'),  # Adjust width as needed
    style={'description_width': 'initial'}  # Allow description to take full width
)

# Widget for backtest offset days
backtest_offset_days_widget = widgets.IntText(
    value=0,
    description='Observation window (Days):',
    disabled=False,
    layout=widgets.Layout(width='300px'),  # Adjust width as needed
    style={'description_width': 'initial'}  # Allow description to take full width
)

# Widget for currency option
currency_dropdown = widgets.Dropdown(
    options=['USD', 'SGD', 'HKD'],
    value='USD',
    description='Currency:',
    disabled=False,
    layout=widgets.Layout(width='300px'),
    style={'description_width': 'initial'}
)
# Display the widgets
print_arial_bold('Input data parameters')
print_arial("<br>".join([
    "•   Start date: format - DD/MM/YYYY",
    "•   End date: format - DD/MM/YYYY",
    "•   Benchmark ticker format - ETF1, ETF2, ETF3, ... | e.g. ACWI, URTH, SPY",
    "•   Observation window (days): the number of days after the specified start date used to initialize the optimization process - \
    this ensures enough data is available before the backtest begins (i.e. if lookback window is 365 days and start date is Jan 1 2010, the actual backtest will begin in Jan 1 2011)"]))

print('\n')
print_arial("❗ If no start and end date is specified, the backtest will begin from the earliest common inception until latest available data")

display(start_date_widget)
display(end_date_widget)
display(bm_tickers_widget)
display(backtest_offset_days_widget)
display(currency_dropdown)


#####################################################################################################################
# @title Define variables

# Function to update variables based on widget values
def update_variables(change):
    global start_date, end_date, tickers, bm_tickers, backtest_offset_days

    start_date = start_date_widget.value

    # Check if end_date_widget has a value (user input)
    if end_date_widget.value is None:
        # If not, set end_date to the latest date from ret_all (assuming ret_all is the downloaded DataFrame)
        if 'ret_all' in globals() and not ret_all.empty:  # Check if ret_all exists and is not empty
            end_date = ret_all.index.max().date()
        else:
            end_date = None  # Or handle the case where ret_all is not yet available
    else:
        end_date = end_date_widget.value  # Use user-provided end date

    bm_tickers = re.split(r'\s*,\s*', bm_tickers_widget.value)
    backtest_offset_days = backtest_offset_days_widget.value

# Observe widget changes and update variables
start_date_widget.observe(update_variables, names='value')
end_date_widget.observe(update_variables, names='value')
bm_tickers_widget.observe(update_variables, names='value')
backtest_offset_days_widget.observe(update_variables, names='value')

# Initial variable update
update_variables(None)

def modify_tickers_based_on_exchange(all_tickers):
  modified_tickers = []
  for ticker in all_tickers:
      exchange = df_sec_list.loc[df_sec_list['Ticker'] == ticker, 'Exchange'].iloc[0] if ticker in df_sec_list['Ticker'].values else None
      if exchange == 'London':
          modified_tickers.append(ticker + '.L')
      elif exchange == 'Swiss':
          modified_tickers.append(ticker + '.SW')
      else:
          modified_tickers.append(ticker)
  return modified_tickers

def download_data(data_source, start_date, end_date):
  global ret_all, ret_bm, earliest_start_date, all_tickers # Declare as global
  all_tickers = get_unique_tickers()
  chosen_currency = currency_dropdown.value
  df_fx = pd.read_csv(data_path + 'PV_daily_ret.csv',header=0, index_col='Date', parse_dates=['Date'],dayfirst=True)[['USDSGD', 'USDHKD']]/100

  if data_source == 'Yahoo Finance':
    modified_tickers = modify_tickers_based_on_exchange(all_tickers)

    if start_date is None and end_date is None:
      df_all = yf.download(modified_tickers, period='max')['Adj Close']
    else:
      df_all = yf.download(modified_tickers, start=start_date, end=end_date)['Adj Close']

    df_all.index = df_all.index.tz_localize(None)
    ret_all = df_all.pct_change().dropna()
    ret_all.columns = all_tickers

  elif data_source == 'Bloomberg':
    ret_all = pd.read_csv(data_path + 'PV_daily_ret.csv',header=0, index_col='Date', parse_dates=['Date'],dayfirst=True)[all_tickers]/100
    first_valid_index = ret_all.apply(lambda series: series.first_valid_index()).max()
    ret_all = ret_all.loc[first_valid_index:].dropna(how='all')  # Use how='all' to drop rows with all NaN values

    # Filter by start_date and end_date if provided
    if start_date is not None:
        start_date = start_date.strftime('%Y-%m-%d')
        ret_all = ret_all.loc[start_date:]
    if end_date is not None:
        end_date = end_date.strftime('%Y-%m-%d')
        ret_all = ret_all.loc[:end_date]

  ret_all = ret_all.fillna(0)
  # adjust for desired currency
  for ticker in all_tickers:
      base_currency = df_sec_list.loc[df_sec_list['Ticker'] == ticker, 'Base currency'].iloc[0]
      if base_currency != chosen_currency:
          if base_currency == 'USD' and chosen_currency == 'SGD':
              fx_rate_changes = df_fx['USDSGD'].loc[ret_all.index] # Filter FX rates by ret_all dates
              ret_all[ticker] = (1 + ret_all[ticker]) * (1 + fx_rate_changes) - 1
          elif base_currency == 'USD' and chosen_currency == 'HKD':
              fx_rate_changes = df_fx['USDHKD'].loc[ret_all.index]
              ret_all[ticker] = (1 + ret_all[ticker]) * (1 + fx_rate_changes) - 1
          elif base_currency == 'SGD' and chosen_currency == 'USD':
              fx_rate_changes = -df_fx['USDSGD'].loc[ret_all.index] / (1 + df_fx['USDSGD'].loc[ret_all.index])
              ret_all[ticker] = (1 + ret_all[ticker]) * (1 + fx_rate_changes) - 1
          elif base_currency == 'HKD' and chosen_currency == 'USD':
              fx_rate_changes = -df_fx['USDHKD'].loc[ret_all.index] / (1 + df_fx['USDHKD'].loc[ret_all.index])
              ret_all[ticker] = (1 + ret_all[ticker]) * (1 + fx_rate_changes) - 1
      else:
        pass

  ret_bm = ret_all[bm_tickers]
  earliest_start_date = ret_all.index.min()
  start_date_timestamp = pd.Timestamp(start_date)

  # Compare earliest_start_date with start_date_timestamp
  if earliest_start_date > start_date_timestamp:
    print('\n', 'Earliest start date: ', earliest_start_date)
  else:
    pass

download_button = widgets.Button(description="Download Data")

#####################################################################################################################
# @title Input portfolio parameters

# Function to create the fields for a single portfolio

def create_portfolio_widgets(portfolio_number):
    """Creates and returns widgets for a single portfolio."""
    widgets_dict = {
        'header': widgets.HTML(f"<h3>Portfolio {portfolio_number}</h3>"),
        'tickers': widgets.Text(value='', description='Tickers:'),
        'allocation_type': widgets.Dropdown(
            options=['Max sharpe', 'Min vol', 'Risk parity', 'Equal weight', 'Custom weight'],
            value='Max sharpe',
            description='Allocation:'
        ),
        'name': widgets.Text(value=f'P{portfolio_number}', description='Name:'),
        'rebal_freq': widgets.Dropdown(
            options=['Monthly', 'Quarterly', 'Semi-annually', 'Annually'],
            value='Annually',
            description='Rebal Freq:'
        ),
        'min_weight': widgets.FloatText(value=0.0, description='Min Weight:'),
        'max_weight': widgets.FloatText(value=1.0, description='Max Weight:'),
        'custom_weight': widgets.Text(
            value='',
            description='Input Weight:',
            layout=widgets.Layout(visibility='hidden')  # Initially hidden
        ),
    }

    def toggle_custom_weight_visibility(change):
        widgets_dict['custom_weight'].layout.visibility = (
            'visible' if widgets_dict['allocation_type'].value == 'Custom weight' else 'hidden'
        )

    widgets_dict['allocation_type'].observe(toggle_custom_weight_visibility, names='value')
    return widgets_dict

# Function to display portfolio widgets
def display_portfolio_widgets(widgets_dict):
    """Organizes portfolio widgets into a VBox for display."""
    return VBox(list(widgets_dict.values()))

# Initialize portfolio storage
portfolio_widgets = {}
portfolio_count = 0
portfolio_output = widgets.Output()

# Add and Delete Portfolio Buttons
add_portfolio_button = widgets.Button(description="Add Portfolio")
delete_portfolio_button = widgets.Button(description="Delete Portfolio")

def add_portfolio(b):
    global portfolio_count
    if portfolio_count < 5:  # Limit to 5 portfolios
        portfolio_count += 1
        portfolio_widgets[portfolio_count] = create_portfolio_widgets(portfolio_count)
        with portfolio_output:
            display(display_portfolio_widgets(portfolio_widgets[portfolio_count]))
    else:
        print("Maximum number of portfolios reached.")

def delete_portfolio(b):
    global portfolio_count
    if portfolio_count > 1:
        with portfolio_output:
            clear_output(wait=True)
            del portfolio_widgets[portfolio_count]
            portfolio_count -= 1
            for i in range(1, portfolio_count + 1):
                display(display_portfolio_widgets(portfolio_widgets[i]))
    else:
        print("Cannot delete the last remaining portfolio.")

add_portfolio_button.on_click(add_portfolio)
delete_portfolio_button.on_click(delete_portfolio)

#####################################################################################################################
print('\n')
print_arial_bold('Upload client holdings from <a href="https://metabase.internal.syfe.com/question/1470-find-clients-portfolios-by-email?email=&portfolio_id=&client_id=&phone=&source_of_funds=" target="_blank">Metabase</a>')

uploader = widgets.FileUpload(
    accept='*',  # Accepts all file types
    multiple=True  # Only allows one file to be uploaded at a time
)

# function to upload file
def handle_upload(change):
    global df_client_portfolios

    # --- FIX START: Handle ipywidgets 8+ structure ---
    if not uploader.value:
        return
        
    # Get the first uploaded file (ipywidgets 8+ returns a tuple of dicts)
    uploaded_file = uploader.value[0]
    uploaded_filename = uploaded_file['name']
    content = uploaded_file['content']
    # --- FIX END ---

    syfe_port_mapping = pd.DataFrame({
    'MB_port_name': ['CASH_PLUS', 'CASH_PLUS_USD', 'CHINA_GROWTH', 'CORE_BALANCED', 'CORE_DEFENSIVE', 'CORE_GROWTH', 'DISRUPTIVE_TECHNOLOGY', 'ESG_AND_CLEAN_ENERGY',
                     'GLOBAL_EQUITY_100', 'HEALTHCARE_INNOVATION', 'INCOME_ENHANCE', 'INCOME_PRESERVE', 'REIT', 'REIT_RISK_MANAGED',
                     'SRS_CASH_PLUS', 'SRS_GLOBAL_EQUITY_100', 'SRS_INCOME_ENHANCE', 'SRS_INCOME_PRESERVE', 'DOWNSIDE_PROTECTED', 'CASH_PLUS_GUARANTEED_SGD'],

    'internal_port_name': ['Cash SGD flexi', 'Cash USD flexi', 'China Growth', 'Core Balanced', 'Core Defensive', 'Core Growth', 'Disruptive Technology',
                           'ESG & Clean Energy', 'Core Equity100', 'Healthcare Innovation', 'Income Enhance', 'Income Preserve', 'REIT 100', 'REIT Risk Managed',
                           'SRS Cash SGD flexi', 'SRS Equity100', 'SRS Income Enhance', 'SRS Income Preserve', 'Downside protected', 'Cash SGD guaranteed']
    })

    with open(uploaded_filename, 'wb') as f:
        f.write(content)
    print(f'Uploaded `{uploaded_filename}` successfully!')

    # Read the CSV file into a pandas DataFrame
    try:
        df_client_portfolios = pd.read_csv(io.BytesIO(content))
        df_client_portfolios_active = df_client_portfolios[df_client_portfolios['status']=='ACTIVE'] # filter for active portfolios only
        # df_client_portfolios_active = df_client_portfolios_active[df_client_portfolios_active['type']!= 'CASH_PLUS_GUARANTEED_SGD']
        df_client_portfolios_active = process_custom_usd(df_client_portfolios_active) # explode custom usd underlying into individual rows
        df_client_portfolios_active = df_client_portfolios_active.merge(syfe_port_mapping, how='left', left_on='type', right_on='MB_port_name')

        # Handle custom usd mapping: filter for rows with NaN MB_port_name and update MB_port_name with 'type'
        nan_mb_port_name_rows = df_client_portfolios_active['MB_port_name'].isna()
        df_client_portfolios_active.loc[nan_mb_port_name_rows, 'MB_port_name'] = df_client_portfolios_active.loc[nan_mb_port_name_rows, 'type']
        df_client_portfolios_active.loc[nan_mb_port_name_rows, 'internal_port_name'] = df_client_portfolios_active.loc[nan_mb_port_name_rows, 'type']
        custom_usd_rows = df_client_portfolios_active[nan_mb_port_name_rows].copy()
        custom_usd_rows = custom_usd_rows.merge(df_sec_list[['Ticker']], how='left', left_on='MB_port_name', right_on='Ticker')

        non_custom_usd_rows = df_client_portfolios_active[~nan_mb_port_name_rows]  # Rows that were not originally NaN
        df_client_portfolios_active = pd.concat([non_custom_usd_rows, custom_usd_rows], ignore_index=True)

        aggregated_holdings = aggregate_client_holdings(df_client_portfolios_active)
        populate_portfolio_widgets(aggregated_holdings)
        with portfolio_output:
            clear_output(wait=True)
        # Display the single portfolio widget in the output
        with portfolio_output:
            display(display_portfolio_widgets(portfolio_widgets[portfolio_count]))

    except pd.errors.ParserError:
        print("Error: Could not parse the file as a CSV. Please ensure it's a valid CSV file.")

uploader.observe(handle_upload, names='value') # Observe the uploader for changes
display(uploader) # Display the uploader

# # function to upload file
# def handle_upload(change):
#     global df_client_portfolios

#     syfe_port_mapping = pd.DataFrame({
#     'MB_port_name': ['CASH_PLUS', 'CASH_PLUS_USD', 'CHINA_GROWTH', 'CORE_BALANCED', 'CORE_DEFENSIVE', 'CORE_GROWTH', 'DISRUPTIVE_TECHNOLOGY', 'ESG_AND_CLEAN_ENERGY',
#                      'GLOBAL_EQUITY_100', 'HEALTHCARE_INNOVATION', 'INCOME_ENHANCE', 'INCOME_PRESERVE', 'REIT', 'REIT_RISK_MANAGED',
#                      'SRS_CASH_PLUS', 'SRS_GLOBAL_EQUITY_100', 'SRS_INCOME_ENHANCE', 'SRS_INCOME_PRESERVE', 'DOWNSIDE_PROTECTED', 'CASH_PLUS_GUARANTEED_SGD'],

#     'internal_port_name': ['Cash SGD flexi', 'Cash USD flexi', 'China Growth', 'Core Balanced', 'Core Defensive', 'Core Growth', 'Disruptive Technology',
#                            'ESG & Clean Energy', 'Core Equity100', 'Healthcare Innovation', 'Income Enhance', 'Income Preserve', 'REIT 100', 'REIT Risk Managed',
#                            'SRS Cash SGD flexi', 'SRS Equity100', 'SRS Income Enhance', 'SRS Income Preserve', 'Downside protected', 'Cash SGD guaranteed']
#     })

#     uploaded_filename = next(iter(uploader.value))
#     content = uploader.value[uploaded_filename]['content']
#     with open(uploaded_filename, 'wb') as f:
#         f.write(content)
#     print(f'Uploaded `{uploaded_filename}` successfully!')

#     # Read the CSV file into a pandas DataFrame
#     try:
#         df_client_portfolios = pd.read_csv(io.BytesIO(content))
#         df_client_portfolios_active = df_client_portfolios[df_client_portfolios['status']=='ACTIVE'] # filter for active portfolios only
#         # df_client_portfolios_active = df_client_portfolios_active[df_client_portfolios_active['type']!= 'CASH_PLUS_GUARANTEED_SGD']
#         df_client_portfolios_active = process_custom_usd(df_client_portfolios_active) # explode custom usd underlying into individual rows
#         df_client_portfolios_active = df_client_portfolios_active.merge(syfe_port_mapping, how='left', left_on='type', right_on='MB_port_name')

#         # Handle custom usd mapping: filter for rows with NaN MB_port_name and update MB_port_name with 'type'
#         nan_mb_port_name_rows = df_client_portfolios_active['MB_port_name'].isna()
#         df_client_portfolios_active.loc[nan_mb_port_name_rows, 'MB_port_name'] = df_client_portfolios_active.loc[nan_mb_port_name_rows, 'type']
#         df_client_portfolios_active.loc[nan_mb_port_name_rows, 'internal_port_name'] = df_client_portfolios_active.loc[nan_mb_port_name_rows, 'type']
#         custom_usd_rows = df_client_portfolios_active[nan_mb_port_name_rows].copy()
#         custom_usd_rows = custom_usd_rows.merge(df_sec_list[['Ticker']], how='left', left_on='MB_port_name', right_on='Ticker')

#         non_custom_usd_rows = df_client_portfolios_active[~nan_mb_port_name_rows]  # Rows that were not originally NaN
#         df_client_portfolios_active = pd.concat([non_custom_usd_rows, custom_usd_rows], ignore_index=True)

#         aggregated_holdings = aggregate_client_holdings(df_client_portfolios_active)
#         populate_portfolio_widgets(aggregated_holdings)
#         with portfolio_output:
#             clear_output(wait=True)
#         # Display the single portfolio widget in the output
#         with portfolio_output:
#             display(display_portfolio_widgets(portfolio_widgets[portfolio_count]))

#     except pd.errors.ParserError:
#         print("Error: Could not parse the file as a CSV. Please ensure it's a valid CSV file.")

# uploader.observe(handle_upload, names='value') # Observe the uploader for changes
# display(uploader) # Display the uploader

def aggregate_client_holdings(df_client_portfolios):
    # Aggregate holdings based on 'internal_port_name'
    aggregated_holdings = df_client_portfolios.groupby('internal_port_name').agg({
        'internal_port_name': lambda x: ', '.join(x.astype(str)),  # Join tickers with commas
        'nav_in_sgd': 'sum'  # Sum market values
    })
    # aggregated_holdings[['nav_in_sgd']] = aggregated_holdings[['nav_in_sgd']].apply(pd.to_numeric)
    # Calculate allocations
    total_market_value = aggregated_holdings['nav_in_sgd'].sum()
    aggregated_holdings['Allocation'] = aggregated_holdings['nav_in_sgd'] / total_market_value
    aggregated_holdings['Allocation'] = round(aggregated_holdings['Allocation'], 8)
    return aggregated_holdings

def process_custom_usd(df):
    df["nav_in_sgd"] = df["nav_in_sgd"].replace(",", "", regex=True)
    df["nav_in_sgd"] = pd.to_numeric(df["nav_in_sgd"], errors="coerce")
    df["nav_in_usd"] = df["nav_in_usd"].replace(",", "", regex=True)
    df["nav_in_usd"] = pd.to_numeric(df["nav_in_usd"], errors="coerce")

    # Separate CUSTOM_USD portfolios
    custom_usd_df = df[df["type"] == "CUSTOM_USD"].copy()
    other_portfolios_df = df[df["type"] != "CUSTOM_USD"]

    split_custom_usd_df = []

    for _, row in custom_usd_df.iterrows():
        holdings_str = row["actual_holding_weightage"]

        if isinstance(holdings_str, str):  # Ensure it's a string
            try:
                # Split by comma and then by spaces
                holdings = [
                    h.strip().rsplit(" ", 1)  # Split ticker and weight
                    for h in holdings_str.split(",")
                    if h.strip()]
                # print(row)
                for ticker, weight_str in holdings:
                    try:
                        weight = float(weight_str.strip("%")) / 100
                        new_row = row.copy()
                        new_row["type"] = ticker
                        nav_in_sgd = row["nav_in_sgd"]
                        if isinstance(nav_in_sgd, str):
                          nav_in_sgd = float(nav_in_sgd.replace(",", ""))
                        new_row["nav_in_sgd"] = nav_in_sgd * weight if nav_in_sgd else None
                        nav_in_usd = row["nav_in_usd"]
                        if isinstance(nav_in_usd, str):
                            nav_in_usd = float(nav_in_usd.replace(",", ""))
                        new_row["nav_in_usd"] = nav_in_usd * weight if nav_in_usd else None

                        split_custom_usd_df.append(new_row)
                    except ValueError:
                        print(f"Warning: Invalid weight format for ticker {ticker}: {weight_str}")
            except Exception as e:
                print(f"Error processing holdings string '{holdings_str}': {e}")
        else:
            print(f"Warning: actual_holding_weightage is not a string: {holdings_str}")

    # Concatenate the split CUSTOM_USD rows with the non-CUSTOM_USD rows
    processed_df = pd.concat([other_portfolios_df, pd.DataFrame(split_custom_usd_df)],ignore_index=True,)
    processed_df[['nav_in_sgd']] = processed_df[['nav_in_sgd']].apply(pd.to_numeric)
    processed_df[['nav_in_usd']] = processed_df[['nav_in_usd']].apply(pd.to_numeric)


    return processed_df

def populate_portfolio_widgets(aggregated_holdings):
    global portfolio_count, portfolio_widgets  # Access global variables

    portfolio_count = 1  # Reset to 1 portfolio
    portfolio_widgets = {}  # Clear existing widgets
    portfolio_widgets[portfolio_count] = create_portfolio_widgets(portfolio_count)  # Create a single portfolio widget

    # Combine all tickers into a single string
    all_tickers = ', '.join(aggregated_holdings.index.astype(str).tolist())

    # Combine all allocations into a single string
    all_allocations = ', '.join(aggregated_holdings['Allocation'].astype(str).tolist())

    # Populate widget values for the single portfolio
    portfolio_widgets[portfolio_count]['tickers'].value = all_tickers
    portfolio_widgets[portfolio_count]['allocation_type'].value = 'Custom weight'
    portfolio_widgets[portfolio_count]['name'].value = 'Client Portfolio'  # Set a name for the portfolio
    portfolio_widgets[portfolio_count]['custom_weight'].value = all_allocations

#####################################################################################################################
# Search Functionality
print('\n')
print_arial_bold('Search and add securities to your portfolio or type in the desired securities in the "Tickers" field')

search_input = widgets.Text(value='', placeholder='Search for a security/portfolio', description='Search:', disabled=False)
search_output = widgets.Output()

def add_ticker_to_portfolio(ticker, portfolio_number):
    """Add a ticker to a specific portfolio."""
    portfolio_tickers = portfolio_widgets[portfolio_number]['tickers']
    current_tickers = portfolio_tickers.value
    if current_tickers:
        portfolio_tickers.value = current_tickers + ", " + ticker
    else:
        portfolio_tickers.value = ticker

def on_search_input_change(change):
    """Filter and display the search results."""
    with search_output:
        clear_output(wait=True)
        search_term = search_input.value
        if search_term:
            filtered_df = df_sec_list[df_sec_list['Ticker'].str.contains(search_term, case=False, na=False) |
                                      df_sec_list['Name'].str.contains(search_term, case=False, na=False)]
        else:
            filtered_df = df_sec_list

        if not filtered_df.empty:
            rows = []
            for _, row in filtered_df.iterrows():
                add_buttons = []
                for i in range(1, portfolio_count + 1):
                    button = widgets.Button(description=f"Add to Portfolio {i}", button_style='primary')
                    button.on_click(lambda b, t=row['Ticker'], p=i: add_ticker_to_portfolio(t, p))
                    add_buttons.append(button)

                rows.append(HBox([
                    widgets.Label(row['Ticker'], layout=widgets.Layout(width='150px')),
                    widgets.Label(row['Name'], layout=widgets.Layout(width='300px')),
                    HBox(add_buttons)
                ]))
            display(VBox(rows))
        else:
            display(widgets.Label("No matching results found."))

search_input.observe(on_search_input_change, names='value')

# Display Widgets
display(search_input, search_output, portfolio_output, HBox([add_portfolio_button, delete_portfolio_button]))

# Preload Portfolio 1
portfolio_count += 1
portfolio_widgets[portfolio_count] = create_portfolio_widgets(portfolio_count)
with portfolio_output:
    display(display_portfolio_widgets(portfolio_widgets[portfolio_count]))

#####################################################################################################################
# @title Confirm portfolio parameters
def get_portfolio_specs():
    portfolio_specs = {}
    num_portfolios = len(portfolio_widgets)
    for i in range(1, num_portfolios + 1):
        portfolio_specs[i] = {
            'tickers': re.split(r'\s*,\s*', portfolio_widgets[i]['tickers'].value),
            'allocation_type': portfolio_widgets[i]['allocation_type'].value,
            'name': portfolio_widgets[i]['name'].value,
            'min_weight': float(portfolio_widgets[i]['min_weight'].value),
            'max_weight': float(portfolio_widgets[i]['max_weight'].value),
            'custom_weight': portfolio_widgets[i]['custom_weight'].value,  # Get the value of the custom weight widget
            'rebal_freq': portfolio_widgets[i]['rebal_freq'].value}

        if portfolio_specs[i]['allocation_type'] == 'Custom weight':
            try:
                portfolio_specs[i]['custom_weight'] = [float(x) for x in portfolio_specs[i]['custom_weight'].split(',')]
            except ValueError:
                print("Invalid custom weight input. Please enter comma-separated values and ensure values sum to 1 (e.g 0.3,0.3,0.4)")
                # You might want to handle the error more gracefully here.

        if portfolio_specs[i]['rebal_freq'] == 'Annually':
            portfolio_specs[i]['rebal_months'] = [1]
        elif portfolio_specs[i]['rebal_freq'] == 'Semi-annually':
            portfolio_specs[i]['rebal_months'] = [1, 7]
        elif portfolio_specs[i]['rebal_freq'] == 'Quarterly':
            portfolio_specs[i]['rebal_months'] = [1, 4, 7, 10]
        elif portfolio_specs[i]['rebal_freq'] == 'Monthly':
            portfolio_specs[i]['rebal_months'] = list(range(1, 13))

    return portfolio_specs

def get_unique_tickers():
        all_tickers_set = set(bm_tickers)  # Start with benchmark tickers
        for portfolio_widgets_dict in portfolio_widgets.values():
            portfolio_tickers = re.split(r'\s*,\s*', portfolio_widgets_dict['tickers'].value)
            all_tickers_set.update(portfolio_tickers)  # Add portfolio tickers
        return list(all_tickers_set)

# Function to execute on button click
def on_download_data_and_confirm_param_button_clicked(b):
      global start_date, end_date, bm_tickers, all_tickers, backtest_offset_days, data_source_widget, earliest_start_date, portfolio_specs, df_portfolio_specs, all_port_ret, all_port_wgt, all_port_bt, ret_bm_filtered, constituent_summary, yearly_df, perf_summary, port_yearly_ret, relative_perf, df_ports_alloc_latest, df_ports_alloc_avg, port_turnover_df, df_forecast

      download_data(data_source_widget.value, start_date, end_date)
      portfolio_specs = get_portfolio_specs()
      df_portfolio_specs = pd.DataFrame(portfolio_specs).T

      # Get the unique tickers after the portfolio specifications are confirmed
      all_tickers = get_unique_tickers()
      tickers = [ticker for ticker in all_tickers if ticker not in bm_tickers]

      display(df_portfolio_specs)
      print('\n')
      check_custom_weight_sum(portfolio_specs)


# Define constraints function
def create_constraints(min_weight, max_weight):
    return ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'ineq', 'fun': lambda weights: weights - min_weight},
            {'type': 'ineq', 'fun': lambda weights: max_weight - weights})


def create_custom_weight_dict(portfolio_specs, portfolio_number):
  # Creates a dictionary with ticker as key and custom weight as value.
  portfolio = portfolio_specs[portfolio_number]
  tickers = portfolio['tickers']
  custom_weights = portfolio['custom_weight']

  if len(tickers) != len(custom_weights):
    raise ValueError("Number of tickers and custom weights must be equal.")

  custom_weight_dict = dict(zip(tickers, custom_weights))
  return custom_weight_dict

def check_custom_weight_sum(portfolio_specs):

  for portfolio_number, portfolio_data in portfolio_specs.items():
      if portfolio_data['allocation_type'] == 'Custom weight':
          custom_weights = portfolio_data['custom_weight']
          if np.isclose(np.sum(custom_weights), 1.0):
              print("Portfolio specifications confirmed! ✅")
          else:
              print(f"Portfolio {portfolio_number}: Custom weights do not sum to 1. Please amend.")


#####################################################################################################################

# Create a confirmation button
confirm_button = widgets.Button(description="Confirm Data and Portfolio Specs", layout=widgets.Layout(width='auto'))

# Attach the function to the button's on_click event
confirm_button.on_click(on_download_data_and_confirm_param_button_clicked)

# Display the button
print('\n')
print_arial_bold('Confirm data and portfolio parameters')
display(confirm_button)



#####################################################################################################################
# @title Run optimisation

optimization_output = widgets.Output()  # Create an output widget

def run_optimization():

    num_portfolios = len(portfolio_widgets)
    portfolio_specs = get_portfolio_specs()
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=5,  # Total number of portfolios
        description='Optimising:',
        bar_style='',
        style={'bar_color': 'navy'},
        orientation='horizontal'
    )
    # Display the progress bar
    with optimization_output:
      display(progress_bar)
    with redirect_stdout(open(os.devnull, 'w')):
      for i in range(1, num_portfolios + 1):
          portfolio = portfolio_specs[i]
          portfolio_name = portfolio['name']
          tickers = portfolio['tickers']
          allocation_type = portfolio['allocation_type']
          min_weight = portfolio['min_weight']
          max_weight = portfolio['max_weight']
          custom_weight = portfolio['max_weight']  # or portfolio['custom_weight'] if intended
          rebal_months = portfolio['rebal_months']

          # Create constraints for each portfolio based on the min and max weight
          constraints = create_constraints(min_weight, max_weight)

          # Calculate weights based on allocation type:
          if allocation_type == 'Max sharpe':
              globals()[f"{portfolio_name}_wgt"] = pc.max_sharpe_rebal_wgt(
                  ret_all[tickers],
                  start_date=earliest_start_date + timedelta(days=backtest_offset_days),
                  end_date=end_date,
                  constraints_dict=constraints,
                  shrink_covar=False,
                  rebal_months=rebal_months,
                  annualized=261,
                  halflife=3.5
              )
          elif allocation_type == 'Min vol':
              globals()[f"{portfolio_name}_wgt"] = pc.min_vol_rebal_wgt(
                  ret_all[tickers],
                  start_date=earliest_start_date + timedelta(days=backtest_offset_days),
                  end_date=end_date,
                  constraints_dict=constraints,
                  shrink_covar=False,
                  rebal_months=rebal_months,
                  annualized=261,
                  halflife=3.5
              )
          elif allocation_type == 'Risk parity':
              globals()[f"{portfolio_name}_wgt"] = rebal_wgt_riskparity(
                  ret_all[tickers],
                  start_date=earliest_start_date + timedelta(days=backtest_offset_days),
                  end_date=end_date,
                  rebal_months=rebal_months,
                  halflife=3.5,
                  annualized=261,
                  shrink_covar=False
              )

          elif allocation_type == 'Equal weight':
              globals()[f"{portfolio_name}_wgt"] = pc.equal_weight_rebal_wgt(
                  ret_all[tickers],
                  start_date=earliest_start_date + timedelta(days=backtest_offset_days),
                  end_date=end_date,
                  rebal_months=rebal_months,
              )

          elif allocation_type == 'Custom weight':
              globals()[f"{portfolio_name}_wgt"] = pc.fixed_weight_rebal_wgt(
                ret_all[tickers],
                start_date=earliest_start_date + timedelta(days=backtest_offset_days),
                end_date=end_date,
                fixed_weights=create_custom_weight_dict(portfolio_specs, i),
                rebal_months=[1],
          )
          # Calculate performance
          globals()[f"{portfolio_name}_perf"] = pc.calc_port_perf(ret_all.loc[earliest_start_date + timedelta(days=backtest_offset_days):end_date][tickers],
              globals()[f"{portfolio_name}_wgt"],
              name_prefix=portfolio_name
          )

          # Update the progress bar
          progress_bar.value += 1

    optimization_output.clear_output(wait=False)  # Clear the output widget after completion
    print("Optimisation complete!")

# Create the button and link it to the function
print('\n')
print_arial_bold('Run optimisation')

run_optimisation_button = widgets.Button(description="Run Optimisation")
run_optimisation_button.on_click(lambda b: run_optimization())  # Call the function on click
display(run_optimisation_button)
display(optimization_output)


for i in range(1, len(portfolio_widgets) + 1):
    portfolio = portfolio_specs[i]
    portfolio_name = portfolio['name']
    print(portfolio_name)
    pc.display_rebal_wgt(globals()[f"{portfolio_name}_wgt"])


import ipywidgets as widgets
from IPython.display import display, clear_output

def generate_constituents_key_metrics():

    tab_names = [
        "Key Summary", 
        "Calendar Year Returns", 
        "Monthly Returns", 
        "Cumulative Returns", 
        "Return/Risk", 
        "Returns Distribution", 
        "Volatility", 
        "Drawdown", 
        "Correlation"
    ]

    # --- create output areas for each tab ---
    tab_contents = [widgets.Output() for _ in tab_names]

    # --- create tab widget ---
    t = widgets.Tab(children=tab_contents)

    # assign tab names
    for i, name in enumerate(tab_names):
        t.set_title(i, name)

    # ---------- TAB 0 ----------
    with tab_contents[0]:
        clear_output()
        display(constituent_summary.T)

    # ---------- TAB 1 ----------
    with tab_contents[1]:
        clear_output()
        print('Date range:', ret_all.index.min(), '-', ret_all.index.max(), '\n')
        display(yearly_df.pipe(pc.apply_style_heatmap_ret))

    # ---------- TAB 2 ----------
    with tab_contents[2]:
        clear_output()
        for col in ret_all.columns:
            security_returns = ret_all[[col]]
            monthly_performance = pc.monthly_performance_table(security_returns)
            print(col)
            display(monthly_performance.pipe(pc.apply_style_heatmap_ret))
            print('\n')

    # ---------- TAB 3 ----------
    with tab_contents[3]:
        clear_output()
        display(pc.plot_cumulative_returns(ret_all, show_data=True))

    # ---------- TAB 4 ----------
    with tab_contents[4]:
        clear_output()
        display(pc.plot_return_risk_scatter_maxrange(ret_all))

    # ---------- TAB 5 ----------
    with tab_contents[5]:
        clear_output()
        pc.plot_returns_distribution_boxplot(ret_all)

    # ---------- TAB 6 ----------
    with tab_contents[6]:
        clear_output()
        pc.plot_rolling_volatility(ret_all, window=261, title='Rolling 1Y Volatility')

    # ---------- TAB 7 ----------
    with tab_contents[7]:
        clear_output()
        display(pc.plot_drawdowns(ret_all, show_data=True))

    # ---------- TAB 8 ----------
    with tab_contents[8]:
        clear_output()
        pc.plot_correlation_heatmap(ret_all)

    display(t)

generate_constituents_key_metrics()

# @title 5. Generate portfolio key metrics

########################################################################################################################

port_names = []  # Initialize an empty list to store names

# Iterate through the portfolio specifications
for portfolio_number, portfolio_data in portfolio_specs.items():
    # Extract the 'name' attribute and append it to the list
    port_names.append(portfolio_data['name'])

all_port_ret, all_port_wgt, all_port_bt = combine_backtest_data(port_names)
ret_bm_filtered = ret_bm.loc[earliest_start_date + timedelta(days=backtest_offset_days):]
ret_bm_filtered.iloc[0] = 0
all_port_ret = pd.concat([all_port_ret, ret_bm], axis=1).dropna()

########################################################################################################################

# perf summary
# print(start_date, end_date)
perf_summary = pc.performance_summary(all_port_ret[port_names], all_port_wgt, all_port_ret[bm_tickers[0]], benchmark_returns=ret_bm,
                                    start_date=earliest_start_date + timedelta(days=backtest_offset_days), end_date=end_date, frequency='daily', rebal_per_year=1)

# calendar year returns
port_yearly_ret = pc.constituents_calendar_year_returns(all_port_ret, frequency='daily')
relative_perf = pc.calculate_relative_per(all_port_ret[port_names], all_port_ret[bm_tickers[0]])
# latest and average allocation
df_ports_alloc_latest = pc.compare_port_alloc(all_port_wgt, port_names, latest_data=True)
df_ports_alloc_avg = pc.compare_port_alloc(all_port_wgt, port_names, latest_data=False)

# portfolio turnover
port_turnover_df = pd.DataFrame()

for portfolio in port_names:
    turnover = pc.portfolio_turnover(all_port_wgt[portfolio])
    port_turnover_df = pd.concat([port_turnover_df, turnover],axis=1)

port_turnover_df.columns = port_names

df_forecast = pc.forecast_portfolio(all_port_ret)

# def generate_key_metrics_func(b):
#     clear_output(wait=True)  # Clear previous output
#     generate_key_metrics()

def generate_key_metrics():
    # Define tab names
    tab_names = ["Key Summary", "Calendar Year Returns", "Monthly Returns", "Cumulative Returns", "Return/Risk", "Returns Distribution", "Volatility", "Portfolio Drawdown", "Correlation", "Forecast Simulation"]

    # Create TabBar
    t = jcc.CompatibleTabBar(tab_names)
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    # Populate tabs with content
    with t.output_to(0):  # Key Summary
        display(perf_summary)

    with t.output_to(1):  # Calendar Year Returns
        print('Calendar year returns')
        display(port_yearly_ret.pipe(pc.apply_style_heatmap_ret))
        print('\n')
        print('Relative calendar year returns')
        display(pc.apply_style_heatmap_ret(relative_perf, subset=relative_perf.columns[1:]))

    with t.output_to(2):  # Monthly Returns
      for port in port_names:
        portfolio_returns = all_port_ret[[port]]
        monthly_performance = pc.monthly_performance_table(portfolio_returns)
        print(port)
        display(monthly_performance.pipe(pc.apply_style_heatmap_ret))
        print('\n')

    with t.output_to(3):  # Cumulative Returns
        display(pc.plot_cumulative_returns(all_port_ret, show_data=True))

    with t.output_to(4):  # Return/risk scatter plot
        display(pc.plot_return_risk_scatter_maxrange(all_port_ret))

    with t.output_to(5):  # Returns Distribution
        pc.plot_returns_distribution_boxplot(all_port_ret)

    with t.output_to(6):  # Rolling volatility
        pc.plot_rolling_volatility(all_port_ret, window=261, title='Rolling 1Y Volatility')

    with t.output_to(7):  # Portfolio Drawdown
        display(pc.plot_drawdowns(all_port_ret, show_data=True))

    with t.output_to(8):  # Correlation
        pc.plot_correlation_heatmap(all_port_ret)
        pc.plot_rolling_correlation(all_port_ret[port_names], benchmark=all_port_ret[bm_tickers[0]], window=261, title='Rolling 1Y Correlation')

    with t.output_to(9):
        for i in port_names:
          pc.plot_forecast_simulation(df_forecast.filter(like=i))

    display(t)

# key_metrics_button = widgets.Button(description="Generate Key Metrics", layout=widgets.Layout(width='auto'))
# key_metrics_button.on_click(generate_key_metrics_func)
# display(key_metrics_button)
generate_key_metrics()

core_groups = ['Core Equity100', 'Core Growth', 'Core Balanced', 'Core Defensive', 'SRS Equity100']
specialised_groups = ['China Growth', 'Disruptive Technology', 'ESG & Clean Energy', 'Healthcare Innovation', 'Downside protected', 'Custom USD']
passive_income_groups = ['Income Enhance', 'Income Preserve', 'REIT 100', 'REIT Risk Managed', 'SRS Income Enhance', 'SRS Income Preserve']
reit_groups = ['REIT 100', 'REIT Risk Managed']
income_plus_groups = ['Income Enhance', 'Income Preserve', 'SRS Income Enhance', 'SRS Income Preserve']

custom_groups = ['Custom USD']
cm_groups = ['Cash SGD flexi', 'Cash USD flexi', 'Cash SGD guaranteed']

growth_groups = core_groups + specialised_groups
income_preservation_groups = passive_income_groups + cm_groups
all_syfe__portfolios = growth_groups + income_preservation_groups

syfe_port_mapping['L1_classification'] = np.where(syfe_port_mapping['internal_port_name'].isin(growth_groups), 'Growth', 'Income & Preservation')

L2_conditions = [
    syfe_port_mapping['internal_port_name'].isin(core_groups),
    syfe_port_mapping['internal_port_name'].isin(specialised_groups),
    syfe_port_mapping['internal_port_name'].isin(reit_groups),
    syfe_port_mapping['internal_port_name'].isin(income_plus_groups),
    syfe_port_mapping['internal_port_name'].isin(cm_groups),
]

labels = ['Core', 'Specialised', 'REITs', 'Income+', 'Cash / MMF']

# Assign L2 classification
syfe_port_mapping['L2_classification'] = np.select(L2_conditions, labels, default='Other')

# @title 6. Generate key holdings & allocations

# configure portfolio groups
core_groups = ['Core E100', 'Core Growth', 'Core Balanced', 'Core Defensive', 'SRS E100']
specialised_groups = ['China Growth', 'Disruptive Technology', 'ESG & Clean Energy', 'Healthcare Innovation', 'Downside protected', 'Custom USD']
passive_income_groups = ['Income Enhance', 'Income Preserve', 'REIT 100', 'REIT Risk Managed', 'SRS Income Enhance', 'SRS Income Preserve']
custom_groups = ['Custom USD']
cm_groups = ['Cash SGD flexi', 'Cash USD flexi', 'Cash SGD guaranteed']

growth_groups = core_groups + specialised_groups
income_preservation_groups = passive_income_groups + cm_groups
all_syfe__portfolios = growth_groups + income_preservation_groups

client_latest_alloc = df_client_portfolios[df_client_portfolios['status']=='ACTIVE'][['type', 'nav_in_sgd']]
client_latest_alloc['nav_in_sgd'] = client_latest_alloc['nav_in_sgd'].astype(str).str.replace(',', '').astype(float)
client_latest_alloc = pd.DataFrame(client_latest_alloc.groupby('type')['nav_in_sgd'].sum())
client_latest_alloc['Client Portfolio'] = client_latest_alloc['nav_in_sgd'] / client_latest_alloc['nav_in_sgd'].sum()
client_latest_alloc = client_latest_alloc[['Client Portfolio']]
client_latest_alloc = client_latest_alloc.rename(index=syfe_port_mapping.set_index('MB_port_name')['internal_port_name'])

client_latest_alloc_adj = client_latest_alloc.T
client_latest_alloc_adj['reb_flag'] = True

# split into growth and income portfolios
income_preservation_groups_cols = [group + '_wgt' for group in income_preservation_groups]
all_port_wgt_income_preservation = all_port_wgt.loc[all_port_wgt[(portfolio, 'reb_flag')] == True].head().iloc[-1:]
all_port_wgt_income_preservation = all_port_wgt_income_preservation.loc[:, [col for col in all_port_wgt_income_preservation.columns
                                                                            if col[1] in income_preservation_groups_cols + ['reb_flag']]]

all_port_wgt_growth = all_port_wgt.loc[all_port_wgt[(portfolio, 'reb_flag')] == True].head().iloc[-1:]
all_port_wgt_growth = all_port_wgt_growth.loc[:, [col for col in all_port_wgt_growth.columns
                                                  if col[1] not in list(all_port_wgt_income_preservation.columns.get_level_values(1)) or col[1] == 'reb_flag']]

# filter out Cash mgmt portfolios for country and sector exposure
# all_port_wgt_exCM = all_port_wgt.loc[all_port_wgt[(portfolio, 'reb_flag')] == True].head().iloc[-1:]
# all_port_wgt_exCM = all_port_wgt_exCM.loc[:, [col for col in all_port_wgt_exCM.columns if not col[1].startswith('Cash')]]

# asset class exposure
# assetclass_analysis_list = [pc.exposure_analysis(all_port_wgt[portfolio], df_asset_class, latest_data=True).rename(columns={0: portfolio}) for portfolio in port_names]
assetclass_analysis_growth = [pc.exposure_analysis(all_port_wgt_growth[portfolio], df_asset_class, latest_data=True).rename(columns={0: portfolio}) for portfolio in port_names]
assetclass_breakdown_growth = pd.concat(assetclass_analysis_growth, axis=1)
assetclass_breakdown_growth.columns = port_names
assetclass_breakdown_growth = pd.concat([df_asset_class[bm_tickers], assetclass_breakdown_growth], axis=1)
assetclass_breakdown_growth = (assetclass_breakdown_growth.loc[(assetclass_breakdown_growth != 0).any(axis=1)].sort_values(by=assetclass_breakdown_growth.columns[0], ascending=False))

assetclass_analysis_income_preservation = [pc.exposure_analysis(all_port_wgt_income_preservation[portfolio], df_asset_class, latest_data=True).rename(columns={0: portfolio}) for portfolio in port_names]
assetclass_breakdown_income_preservation = pd.concat(assetclass_analysis_income_preservation, axis=1)
assetclass_breakdown_income_preservation.columns = port_names
assetclass_breakdown_income_preservation = pd.concat([df_asset_class[bm_tickers], assetclass_breakdown_income_preservation], axis=1)
assetclass_breakdown_income_preservation = (assetclass_breakdown_income_preservation.loc[(assetclass_breakdown_income_preservation != 0).any(axis=1)].sort_values(by=assetclass_breakdown_income_preservation.columns[0], ascending=False))

# country exposure
# country_analysis_list = [pc.exposure_analysis(all_port_wgt[portfolio], df_country, latest_data=True).rename(columns={0: portfolio}) for portfolio in port_names]
country_analysis_growth = [pc.exposure_analysis(all_port_wgt_growth[portfolio], df_country, latest_data=True).rename(columns={0: portfolio}) for portfolio in port_names]
country_breakdown_growth = pd.concat(country_analysis_growth, axis=1)
country_breakdown_growth.columns = port_names
country_breakdown_growth = pd.concat([df_country[bm_tickers], country_breakdown_growth], axis=1)
country_breakdown_growth = (country_breakdown_growth.loc[(country_breakdown_growth != 0).any(axis=1)].sort_values(by=country_breakdown_growth.columns[0], ascending=False))
country_breakdown_growth.pipe(pc.apply_style_heatmap).pipe(pc.apply_2dp_percentage)


# sector exposure
# sector_analysis_list = [pc.exposure_analysis(all_port_wgt[portfolio], df_sector, latest_data=True).rename(columns={0: portfolio}) for portfolio in port_names]
if all_port_wgt_growth.empty == False:
  sector_analysis_growth = [pc.exposure_analysis(all_port_wgt_growth[portfolio], df_sector, latest_data=True).rename(columns={0: portfolio}) for portfolio in port_names]
  sector_breakdown_growth = pd.concat(sector_analysis_growth, axis=1)
  sector_breakdown_growth.columns = port_names
  sector_breakdown_growth = pd.concat([df_sector[bm_tickers], sector_breakdown_growth], axis=1)
  sector_breakdown_growth = (sector_breakdown_growth.loc[(sector_breakdown_growth != 0).any(axis=1)].sort_values(by=sector_breakdown_growth.columns[0], ascending=False))
  sector_breakdown_growth.pipe(pc.apply_style_heatmap).pipe(pc.apply_2dp_percentage)

if all_port_wgt_income_preservation.empty == False:
    sector_analysis_income_preservation = [pc.exposure_analysis(all_port_wgt_income_preservation[portfolio], df_sector, latest_data=True).rename(columns={0: portfolio}) for portfolio in port_names]
    sector_breakdown_income_preservation = pd.concat(sector_analysis_income_preservation, axis=1)
    sector_breakdown_income_preservation.columns = port_names
    sector_breakdown_income_preservation = pd.concat([df_sector[bm_tickers], sector_breakdown_income_preservation], axis=1)
    sector_breakdown_income_preservation = (sector_breakdown_income_preservation.loc[(sector_breakdown_income_preservation != 0).any(axis=1)].sort_values(by=sector_breakdown_income_preservation.columns[0], ascending=False))
    sector_breakdown_income_preservation.pipe(pc.apply_style_heatmap).pipe(pc.apply_2dp_percentage)

def generate_key_holdings():
    # Define tab names
    tab_names = ["Latest Allocation", "Average Allocation", "Historical Allocation", "Asset class exposure (Growth)", "Country exposure (Growth)", "Sector exposure (Growth)"]

    # Create TabBar
    t = jcc.CompatibleTabBar(tab_names)

    # Populate tabs with content
    with t.output_to(tab_names[0]):  # Latest Allocation
        df_ports_alloc_latest.index = df_ports_alloc_latest.index.str.replace('_wgt', '')
        display(df_ports_alloc_latest.pipe(pc.apply_style_heatmap).pipe(pc.apply_2dp_percentage))

    with t.output_to(tab_names[1]):  # Average Allocation
        df_ports_alloc_avg.index = df_ports_alloc_avg.index.str.replace('_wgt', '')
        display(df_ports_alloc_avg.pipe(pc.apply_style_heatmap).pipe(pc.apply_2dp_percentage))

    with t.output_to(tab_names[2]):  # Portfolio Allocation
      for port in port_names:
        print(port)
        pc.plot_weights_and_turnover(all_port_wgt[port], show_data=True, show_turnover=False, show_rebal=True)

    with t.output_to(tab_names[3]):  # Asset class exposure
        display(assetclass_breakdown_growth.pipe(pc.apply_style_heatmap).pipe(pc.apply_2dp_percentage))

    with t.output_to(tab_names[4]):  # Country exposure
        display(country_breakdown_growth.pipe(pc.apply_style_heatmap).pipe(pc.apply_2dp_percentage))

    with t.output_to(tab_names[5]):  # Sector xposure
        display(sector_breakdown_growth.pipe(pc.apply_style_heatmap).pipe(pc.apply_2dp_percentage))

    display(t)

# key_holdings_button = widgets.Button(description="Generate Key Holdings & Allocations", layout=widgets.Layout(width='auto'))
# key_holdings_button.on_click(generate_key_holdings_func)
# display(key_holdings_button)
generate_key_holdings()

# @title 7. Fixed income metrics - duration, YTM, dividend yield, credit quality

# calc dividend of all constituents
dvd_breakdown_list = [pc.exposure_analysis(all_port_wgt[portfolio], df_fi_metrics.loc[['T12M dividend yield']].astype(float), latest_data=True).rename(columns={0: portfolio}) for portfolio in port_names]
dvd_breakdown_all = pd.concat(dvd_breakdown_list, axis=1)
dvd_breakdown_all.columns = port_names
dvd_breakdown_all = pd.concat([dvd_breakdown_all, df_fi_metrics.loc[['T12M dividend yield']][bm_tickers]], axis=1)
dvd_breakdown_all = round(dvd_breakdown_all, 2)


# Duration, YTM and credit quality of FI components only
fixed_income_list = df_fi_metrics.columns[df_fi_metrics.iloc[df_fi_metrics.index.get_loc('YTM / est. yield')] != 0].tolist()
port_fi_metrics = pd.DataFrame()

for i in (port_names):
    ytm_duration_df = pc.exposure_analysis_mixed(all_port_wgt[i], df_fi_metrics.loc[['Duration', 'YTM / est. yield']].astype(float), fixed_income_list, latest_data=True, start_date=start_date, end_date=end_date)
    ytm_duration_df.columns = [i]
    port_fi_metrics = pd.concat([port_fi_metrics, ytm_duration_df], axis=1)
    port_fi_metrics.loc['Credit rating'] = pc.calc_avg_credit_rating(all_port_wgt[i], df_fi_metrics.loc[['Credit rating']], latest_data=True)

port_fi_metrics = pd.concat([port_fi_metrics, df_fi_metrics.loc[['Duration', 'YTM / est. yield', 'Credit rating']][bm_tickers]], axis=1)
port_fi_metrics = pd.concat([port_fi_metrics, dvd_breakdown_all], axis=0)
port_fi_metrics

# @title Func sunburst charts
# Function to save charts to a single PDF

from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, BaseDocTemplate, Frame, PageTemplate, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm  # Import cm for specifying width in centimeters


########################################################################################################################

syfe_colors = ['#263159', '#2f51c9', '#879be3', '#bcbed7', '#dedfee','#fff2cc', '#ffe599', '#e3bf61', '#666666', '#7d839b', '#414e7d']

def convert_maturity_period(value):
    """Converts maturity period values to the desired format, handling non-string values."""
    if isinstance(value, str):  # Check if value is a string
        match = re.search(r'(\w+)_MONTHS', value)
        if match:
            period_num = {
                'THREE': '3',
                'SIX': '6',
                'TWELVE': '12'  # Add more mappings as needed
            }.get(match.group(1))
            if period_num:
                return f"{period_num} months"
    return value  # Return original value if not a string or no match

def plot_sunburst_chart_MP(df_allocation, title=None, width=1000, height=600):
    # Define groups
    core_groups = ['Core E100', 'Core Growth', 'Core Balanced', 'Core Defensive', 'SRS E100']
    specialised_groups = ['China Growth', 'Disruptive Tech', 'ESG & Clean Energy', 'Healthcare Innovation', 'Downside protected']
    passive_income_groups = ['Income Enhance', 'Income Preserve', 'REIT 100', 'REIT RM', 'SRS Income Enhance', 'SRS Income Preserve']
    custom_groups = ['CUSTOM_USD']

    # Filter out rows with allocation <= 0.0001 or NaN
    df_allocation = df_allocation[df_allocation != 0].dropna()

    # Categorize portfolios
    categories = []
    for portfolio in df_allocation.index:
        if portfolio in core_groups:
            categories.append('Core')
        elif portfolio in specialised_groups:
            categories.append('Specialised')
        elif portfolio in passive_income_groups:
            categories.append('Passive Income')
        elif portfolio in custom_groups:
            categories.append('Custom')
        else:
            categories.append('Other')

    # Dynamically create the list of unique categories based on the data
    unique_categories = sorted(set(categories))  # Only include categories present in the data
    labels = ['Portfolio Allocation'] + unique_categories + list(df_allocation.index)
    parents = [''] + ['Portfolio Allocation'] * len(unique_categories) + categories

    # Calculate values dynamically for existing categories
    category_sums = []
    for unique_cat in unique_categories:
        category_sum = df_allocation.loc[
            [portfolio for portfolio in df_allocation.index if
             (portfolio in core_groups and unique_cat == 'Core') or
             (portfolio in specialised_groups and unique_cat == 'Specialised') or
             (portfolio in custom_groups and unique_cat == 'Custom') or
             (portfolio in passive_income_groups and unique_cat == 'Passive Income')]
        ].sum()
        category_sums.append(category_sum)

    values = [df_allocation.sum()] + category_sums + list(df_allocation.values)

    # Filter out nodes where values <= 0
    filtered_data = [
        (label, parent, value) for label, parent, value in zip(labels, parents, values) if value > 0
    ]
    filtered_labels = [x[0] for x in filtered_data]
    filtered_parents = [x[1] for x in filtered_data]
    filtered_values = [x[2] for x in filtered_data]

    # Create the sunburst chart
    fig = go.Figure(go.Sunburst(
        labels=filtered_labels,
        parents=filtered_parents,
        values=filtered_values,
        branchvalues="total",
        textinfo="label+percent parent",  # Include labels and percent for all nodes
        insidetextorientation="horizontal",
    ))

    # Customize traces
    fig.update_traces(
        texttemplate=[
            "%{label}<br>%{percentRoot:.2%}" if parent != "" else ""
            for parent in filtered_parents
        ],
        outsidetextfont={"size": 12, "family": "Arial"},
        marker=dict(line=dict(width=0.5, color="white"))
    )

    # Update layout
    fig.update_layout(
        title_text=title,
        margin=dict(t=50, l=0, r=0, b=0),
        width=width,
        height=height,
        uniformtext=dict(minsize=10, mode='show')  # Uniform text size setting
    )
    return fig


def plot_sunburst_chart_MP_and_CM(df_allocation, title=None, width=1000, height=600):
    # Define groups
    core_groups = ['Core E100', 'Core Growth', 'Core Balanced', 'Core Defensive', 'SRS E100']
    specialised_groups = ['China Growth', 'Disruptive Tech', 'ESG & Clean Energy', 'Healthcare Innovation', 'Downside protected']
    passive_income_groups = ['Income Enhance', 'Income Preserve', 'REIT 100', 'REIT RM', 'SRS Income Enhance', 'SRS Income Preserve']
    custom_groups = ['CUSTOM_USD']

    mp_groups = core_groups + specialised_groups + passive_income_groups + custom_groups
    cm_groups = ['Cash SGD flexi', 'Cash USD flexi', 'Cash SGD guaranteed']

    # Filter out rows with allocation of 0 or NaN
    df_allocation = df_allocation[df_allocation != 0].dropna()

    # Categorize portfolios
    categories = []
    for portfolio in df_allocation.index:
        if portfolio in mp_groups:
            categories.append('Managed Portfolios')
        elif portfolio in cm_groups:
            categories.append('Cash Management')
        else:
            categories.append('Other')

    # Create labels, parents, and values
    unique_categories = ['Managed Portfolios', 'Cash Management']
    labels = ['Portfolio Allocation'] + unique_categories + list(df_allocation.index)
    parents = [''] + ['Portfolio Allocation'] * len(unique_categories) + categories

    # Corrected calculation of values
    category_sums = []
    for unique_cat in unique_categories:
        category_sum = df_allocation.loc[[portfolio for portfolio in df_allocation.index if portfolio in mp_groups and unique_cat == 'Managed Portfolios' or \
                                          portfolio in cm_groups and unique_cat == 'Cash Management']].sum()
        category_sums.append(category_sum)

    values = [df_allocation.sum()] + category_sums + list(df_allocation.values)

    # Create the sunburst chart
    fig = go.Figure(go.Sunburst(
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total",
    textinfo="label+percent parent",  # Include labels and percent for all nodes
    insidetextorientation="horizontal",))

    # Customize traces to hide the root node label
    fig.update_traces(
    texttemplate=[
        "%{label}<br>%{percentRoot:.2%}" if parent != "" else ""
        for parent in parents
    ],
    outsidetextfont={"size": 12},
    marker=dict(line=dict(width=0.5, color="white"))
    )

    fig.update_layout(
        title_text=title,
        margin=dict(t=50, l=0, r=0, b=0),
        width=width,
        height=height,
        uniformtext=dict(minsize=10, mode='show')  # Uniform text size setting
    )
    return fig

def plot_sunburst_chart_growth(df_allocation, title=None, width=1000, height=600):
    # Define groups
    core_groups = ['Core E100', 'Core Growth', 'Core Balanced', 'Core Defensive', 'SRS E100']
    specialised_groups = ['China Growth', 'Disruptive Tech', 'ESG & Clean Energy', 'Healthcare Innovation', 'Downside protected', 'Custom USD']

    # Filter out rows with allocation of 0 or NaN
    df_allocation = df_allocation[df_allocation != 0].dropna()

    # Categorize portfolios
    categories = []
    for portfolio in df_allocation.index:
        if portfolio in core_groups:
            categories.append('Core')
        elif portfolio in specialised_groups:
            categories.append('Specialised')
        else:
            categories.append('Other')

    # Create labels, parents, and values
    unique_categories = ['Core', 'Specialised']
    labels = ['Portfolio Allocation'] + unique_categories + list(df_allocation.index)
    parents = [''] + ['Portfolio Allocation'] * len(unique_categories) + categories

    # Calculate category sums
    category_sums = []
    for unique_cat in unique_categories:
        category_sum = df_allocation.loc[[portfolio for portfolio in df_allocation.index if portfolio in core_groups and unique_cat == 'Core' or \
                                          portfolio in specialised_groups and unique_cat == 'Specialised']].sum()
        category_sums.append(category_sum)

    values = [df_allocation.sum()] + category_sums + list(df_allocation.values)

    # Filter nodes with values > 0
    filtered_data = [
        (label, parent, value) for label, parent, value in zip(labels, parents, values) if value > 0
    ]
    filtered_labels = [x[0] for x in filtered_data]
    filtered_parents = [x[1] for x in filtered_data]
    filtered_values = [x[2] for x in filtered_data]

    # Create the sunburst chart
    fig = go.Figure(go.Sunburst(
        labels=filtered_labels,
        parents=filtered_parents,
        values=filtered_values,
        branchvalues="total",
        textinfo="label+percent parent",  # Include labels and percent for all nodes
        insidetextorientation="horizontal",
    ))

    # Customize traces to hide the root node label
    fig.update_traces(
        texttemplate=[
            "%{label}<br>%{percentRoot:.2%}" if parent != "" else ""
            for parent in filtered_parents
        ],
        outsidetextfont={"size": 12},
        marker=dict(line=dict(width=0.5, color="white"))
    )

    # Update layout
    fig.update_layout(
        title_text=title,
        margin=dict(t=50, l=0, r=0, b=0),
        width=width,
        height=height,
        uniformtext=dict(minsize=10, mode='show')  # Uniform text size setting
    )
    return fig

def plot_sunburst_chart_income_preservation(df_allocation, title=None, width=1000, height=600):
    # Define groups
    income_groups = ['Income Enhance', 'Income Preserve', 'SRS Income Enhance', 'SRS Income Preserve']
    reits_groups = ['REIT 100', 'REIT RM']
    cm_groups = ['Cash SGD flexi', 'Cash USD flexi', 'Cash SGD guaranteed']

    # Filter out rows with allocation of 0 or NaN
    df_allocation = df_allocation[df_allocation != 0].dropna()

    # Categorize portfolios
    categories = []
    for portfolio in df_allocation.index:
        if portfolio in income_groups:
            categories.append('Bonds')
        elif portfolio in reits_groups:
            categories.append('REITs')
        elif portfolio in cm_groups:
            categories.append('Cash / MMF')
        else:
            categories.append('Other')

    # Create labels, parents, and values
    unique_categories = ['Bonds', 'REITs', 'Cash / MMF']
    labels = ['Portfolio Allocation'] + unique_categories + list(df_allocation.index)
    parents = [''] + ['Portfolio Allocation'] * len(unique_categories) + categories

    # Corrected calculation of values
    category_sums = []
    for unique_cat in unique_categories:
        category_sum = df_allocation.loc[[portfolio for portfolio in df_allocation.index if portfolio in income_groups and unique_cat == 'Bonds' or
                                          portfolio in reits_groups and unique_cat == 'REITs' or
                                          portfolio in cm_groups and unique_cat == 'Cash / MMF']].sum()
        category_sums.append(category_sum)

    values = [df_allocation.sum()] + category_sums + list(df_allocation.values)

    # Create the sunburst chart
    fig = go.Figure(go.Sunburst(
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total",
    textinfo="label+percent parent",  # Include labels and percent for all nodes
    insidetextorientation="horizontal",))

    # Customize traces to hide the root node label
    fig.update_traces(
    texttemplate=[
        "%{label}<br>%{percentRoot:.2%}" if parent != "" else ""
        for parent in parents
    ],
    outsidetextfont={"size": 12},
    marker=dict(line=dict(width=0.5, color="white"))
    )

    fig.update_layout(
        title_text=title,
        margin=dict(t=50, l=0, r=0, b=0),
        width=width,
        height=height,
        uniformtext=dict(minsize=10, mode='show')  # Uniform text size setting
    )
    return fig

def plot_sunburst_chart_growth_income(df_allocation, title=None, width=1000, height=600):
    # Define groups
    growth_groups = ['Core E100', 'Core Growth', 'Core Balanced', 'Core Defensive','SRS E100', 'China Growth','Disruptive Tech', 'ESG & Clean Energy', 'Healthcare Innovation', 'Downside protected', 'Custom USD']
    income_preservation_groups = ['Income Enhance', 'Income Preserve','REIT 100', 'REIT RM', 'SRS Income Enhance','SRS Income Preserve', 'Cash SGD flexi', 'Cash USD flexi', 'Cash SGD guaranteed']

    # Filter out rows with allocation of 0 or NaN
    df_allocation = df_allocation[df_allocation != 0].dropna()

    # Categorize portfolios
    categories = []
    for portfolio in df_allocation.index:
        if portfolio in growth_groups:
            categories.append('Growth')
        elif portfolio in income_preservation_groups:
            categories.append('Income & Preservation')
        else:
            categories.append('Other')

    # Create labels, parents, and values
    unique_categories = ['Growth', 'Income & Preservation']
    labels = ['Portfolio Allocation'] + unique_categories + list(df_allocation.index)
    parents = [''] + ['Portfolio Allocation'] * len(unique_categories) + categories

    # Corrected calculation of values
    category_sums = []
    for unique_cat in unique_categories:
        category_sum = df_allocation.loc[[portfolio for portfolio in df_allocation.index if portfolio in growth_groups and unique_cat == 'Growth' or
                                          portfolio in income_preservation_groups and unique_cat == 'Income & Preservation']].sum()
        category_sums.append(category_sum)

    values = [df_allocation.sum()] + category_sums + list(df_allocation.values)

    # Create the sunburst chart
    fig = go.Figure(go.Sunburst(
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total",
    textinfo="label+percent parent",  # Include labels and percent for all nodes
    insidetextorientation="horizontal",))

    # Customize traces to hide the root node label
    fig.update_traces(
    texttemplate=[
        "%{label}<br>%{percentRoot:.2%}" if parent != "" else ""
        for parent in parents
    ],
    outsidetextfont={"size": 12},
    marker=dict(line=dict(width=0.5, color="white"))
    )

    fig.update_layout(
        title_text=title,
        margin=dict(t=50, l=0, r=0, b=0),
        width=width,
        height=height,
        uniformtext=dict(minsize=10, mode='show')  # Uniform text size setting
    )
    return fig

mb_port_names_core = syfe_port_mapping[syfe_port_mapping['internal_port_name'].isin(core_groups)]['MB_port_name'].tolist()
mb_port_names_specialised = syfe_port_mapping[syfe_port_mapping['internal_port_name'].isin(specialised_groups)]['MB_port_name'].tolist()
mb_port_names_passive_income = syfe_port_mapping[syfe_port_mapping['internal_port_name'].isin(passive_income_groups)]['MB_port_name'].tolist()
mb_port_names_cm = syfe_port_mapping[syfe_port_mapping['internal_port_name'].isin(cm_groups)]['MB_port_name'].tolist()
mb_port_names_growth = syfe_port_mapping[syfe_port_mapping['internal_port_name'].isin(growth_groups)]['MB_port_name'].tolist()
mb_port_names_income_preservation = syfe_port_mapping[syfe_port_mapping['internal_port_name'].isin(income_preservation_groups)]['MB_port_name'].tolist()
mb_port_names_income_plus = ['INCOME_ENHANCE', 'INCOME_PRESERVE', 'SRS_INCOME_ENHANCE', 'SRS_INCOME_PRESERVE']
mb_port_names_reits = ['REIT', 'REIT_RISK_MANAGED']

order_of_ports = ['GLOBAL_EQUITY_100', 'CORE_GROWTH', 'CORE_BALANCED', 'CORE_DEFENSIVE', 'SRS_GLOBAL_EQUITY_100',
                  'DISRUPTIVE_TECHNOLOGY', 'ESG_AND_CLEAN_ENERGY', 'HEALTHCARE_INNOVATION', 'CHINA_GROWTH', 'DOWNSIDE_PROTECTED', 'CUSTOM_USD',
                  'INCOME_PRESERVE', 'INCOME_ENHANCE', 'SRS_INCOME_PRESERVE', 'SRS_INCOME_ENHANCE', 'REIT' ,'REIT_RISK_MANAGED',
                  'CASH_PLUS', 'CASH_PLUS_USD', 'CASH_PLUS_GUARANTEED_SGD']

# @title Portfolio summary table
for col in ['nav_in_sgd', 'nav_in_usd', 'pnl_inception', 'invested_amount', 'dividend_balance']:
    df_client_portfolios[col] = pd.to_numeric(df_client_portfolios[col].astype(str).str.replace(',', '', regex=True), errors='coerce')

df_client_portfolios = df_client_portfolios.sort_values(by='nav_in_sgd', ascending=False)
df_client_portfolios_to_show = df_client_portfolios[df_client_portfolios['status'] == 'ACTIVE']

# Select and format columns
cols_to_show_all = ['portfolio_id', 'type', 'activated_date_sgt', 'currency', 'nav_in_sgd', 'nav_in_usd', 'dividend_balance', 'maturity_period', 'fd_created_at_rate']
client_port_rename_cols_all = ['Portfolio ID', 'Portfolio Type', 'Activated date', 'Currency', 'NAV (SGD)', 'NAV (USD)', 'Dividend (SGD)','Maturity period', 'Guaranteed rate']
df_client_portfolios_to_show_all = df_client_portfolios_to_show[cols_to_show_all]
df_client_portfolios_to_show_all.columns = client_port_rename_cols_all

# Portfolio summary
all_portfolios_summary = df_client_portfolios_to_show_all.copy()
# all_portfolios_summary['Portfolio Classification'] = all_portfolios_summary['Portfolio Type'].apply(lambda x: 'Growth' if x in mb_port_names_growth else
#                                                                                                     'Income & Preservation' if x in mb_port_names_income_preservation else '')

all_portfolios_summary = all_portfolios_summary.merge(syfe_port_mapping, left_on='Portfolio Type', right_on='MB_port_name', how='left')


all_portfolios_summary['Allocation'] = all_portfolios_summary['NAV (SGD)'] / all_portfolios_summary['NAV (SGD)'].sum()

total_row_all = pd.DataFrame({col: [all_portfolios_summary[col].sum() if col in ['NAV (SGD)', 'NAV (USD)', 'Invested Amount', 'P&L since inception', 'Dividend ($)', 'Allocation'] else '']
                          for col in all_portfolios_summary.columns}, index=['Total'])
total_row_all.iloc[0, 0] = 'Total'

all_portfolios_summary[['NAV (SGD)', 'NAV (USD)']] = round(all_portfolios_summary[['NAV (SGD)', 'NAV (USD)']], 2)

growth_subtotal = all_portfolios_summary[all_portfolios_summary['Portfolio Type'].isin(mb_port_names_growth)].sum(numeric_only=True)
growth_subtotal['Portfolio ID'] = 'Growth Portfolios'

# Calculate subtotals for income_preservation_groups
income_pres_subtotal = all_portfolios_summary[all_portfolios_summary['Portfolio Type'].isin(mb_port_names_income_preservation)].sum(numeric_only=True)
income_pres_subtotal['Portfolio ID'] = 'Income & Preservation Portfolios'

# Concatenate subtotals and original DataFrame
# all_portfolios_summary = pd.concat([all_portfolios_summary, pd.DataFrame([growth_subtotal, income_pres_subtotal])]).sort_values(by=['Portfolio Classification'])
if total_row_all['NAV (SGD)'].values[0] > 0:  # Check if NAV (SGD) is greater than 0
    all_portfolios_summary_final = pd.concat([all_portfolios_summary, total_row_all])

all_portfolios_summary_final['Portfolio Type'] = pd.Categorical(
    all_portfolios_summary_final['Portfolio Type'],
    categories=order_of_ports,
    ordered=True
)

# Sort the DataFrame
all_portfolios_summary_final = all_portfolios_summary_final.sort_values(by='Portfolio Type').reset_index(drop=True)

growth_rows = all_portfolios_summary_final[all_portfolios_summary_final['L1_classification'] == 'Growth']
income_pres_rows = all_portfolios_summary_final[all_portfolios_summary_final['L1_classification'] == 'Income & Preservation']
other_rows = all_portfolios_summary_final[~all_portfolios_summary_final['L1_classification'].isin(['Growth', 'Income & Preservation'])]

all_portfolios_summary_final = pd.concat([growth_rows, growth_subtotal.to_frame().T, income_pres_rows, income_pres_subtotal.to_frame().T, other_rows])
all_portfolios_summary_final[['NAV (SGD)', 'NAV (USD)']] = all_portfolios_summary_final[['NAV (SGD)', 'NAV (USD)']].applymap(lambda x: "{:,.2f}".format(x) if isinstance(x, (int, float)) else x)
all_portfolios_summary_final['Allocation'] = (all_portfolios_summary_final['Allocation'] * 100).map('{:.2f}%'.format)

for col in all_portfolios_summary_final.columns:
    if pd.api.types.is_categorical_dtype(all_portfolios_summary_final[col]):
        # Add an empty string to the categories
        all_portfolios_summary_final[col] = all_portfolios_summary_final[col].cat.add_categories([''])
        # Fill NaN with the empty string
        all_portfolios_summary_final[col] = all_portfolios_summary_final[col].fillna('')

    elif all_portfolios_summary_final[col].dtype == 'object':  # For string columns
        all_portfolios_summary_final[col] = all_portfolios_summary_final[col].fillna('')

    else:  # For numeric columns
        all_portfolios_summary_final[col] = all_portfolios_summary_final[col].fillna(0)


all_portfolios_summary_final = all_portfolios_summary_final[['Portfolio ID', 'internal_port_name', 'Activated date', 'Currency', 'NAV (SGD)', 'NAV (USD)', 'Allocation']]
all_portfolios_summary_final.rename(columns={'internal_port_name': 'Portfolio Type'}, inplace=True)

# @title Growth portfolio table
#########################################################################################################################################################################
# Separate the dataframe based on the 'type' column
growth_portfolios = all_portfolios_summary[all_portfolios_summary['L1_classification']=='Growth']
growth_portfolios['Allocation'] = growth_portfolios['NAV (SGD)'] / growth_portfolios['NAV (SGD)'].sum()

core_subtotal = growth_portfolios[growth_portfolios['L2_classification']=='Core'].sum(numeric_only=True)
core_subtotal['Portfolio ID'] = 'Core'
specialised_subtotal = growth_portfolios[growth_portfolios['L2_classification']=='Specialised'].sum(numeric_only=True)
specialised_subtotal['Portfolio ID'] = 'Specialised'

total_row_growth = pd.DataFrame({col: [growth_portfolios[col].sum() if col in ['NAV (SGD)', 'NAV (USD)', 'Invested Amount', 'P&L since inception', 'Dividend (SGD)', 'Allocation'] else '']
                          for col in growth_portfolios.columns}, index=['Total'])

total_row_growth.iloc[0, 0] = 'Total'

if total_row_growth['NAV (SGD)'].values[0] > 0:  # Check if NAV (SGD) is greater than 0
    growth_portfolios = pd.concat([growth_portfolios, total_row_growth])

growth_portfolios['Portfolio Type'] = pd.Categorical(
    growth_portfolios['Portfolio Type'],
    categories=order_of_ports,
    ordered=True
)

# Sort the DataFrame
growth_portfolios = growth_portfolios.sort_values(by='Portfolio Type').reset_index(drop=True)
core_rows = growth_portfolios[growth_portfolios['L2_classification'] == 'Core']
specialised_rows = growth_portfolios[growth_portfolios['L2_classification'] == 'Specialised']
other_rows = growth_portfolios[~growth_portfolios['L2_classification'].isin(['Core', 'Specialised'])]

growth_portfolios = pd.concat([core_rows, core_subtotal.to_frame().T, specialised_rows, specialised_subtotal.to_frame().T, other_rows])
growth_portfolios[['NAV (SGD)', 'NAV (USD)']] = growth_portfolios[['NAV (SGD)', 'NAV (USD)']].applymap(lambda x: "{:,.2f}".format(x) if isinstance(x, (int, float)) else x)
growth_portfolios['Allocation'] = (growth_portfolios['Allocation'] * 100).map('{:.2f}%'.format)

for col in growth_portfolios.columns:
    if pd.api.types.is_categorical_dtype(growth_portfolios[col]):
        # Add an empty string to the categories
        growth_portfolios[col] = growth_portfolios[col].cat.add_categories([''])
        # Fill NaN with the empty string
        growth_portfolios[col] = growth_portfolios[col].fillna('')

    elif growth_portfolios[col].dtype == 'object':  # For string columns
        growth_portfolios[col] = growth_portfolios[col].fillna('')

    else:  # For numeric columns
        growth_portfolios[col] = growth_portfolios[col].fillna(0)

growth_portfolios = growth_portfolios[['Portfolio ID', 'internal_port_name',  'Currency', 'NAV (SGD)', 'NAV (USD)',  'Allocation']]
growth_portfolios.rename(columns={'internal_port_name': 'Portfolio Type'}, inplace=True)

# @title Income & Preservation portfolio table
#########################################################################################################################################################################
income_preservation_fi_metrics = df_fi_metrics[[ticker for ticker in df_fi_metrics.columns if ticker in income_preservation_groups]]
# Create a dictionary for ticker to MB_port_name mapping
ticker_to_mb_mapping = dict(zip(syfe_port_mapping['internal_port_name'], syfe_port_mapping['MB_port_name']))
income_preservation_fi_metrics = income_preservation_fi_metrics.rename(columns=ticker_to_mb_mapping)
income_preservation_fi_metrics.T[['YTM / est. yield']]

income_pres_portfolios = all_portfolios_summary[all_portfolios_summary['L1_classification']=='Income & Preservation']
income_pres_portfolios = pd.merge(income_pres_portfolios, income_preservation_fi_metrics.T[['YTM / est. yield']], left_on='Portfolio Type', right_index=True, how='left')
income_pres_portfolios['Current Yield (%)'] = income_pres_portfolios.apply(lambda row: row['YTM / est. yield'] if row['Portfolio Type'] != 'CASH_PLUS_GUARANTEED_SGD' else row['Guaranteed rate'], axis=1)
income_pres_portfolios = income_pres_portfolios.drop(columns=['YTM / est. yield', 'Guaranteed rate'])

def calculate_subtotal_income_preservation(df, classification):
    # Filter data for the given classification
    subset = df[df['L2_classification'] == classification]

    # Sum relevant numeric columns
    subtotal = subset[['NAV (SGD)', 'NAV (USD)', 'Dividend (SGD)', 'Allocation']].sum()

    # Assign Portfolio ID
    subtotal['Portfolio ID'] = classification

    # Calculate weighted average for Current Yield (%)
    allocation_sum = subset['Allocation'].sum()

    if allocation_sum != 0:
        weighted_yield = (subset['Current Yield (%)'] * subset['Allocation']).sum() / allocation_sum
    else:
        weighted_yield = 0  # Avoid division by zero

    # Add weighted Current Yield (%) to the subtotal
    subtotal['Current Yield (%)'] = weighted_yield

    return subtotal

# Ensure correct data types
income_pres_portfolios['Allocation'] = pd.to_numeric(income_pres_portfolios['NAV (SGD)'], errors='coerce') / income_pres_portfolios['NAV (SGD)'].sum()
income_pres_portfolios['Current Yield (%)'] = pd.to_numeric(income_pres_portfolios['Current Yield (%)'], errors='coerce')
income_pres_portfolios['Allocation'] = pd.to_numeric(income_pres_portfolios['Allocation'], errors='coerce')

# Calculate subtotals
income_plus_subtotal = calculate_subtotal_income_preservation(income_pres_portfolios, 'Income+')
reits_subtotal = calculate_subtotal_income_preservation(income_pres_portfolios, 'REITs')
cash_mmf_subtotal = calculate_subtotal_income_preservation(income_pres_portfolios, 'Cash / MMF')

income_pres_portfolios['Current Yield (%)'] = pd.to_numeric(income_pres_portfolios['Current Yield (%)'], errors='coerce')

total_row_income_pres = pd.DataFrame({
    col: [
        # Sum numeric columns
        income_pres_portfolios[col].sum() if col in ['NAV (SGD)', 'NAV (USD)', 'Invested Amount', 'P&L since inception', 'Dividend (SGD)', 'Allocation'] else
        # Calculate weighted average for Yield (%) column safely
        ((income_pres_portfolios[col] * income_pres_portfolios['Allocation']).sum() / income_pres_portfolios['Allocation'].sum())
        if col == 'Current Yield (%)' and income_pres_portfolios['Allocation'].sum() != 0 else
        ''  # Keep other columns empty
    ] for col in income_pres_portfolios.columns
}, index=['Total'])

total_row_income_pres.iloc[0, 0] = 'Total'

if total_row_income_pres['NAV (SGD)'].values[0] > 0:  # Check if NAV (SGD) is greater than 0
    income_pres_portfolios = pd.concat([income_pres_portfolios, total_row_income_pres])

income_pres_portfolios['Portfolio Type'] = pd.Categorical(
    income_pres_portfolios['Portfolio Type'],
    categories=order_of_ports,
    ordered=True
)

# Sort the DataFrame
income_pres_portfolios = income_pres_portfolios.sort_values(by='Portfolio Type').reset_index(drop=True)
income_plus_rows = income_pres_portfolios[income_pres_portfolios['L2_classification'] == 'Income+']
reits_rows = income_pres_portfolios[income_pres_portfolios['L2_classification'] == 'REITs']
cm_rows = income_pres_portfolios[income_pres_portfolios['L2_classification'] == 'Cash / MMF']
others_rows = income_pres_portfolios[~income_pres_portfolios['L2_classification'].isin(['Income+', 'REITs', 'Cash / MMF'])]

income_pres_portfolios = pd.concat([income_plus_rows, income_plus_subtotal.to_frame().T, reits_rows, reits_subtotal.to_frame().T, cm_rows, cash_mmf_subtotal.to_frame().T, total_row_income_pres])
income_pres_portfolios[['NAV (SGD)', 'NAV (USD)', 'Current Yield (%)', 'Dividend (SGD)']] = income_pres_portfolios[['NAV (SGD)', 'NAV (USD)', 'Current Yield (%)', 'Dividend (SGD)']].applymap(lambda x: "{:,.2f}".format(x) if isinstance(x, (int, float)) else x)
income_pres_portfolios['Allocation'] = (income_pres_portfolios['Allocation'] * 100).map('{:.2f}%'.format)

for col in income_pres_portfolios.columns:
    if pd.api.types.is_categorical_dtype(income_pres_portfolios[col]):
        # Add an empty string to the categories
        income_pres_portfolios[col] = income_pres_portfolios[col].cat.add_categories([''])
        # Fill NaN with the empty string
        income_pres_portfolios[col] = income_pres_portfolios[col].fillna('')

    elif income_pres_portfolios[col].dtype == 'object':  # For string columns
        income_pres_portfolios[col] = income_pres_portfolios[col].fillna('')

    else:  # For numeric columns
        income_pres_portfolios[col] = income_pres_portfolios[col].fillna(0)

income_pres_portfolios = income_pres_portfolios[['Portfolio ID', 'internal_port_name', 'Currency', 'NAV (SGD)', 'NAV (USD)', 'Dividend (SGD)', 'Current Yield (%)','Maturity period', 'Allocation']]
income_pres_portfolios.rename(columns={'internal_port_name': 'Portfolio Type', 'Dividend (SGD)':'Dividend (SGD)*', 'Current Yield (%)':'Current Yield (%)**'}, inplace=True)

# @title PDF Funcs
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image

def create_table(df, title):

    elements = []
    if df.empty:
        return elements
    else:
        table_data = [df.columns.tolist()]  # Add header row
        for _, row in df.iterrows():
            table_data.append(list(row))

        table = Table(table_data)
        table_style = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263159")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                ("TOPPADDING", (0, 0), (-1, 0), 5),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 2),
                ("TOPPADDING", (0, 1), (-1, -1), 2),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ]
        )

        for i in range(len(df)):  # Iterate using numerical index
              if df.iloc[i]['Portfolio ID'] in ['Growth Portfolios', 'Income & Preservation Portfolios', "Core", "Specialised", "Income+", "REITs", "Cash / MMF"]:
                  table_style.add("BACKGROUND", (0, i + 1), (-1, i + 1), colors.HexColor("#879be4"))

        table.setStyle(table_style)

        custom_title_style = ParagraphStyle(
            name="CustomTitle",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=12,  # Line height
            alignment=0,  # Left align
        )

        # Update the title line in your function
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(f"{title}", custom_title_style))
        elements.append(Spacer(1, 0.2 * inch))  # Reduced space after the title
        elements.append(table)

    return elements

def add_page_numbers(input_pdf, output_pdf, page_size=landscape(A4)):
    # Get dimensions for A4 landscape
    page_width, page_height = page_size

    # Read the input PDF
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    total_pages = len(reader.pages)  # Get total number of pages

    for page_num, page in enumerate(reader.pages, start=1):
        # Create a canvas for the page number
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=page_size)
        can.setFont("Helvetica", 8)

        # Adjust position for bottom-right corner
        page_number_text = f"Page {page_num} of {total_pages}"
        text_width = can.stringWidth(page_number_text, "Helvetica", 10)
        x_position = page_width - 20 - text_width  # Align text to the bottom-right corner
        y_position = 20
        can.drawString(x_position, y_position, page_number_text)

        can.save()

        # Merge the page number with the current PDF page
        packet.seek(0)
        page_overlay = PdfReader(packet)
        page.merge_page(page_overlay.pages[0])
        writer.add_page(page)

    # Save the output PDF with page numbers
    with open(output_pdf, "wb") as f:
        writer.write(f)

# @title PDF Funcs
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Image
import os

def create_pdf_with_table_and_charts(output_file, table_df, table_title, charts, col_width_manual=None):
    elements = []
    temp_images = []  # Track temporary image files for cleanup

    # Create PDF document
    doc = SimpleDocTemplate(
        output_file,
        pagesize=landscape(A4),
        leftMargin=0.4 * inch,  # Adjust left margin
        rightMargin=0.3 * inch,  # Adjust right margin
        topMargin=0.6 * inch,  # Adjust top margin
        bottomMargin=0.2 * inch  # Adjust bottom margin
    )

    # Calculate available width for table
    page_width = landscape(A4)[0]
    available_width = page_width - (doc.leftMargin + doc.rightMargin)

    # Create table elements
    table_elements = create_table(table_df, table_title)

    if table_elements:
        table = table_elements[-1]  # Last element is the table

        if isinstance(table, Table):
            # Calculate column widths dynamically based on text length
            text_lengths = table_df.applymap(lambda x: len(str(x))).sum(axis=0)
            total_text_length = sum(text_lengths)

            # Normalize column widths based on relative text length
            if col_width_manual is None:
              col_widths = [(available_width * (col_size / total_text_length)) for col_size in text_lengths]
              table._argW = col_widths  # Manually set column widths
            else:
              table._argW = col_width_manual

        elements.extend(table_elements)  # Add table elements
        elements.append(Spacer(1, 0.2 * inch))  # Space before charts

    # Chart settings
    num_charts = len(charts)
    cols_per_row = 2 if num_charts % 2 == 0 else 3  # Adjust layout

    chart_width = 3.5 * inch
    chart_height = 2.6 * inch

    chart_grid = []
    row = []

    for idx, chart in enumerate(charts):
        # Save chart to temporary image
        image_file = f"temp_chart_{idx}.png"
        chart.write_image(image_file, format="png", width=4000, height=3000)
        temp_images.append(image_file)

        # Add chart image to row
        img = Image(image_file, width=chart_width, height=chart_height)
        row.append(img)

        # Add row when cols_per_row charts are added or if it's the last chart
        if len(row) == cols_per_row or idx == len(charts) - 1:
            chart_grid.append(row)
            row = []

    # Create a table for charts with borders and adjusted colWidths
    for chart_row in chart_grid:
        col_widths = [available_width / len(chart_row)] * len(chart_row)  # Auto-fit chart columns
        chart_table = Table([chart_row], colWidths=col_widths)

        # Add border to chart table
        chart_table.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 0.1, colors.white),
        ]))
        elements.append(chart_table)
        elements.append(Spacer(1, 0.1 * inch))

    doc.build(elements)

    # Clean up temporary image files
    for image_file in temp_images:
        if os.path.exists(image_file):
            os.remove(image_file)


def merge_with_template(template_pdf_path, overlay_pdf_path, output_pdf_path):
    """Merges multiple overlay pages onto a single-page template, repeating the template as needed."""

    template_pdf = PdfReader(template_pdf_path)
    overlay_pdf = PdfReader(overlay_pdf_path)
    output_pdf = PdfWriter()

    template_page = template_pdf.pages[0]  # Since the template has only 1 page

    for page_num in range(len(overlay_pdf.pages)):
        overlay_page = overlay_pdf.pages[page_num]

        # Copy the template so it doesn't get modified
        template_copy = template_page.create_blank_page(
            width=template_page.mediabox[2], height=template_page.mediabox[3]
        )
        template_copy.merge_page(template_page)  # Copy original template
        template_copy.merge_page(overlay_page)  # Overlay the new content

        output_pdf.add_page(template_copy)

    # Write final merged PDF
    with open(output_pdf_path, "wb") as final_pdf:
        output_pdf.write(final_pdf)

# @title Generate PDF

client_id = 'Svava'

template_pdf = PyPDF2.PdfReader(data_path + 'Portfolio_Visualiser_Template_Landscape.pdf')


if client_latest_alloc[i][client_latest_alloc[i].index.isin(growth_groups)].empty == False:
  growth_breakdown_chart = plot_sunburst_chart_growth(client_latest_alloc[i][client_latest_alloc[i].index.isin(growth_groups)], title='Growth Allocation')

if client_latest_alloc[i][client_latest_alloc[i].index.isin(income_preservation_groups)].empty == False:
  income_pres_breakdown_chart = plot_sunburst_chart_income_preservation(client_latest_alloc[i][client_latest_alloc[i].index.isin(income_preservation_groups)],
                                                              title='Income & Preservation Allocation')
if client_latest_alloc[i][client_latest_alloc[i].index.isin(income_preservation_groups)].empty == False and client_latest_alloc[i][client_latest_alloc[i].index.isin(growth_groups)].empty == False:
  growth_income_breakdown_chart = plot_sunburst_chart_growth_income(client_latest_alloc[i], title='Growth & Income Allocation')

top_asset_class_chart_growth = plot_pie_chart_topX(assetclass_breakdown_growth[i], title='Asset Class Exposure', colors=syfe_colors)
top_asset_class_chart_income_preservation = plot_pie_chart_topX(assetclass_breakdown_income_preservation[i], title='Asset Class Exposure', colors=syfe_colors)

top_country_chart_growth = plot_bar_chart_topX(country_breakdown_growth[i], title="Country Exposure",  colors=syfe_colors)

top_sector_chart_growth = plot_bar_chart_topX(sector_breakdown_growth[i], title="Sector Exposure",  colors=syfe_colors)
top_sector_chart_income_preservation = plot_bar_chart_topX(sector_breakdown_income_preservation[i], title="Sector Exposure", colors=syfe_colors)

# CREATE INDIVIDUAL PDF & MERGE PDFS
# Assuming you have your table DataFrame, title, and chart variables
create_pdf_with_table_and_charts("portfolio_summary.pdf", all_portfolios_summary_final, "Portfolio Summary",
                                [])

create_pdf_with_table_and_charts("growth_portfolios.pdf", growth_portfolios, "Growth Portfolios",
                                [top_asset_class_chart_growth, top_country_chart_growth, top_sector_chart_growth])

create_pdf_with_table_and_charts("income_preservation_portfolios.pdf", income_pres_portfolios, "Income & Preservation Portfolios",
                                [top_asset_class_chart_income_preservation , top_sector_chart_income_preservation], col_width_manual=[93, 93, 45, 103, 103, 100, 90, 100, 55])

# Merge PDFs
appendix_pdf = PyPDF2.PdfReader(data_path + 'Portfolio_Visualiser_Appendix.pdf')

merger = PdfMerger()
merger.append("portfolio_summary.pdf")
merger.append("growth_portfolios.pdf")
merger.append("income_preservation_portfolios.pdf")
merger.write(str(client_id) + "_report_merged.pdf")

# Paths
template_pdf_path = os.path.join(data_path, "Portfolio_Visualiser_Template_Landscape.pdf")
cover_pdf_path = os.path.join(data_path, "Portfolio_Visualiser_Cover.pdf")
overlay_pdf_path = os.path.join('/content/Projects/Svava_report_merged.pdf')
output_pdf_path = os.path.join('/content/Projects/' + str(client_id) + '_report_final.pdf')

# Merge overlay onto the template PDF
merge_with_template(template_pdf_path, overlay_pdf_path, output_pdf_path)

merger = PdfMerger()
merger.append(output_pdf_path)
merger.append(appendix_pdf)
merger.write(output_pdf_path)

add_page_numbers(output_pdf_path, output_pdf_path)
merger = PdfMerger()
merger.append(cover_pdf_path)
merger.append(output_pdf_path)
merger.write(output_pdf_path)

files.download(output_pdf_path)

print(f"Final report generated: {output_pdf_path}")


# @title 9. Factor exposure

import requests
import zipfile
import io

ff_link = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip'

# Download the file
response = requests.get(ff_link, verify=False)

# Check if the request was successful
if response.status_code == 200:
    # Open the zip file from the downloaded content
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # List the files in the zip archive
        # zf.printdir()
        # Extract and read the CSV file (usually there is only one)
        with zf.open('F-F_Research_Data_5_Factors_2x3_daily.CSV') as file:
            df_ff_factor = pd.read_csv(file, skiprows=3,index_col=0)/100
            df_ff_factor.index = pd.to_datetime(df_ff_factor.index, format='%Y%m%d')

else:
    print("Failed to download the file. Status code:", response.status_code)


# Ensure alignment before performing factor regression
common_index = ret_all.index.intersection(df_ff_factor.index)
etf_factor = pc.factor_exposure_coefficient(ret_all.loc[common_index], df_ff_factor.loc[common_index], all_tickers)
display(etf_factor.style.background_gradient(cmap='Blues', subset=etf_factor.columns[:],axis=None))

# Filter all_port_ret to the common index as well for portfolio factor analysis
common_index_2 = all_port_ret.index.intersection(df_ff_factor.index)
all_port_ret_filtered = all_port_ret.loc[common_index_2]
port_factor = pc.factor_exposure_coefficient(all_port_ret_filtered, df_ff_factor.loc[common_index_2], port_names)
display(port_factor.style.background_gradient(cmap='Blues', subset=port_factor.columns[:],axis=None))

# @title 10. Input file name and export key data to excel

file_path_widget = widgets.Text(
    value='Portfolio_Visualizer_Output.xlsx',
    description='File Name:',
    disabled=False
)

display(file_path_widget) # Display the widget

# Function to save the data to Excel
def save_to_excel(file_path):
    with pd.ExcelWriter(file_path) as writer:
      constituent_summary.to_excel(writer, sheet_name='Constituents summary', index=True)
      yearly_df.to_excel(writer, sheet_name='Constituents calendar year returns', index=True)
      perf_summary.to_excel(writer, sheet_name='Perf summary', index=True)
      port_yearly_ret.to_excel(writer, sheet_name='Calendar year returns', index=True)
      df_ports_alloc_latest.pipe(pc.apply_style_heatmap).pipe(pc.apply_2dp_percentage).to_excel(writer, sheet_name='Latest alloc', index=True)
      df_ports_alloc_avg.pipe(pc.apply_style_heatmap).pipe(pc.apply_2dp_percentage).to_excel(writer, sheet_name='Average alloc', index=True)
      assetclass_breakdown_all.pipe(pc.apply_style_heatmap).pipe(pc.apply_2dp_percentage).to_excel(writer, sheet_name='Asset class exposure', index=True)
      country_breakdown_all.pipe(pc.apply_style_heatmap).pipe(pc.apply_2dp_percentage).to_excel(writer, sheet_name='Country exposure', index=True)
      sector_breakdown_all.pipe(pc.apply_style_heatmap).pipe(pc.apply_2dp_percentage).to_excel(writer, sheet_name='Sector exposure', index=True)
      etf_factor.sort_values(by='Mkt', ascending=False).style.background_gradient(cmap='Blues', subset=etf_factor.columns[:],axis=None).to_excel(writer, sheet_name='ETF factor', index=True)
      port_factor.sort_values(by='Mkt', ascending=False).style.background_gradient(cmap='Blues', subset=etf_factor.columns[:],axis=None).to_excel(writer, sheet_name='Portfolio factor', index=True)
      df_forecast.to_excel(writer, sheet_name='Forecast simulation', index=True)

      for n in port_names:
          all_port_wgt[n].to_excel(writer, sheet_name=n+'_hist_wgt', index=True)

      print(f"Excel file saved to: Projects/Projects/{file_path}")

# Create a button to trigger the save function
save_button = widgets.Button(description="Save as Excel")

# Define the button's on_click behavior
def on_save_button_clicked(b):
    save_to_excel(file_path_widget.value)

save_button.on_click(on_save_button_clicked)
display(save_button) # Display the button

