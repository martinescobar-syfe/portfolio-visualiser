import pandas as pd
import numpy as np
import datetime
import os
import math
from scipy.stats import norm

# CONSTANTS
TARGET_DATE = 20260216
DATA_PATH = './'
OUTPUT_PATH_CORE = "C:\Syfe\Quantitative Research\Performance Files\Output"
OUTPUT_PATH_THEMATIC = "C:\Syfe\Quantitative Research\Performance Files\Output"
OUTPUT_PATH_PORTFOLIO_TEMPLATES = "C:\Syfe\Quantitative Research\Performance Files\Output\Portfolio_Templates"
# TARGET_PORTFOLIO_PATH = os.path.join(DATA_PATH, 'target_portfolios')

FILE_MONTHLY_DATA = "C:\Syfe\Quantitative Research\Performance Files\monthly_update_data.xlsx"
FILE_GLOBAL_DATA = "C:\Syfe\Quantitative Research\Performance Files\Global_data.xlsx" # contains dividend data and weekly return data that we need to update via port

# Ensure output directories exist
os.makedirs(OUTPUT_PATH_CORE, exist_ok=True)
os.makedirs(OUTPUT_PATH_THEMATIC, exist_ok=True)
os.makedirs(OUTPUT_PATH_PORTFOLIO_TEMPLATES, exist_ok=True)


class PortfolioConfig:
    """
    Configuration class for portfolio settings.

    It takes in the following parameters:
    - mode: 'CORE', 'THEMATIC', or 'PORTFOLIO_TEMPLATES'
    - portfolio_names: list of portfolio names (i.e. E100 DBG for Core and so on but generalisable to other portfolio TYPES like REIT100 or REIT RM etc)
    - return_sheet_name: name of the sheet in the monthly update data file (this is taken from Bloomberg via PORT weekly returns)
    - target_file_map: dictionary of target file paths (generated via our trading scripts)
    - single_target_file: single target file path
    - deterministic_countries: list of deterministic countries # It takes in a list of determinisitic countries,
        decided to leave that optionality inside the function in case we change our minds in the future
    """

    def __init__(self, mode, portfolio_names, return_sheet_name, target_file_map=None, single_target_file=None,
                 deterministic_countries=None, overrides=None):
        self.mode = mode
        self.portfolio_names = portfolio_names
        self.return_sheet_name = return_sheet_name
        self.target_file_map = target_file_map
        self.single_target_file = single_target_file
        self.deterministic_countries = deterministic_countries
        if overrides is not None: # Written this way intentionally to prevent Mutable Default Argument trap, None is intentional 
            self.overrides = overrides
        else:
            self.overrides = {}
 
CORE_CONFIG = PortfolioConfig(
    mode='CORE',
    portfolio_names=['Defensive', 'Balanced', 'Growth', 'E100'],
    return_sheet_name='Core_weekly_ret',
    target_file_map={
        'Defensive': "C:\Syfe\Core+\Target_Portfolio\Target_Portfolio_1.csv",
        'Balanced': "C:\Syfe\Core+\Target_Portfolio\Target_Portfolio_2.csv",
        'Growth': "C:\Syfe\Core+\Target_Portfolio\Target_Portfolio_3.csv",
        'E100': "C:\Syfe\GE100\Target_Portfolio\Target_Portfolio.csv"
    },
    deterministic_countries=[
        'United States', 'China', 'Germany', 'Japan', 'United Kingdom',
        'France', 'Italy', 'Canada', 'Switzerland'
    ],
    overrides={
        'GLDM': {'sector': ('Commodities', 1.0), 'country': ('Not Classified (commodities)', 1.0)},
        'KWEB': {'country': ('China', 1.0)},
        'MCHI': {'country': ('China', 1.0)},
    }
)

THEMATIC_CONFIG = PortfolioConfig(
    mode='THEMATIC',
    portfolio_names=['CHINA_GROWTH', 'DISRUPTIVE_TECHNOLOGY', 'ESG_AND_CLEAN_ENERGY', 'HEALTHCARE_INNOVATION'],
    return_sheet_name='Thenatic_weekly_ret',
    single_target_file= "C:\Syfe\Custom Portfolio\Thematic Portfolios\Target_Portfolio\Target_Portfolio.csv",
    overrides={
        'KWEB': {'country': ('China', 1.0)},
        'MCHI': {'country': ('China', 1.0)},
        'KBA': {'country': ('China', 1.0)},
        'KGRN': {'country': ('China', 1.0)},
        'KSTR': {'country': ('China', 1.0)},
        'KURE': {'country': ('China', 1.0)},
        'CXSE': {'country': ('China', 1.0)},
        'CHIQ': {'country': ('China', 1.0)},
    }
)

# Added as part of INV-309
PORTFOLIO_TEMPLATES_CONFIG = PortfolioConfig( 
    mode='PORTFOLIO_TEMPLATES',
    portfolio_names=['JPM_ACTIVE'],
    return_sheet_name='portfolio_template_weekly_ret',
    single_target_file=r"C:\Syfe\Portfolio Templates\Target Portfolio\Portfolio_Template_Target_Portfolio.csv",
    overrides={}
)

# INV-218: Significant change made wherein we decided to move away from hardcoding specific tickers and instead directly referenced from our
# target portfolio (which historically needs to be generated via our trading scripts) as our single source of truth.

def load_target_portfolio_series(filepath, date_val, theme_name=None):
    """
    Loads portfolio weights for a specific date from a CSV file, handling both standard
    time-series files and 'Thematic' master files with filtering.
    """

    if not os.path.exists(filepath):
        print(
            f"Portfolio file not found: {filepath}")  # save failing in the event the portfolio filepath isnt properly configured above
        return pd.Series(dtype=float)

    try:
        if theme_name:  # this checks whether we have a theme name, if we do, then
            # Thematic Logic
            df = pd.read_csv(filepath)
            df_subset = df[df['Theme'].astype(str).str.upper() == theme_name.upper()]  # make case insensitive

            row = df_subset[df_subset['Date'].astype(str) == str(date_val)]

            if row.empty:
                print(
                    f"WARNING: Date {date_val} not found for theme {theme_name} in {filepath}")  # this means the target portfolio hasn't been updated yet
                exit()

            data_row = row.iloc[0].drop(
                ['Date', 'Theme'])  # Extract weights. Need to drop 'Date' and 'Theme' and any 0 columns
            return pd.to_numeric(data_row, errors='coerce').fillna(0.0)

        else:
            # Core Logic (date index)
            df = pd.read_csv(filepath, index_col=0)
            if date_val not in df.index:
                if str(date_val) in df.index:  # an extra layer of check if it is a formatting issue
                    date_val = str(date_val)
                else:
                    print(
                        f"WARNING: Date {date_val} not found in {filepath}")  # this means the target portfolio hasn't been updated yet
                    exit()
            return df.loc[date_val]

    except Exception as e:
        print(f"WARNING: Error loading {filepath}: {e}")
        exit()


# Added exposure type so we can very easily adapt to any sort of exposure type (i.e. country / sector)
def calculate_weighted_exposure(raw_df, target_alloc_df, port_names, exposure_type='generic', overrides=None):
    """
    Calculates the aggregated weighted exposure of portfolios to specific categories (e.g., Sector, Country),
    applying manual overrides where necessary.

    The override logic works in two steps for tickers present in both the raw data and the override list:
    1. It subtracts the contribution calculated from `raw_df` (to remove potentially incorrect data).
    2. It adds the manual contribution defined in the `overrides` dictionary.
    """

    common_tickers = raw_df.columns.intersection(target_alloc_df.index)
    res = pd.DataFrame(0.0, index=raw_df.index, columns=port_names)

    if len(common_tickers) > 0:
        exposure_aligned = raw_df[common_tickers].fillna(0)
        exposure_aligned = exposure_aligned.apply(pd.to_numeric, errors='coerce').fillna(0)
        tgt_aligned = target_alloc_df.loc[common_tickers]

        calculated = pd.DataFrame(np.dot(exposure_aligned.values, tgt_aligned.values),
                                  index=exposure_aligned.index,
                                  columns=port_names)
        res = res.add(calculated, fill_value=0)

    # portfolio overrides
    overrides = overrides or {}

    for ticker, rules in overrides.items():
        # Checks if ticker is in the portfolio
        if ticker in target_alloc_df.index:
            weights = target_alloc_df.loc[ticker]
            # Checks that it has a non-zero weight
            if weights.sum() > 0:
                # Do we have a "rule" i.e. country/sector override. Did it in a nested dictionary to make it more readable and also
                # if we wanted to have more overrides in the future it would just be an extension of the inner dictionary
                if exposure_type in rules:
                    category, weight_pct = rules[exposure_type]
                    if ticker == 'GLDM':
                        pass

                    if category not in res.index:
                        new_row = pd.DataFrame(0.0, index=[category], columns=port_names)
                        res = pd.concat([res,
                                         new_row])  # Note: In newer pandas concat might behave differently with alignment if not careful

                    # We would have data from Bloomberg that needs to be overridden here
                    if ticker in common_tickers:
                        exp_val = exposure_aligned[ticker]
                        for col in port_names:
                            w = weights[col]
                            if w > 0:
                                res[col] = res[col] - (
                                            exp_val * w)  # we subtract the weighted contribution of the Bloomberg data (before overriding)

                    # Add the manual override exposure
                    for col in port_names:
                        w = weights[col]
                        if w > 0:
                            current_val = res.loc[category, col]
                            # Check if current_val is valid
                            if pd.isna(current_val):
                                current_val = 0.0

                            new_val = current_val + (weight_pct * w)
                            res.loc[category, col] = new_val

    return res


# Discussed in person with AP that we should just leave as is rather than changing it to Monte Carlo.
# FIXME: Long term we should change this to Monte Carlo but can weight the benefits before hand...
def run_forecast(weekly_ret, port_names, halflife=7):
    """
    Forecasting using Deterministic Analytical Method.
    Calculates 5th, 50th, and 95th percentiles.
    """

    decay = 1 - math.exp(np.log(0.5) / halflife / 52)

    # Ensure all port_names are in weekly_ret, added this check for thematic portfolios
    available_port_names = [p for p in port_names if p in weekly_ret.columns]
    weekly_nav = pd.DataFrame(0.0, columns=weekly_ret.columns,
                              index=weekly_ret.index)  # Use all columns for index alignment

    if len(weekly_nav) == 0:
        print("Weekly returns empty, cannot run forecast.")
        return pd.DataFrame()

    weekly_nav.iloc[0] = 100.0
    for i in range(1, len(weekly_ret)):
        weekly_nav.iloc[i] = weekly_nav.iloc[i - 1] * (1 + weekly_ret.iloc[i])

    weekly_geo_ret = np.log(weekly_nav / weekly_nav.shift(1))

    exp_wgt = pd.DataFrame(0.0, index=weekly_geo_ret.index, columns=['Wgt'])
    for i in range(len(exp_wgt) - 1, -1, -1):
        if i == len(exp_wgt) - 1:
            exp_wgt.iloc[i] = decay
        else:
            exp_wgt.iloc[i] = exp_wgt.iloc[i + 1] * (1 - decay)

    avg_ret_list = []

    for j in available_port_names:
        w_val = exp_wgt['Wgt'].values
        geo_val = weekly_geo_ret[j].fillna(0).values

        avg_annual = np.dot(geo_val, w_val) * 52 / w_val.sum()
        avg_ret_list.append(avg_annual)

    stdev_series = weekly_geo_ret[available_port_names].std() * np.sqrt(52)
    stdev_list = stdev_series.tolist()

    core_simulations = pd.DataFrame()
    months_projected = 376
    conf_interval = [0.05, 0.5, 0.95]  # Conservative, Likely, Optimistic

    print(f"Starting Deterministic Forecast for {available_port_names}")

    for num, name in enumerate(available_port_names):
        exp_rtn = avg_ret_list[num]
        stdev = stdev_list[num]

        df_sim = pd.DataFrame(index=range(months_projected))
        df_sim.index.name = f'{name}_Month'

        months = np.arange(months_projected)
        m_vec = months[1:]

        term_drift = (m_vec / 12) * exp_rtn
        term_vol_base = np.sqrt(m_vec / 12) * stdev

        conservative_vals = np.full(months_projected, 100.0)
        likely_vals = np.full(months_projected, 100.0)
        optimistic_vals = np.full(months_projected, 100.0)

        conservative_vals[1:] = np.exp(term_drift + term_vol_base * norm.ppf(conf_interval[0])) * 100
        likely_vals[1:] = np.exp(term_drift + term_vol_base * norm.ppf(conf_interval[1])) * 100
        optimistic_vals[1:] = np.exp(term_drift + term_vol_base * norm.ppf(conf_interval[2])) * 100

        # INV-218: Removed percentile tags seen in the old script, to make it clearer for user.
        df_sim[f'{name}_Conservative'] = conservative_vals
        df_sim[f'{name}_Likely'] = likely_vals
        df_sim[f'{name}_Optimistic'] = optimistic_vals

        core_simulations = pd.concat([core_simulations, df_sim], axis=1)

    print("Forecast Complete")
    return core_simulations


# TODO: long term is to get all portfolios exposures (and potentially returns) calculated via this script
def run_monthly_update(config):
    """
    Orchestrates the end-to-end monthly portfolio update process, generating a multi-sheet Excel report
    containing composition, exposures, yields, and look-through holdings.

    This function acts as the main pipeline controller. It loads portfolio weights based on the
    configured mode ('CORE', 'THEMATIC', or 'PORTFOLIO_TEMPLATES'), fetches external market data (Sector, Country,
    Underlying Holdings), performs weighted aggregation (see other functions above), and exports the results to excel.

    """

    print(f"Running {config.mode} Update for {TARGET_DATE}")

    portfolios = {}
    if config.mode == 'CORE':
        for name, path in config.target_file_map.items():
            portfolios[name] = load_target_portfolio_series(path, TARGET_DATE)
    elif config.mode in ('THEMATIC', 'PORTFOLIO_TEMPLATES'):
        for name in config.portfolio_names:
            portfolios[name] = load_target_portfolio_series(config.single_target_file, TARGET_DATE, theme_name=name)

    # Combine into a single alloc DataFrame
    all_tickers = sorted(list(set().union(*[p.index for p in portfolios.values() if not p.empty])))
    tgt_alloc = pd.DataFrame(0.0, index=all_tickers, columns=config.portfolio_names)

    for name, series in portfolios.items():
        if not series.empty:
            tgt_alloc.loc[series.index, name] = series.values

    # Remove rows with 0 sum (tickers not present in any active portfolio) i.e. only show that are present in at least one portfolio
    tgt_alloc = tgt_alloc[tgt_alloc.sum(axis=1) > 0]

    print(f"Combined Portfolio Loaded. Total unique tickers: {len(tgt_alloc)}")

    try:
        monthly_data = pd.ExcelFile(FILE_MONTHLY_DATA)
        global_data = pd.ExcelFile(FILE_GLOBAL_DATA)
        df_countries_raw = pd.read_excel(monthly_data, 'Country', index_col=0)

        df_sectors_raw = pd.read_excel(monthly_data, 'Sector', index_col=0)
        df_underlying = pd.read_excel(monthly_data, 'Underlying')
        df_div = pd.read_excel(global_data, 'Dividend', index_col=0)

        if config.return_sheet_name in global_data.sheet_names:
            df_returns = pd.read_excel(global_data, config.return_sheet_name, index_col=0)
        else:
            df_returns = pd.DataFrame()

    except Exception as e:
        print(f"Error loading data files: {e}")
        return

    # Calculate dividend yields
    common_tickers = df_div.index.intersection(tgt_alloc.index)
    div_yields = df_div.loc[
        common_tickers, 'EQY_DVD_YLD_12M']  # same methodology as our current Dividends calculation (just sumproduct of weights and dividends)
    tgt_aligned_div = tgt_alloc.loc[common_tickers]

    calc_div_yld = []
    for col in config.portfolio_names:
        if tgt_aligned_div[col].sum() == 0:
            calc_div_yld.append(0.0)
        else:
            port_yld = np.dot(div_yields.values, tgt_aligned_div[col].values)
            calc_div_yld.append(port_yld)

    df_div_yld_res = pd.DataFrame(calc_div_yld, index=config.portfolio_names, columns=['Dividend Yield']) / 100

    # run deterministic forecast per the old code
    df_sim_res = pd.DataFrame()
    if not df_returns.empty:
        df_sim_res = run_forecast(df_returns, config.portfolio_names)

    # calculate sector exposure
    df_sector = calculate_weighted_exposure(df_sectors_raw, tgt_alloc, config.portfolio_names, exposure_type='sector',
                                            overrides=config.overrides)
    for col in df_sector.columns:
        df_sector[col] = df_sector[col] * 100

    # noramlise
    df_sector = (df_sector / df_sector.sum()) * 100

    # TO CHANGE THIS SORTING LOGIC ONCE ENG DOUBLE CONFIRMS POST GT CHAT WITH THEM
    sort_by_col = config.portfolio_names[0] if config.portfolio_names[0] in df_sector.columns else df_sector.columns[
        0]  # 0 is defensive, sort by a portfolio name
    df_sector = df_sector.sort_values(by=sort_by_col, ascending=False)

    # country exposure
    df_country = calculate_weighted_exposure(df_countries_raw, tgt_alloc, config.portfolio_names,
                                             exposure_type='country', overrides=config.overrides)

    for col in df_country.columns:
        df_country[col] = df_country[col] * 100

    df_country = (df_country / df_country.sum()) * 100  # normalise

    # Country Exposure Processing
    # We generate two sets of side-by-side DataFrames:
    # 1. Main Country Sheet: Top 10 countries + "Others" aggregate row
    # 2. Country Extra Sheet: The breakdown of "Others" (Rank 11 onwards), also side-by-side

    top_n = 10
    side_by_side_dfs = []         # For Main Sheet
    side_by_side_extra_dfs = []   # For Extra Sheet

    for col in config.portfolio_names:
        if col in df_country.columns:
            # 1. Sort descending for this SPECIFIC portfolio
            sorted_series = df_country[col].sort_values(ascending=False)

            # 2. Filter out zero/negligible values
            sorted_series = sorted_series[sorted_series > 1e-6]

            # 3. Main Sheet Logic (Top 10 + Others)
            top_10 = sorted_series.head(top_n)
            
            if len(sorted_series) > top_n:
                others_val = sorted_series.iloc[top_n:].sum()
                others_series = pd.Series([others_val], index=['Others'])
                final_series_main = pd.concat([top_10, others_series])
                
                # 4. Extra Sheet Logic (The Rest / Tail)
                extra_series = sorted_series.iloc[top_n:]
                
            else:
                final_series_main = top_10
                extra_series = pd.Series(dtype=float)

            # Format Main Sheet DataFrame
            _df_main = final_series_main.reset_index()
            _df_main.columns = [f"{col} Country", col]
            side_by_side_dfs.append(_df_main)
            
            # Format Extra Sheet DataFrame
            if not extra_series.empty:
                _df_extra = extra_series.reset_index()
                _df_extra.columns = [f"{col} Country", col]
                side_by_side_extra_dfs.append(_df_extra)
            else:
                # Add empty columns to maintain alignment if needed, or just skip?
                # Using empty DF with correct columns to maintain side-by-side structure visually
                empty_df = pd.DataFrame(columns=[f"{col} Country", col])
                side_by_side_extra_dfs.append(empty_df)

    # Concatenate Main Sheet
    if side_by_side_dfs:
        df_country_final = pd.concat(side_by_side_dfs, axis=1)
    else:
        df_country_final = pd.DataFrame()

    # Concatenate Extra Sheet
    if side_by_side_extra_dfs:
        df_country_extra = pd.concat(side_by_side_extra_dfs, axis=1)
    else:
        df_country_extra = pd.DataFrame()

    all_portfolio_holdings = {}
    top_10_union_names = set()  # set datastructure only holds unique values

    for col in config.portfolio_names:
        port_weights = tgt_alloc[col]
        active_tickers = port_weights[port_weights > 0]

        if active_tickers.empty:
            all_portfolio_holdings[col] = {}
            continue

        aggregated_holdings = {}
        for ticker, weight_in_port in active_tickers.items():
            # INV-218: Changed the schema of our sheet it will be {ticker holdings} {ticker holding weights} * n tickers as columns
            # this makes it extremely generalisable to any number of tickers
            col_name = f"{ticker} Holdings"
            col_weight = f"{ticker} Holding Weights"

            if col_name not in df_underlying.columns:
                print(f"WARNING: {col_name} not found")
                exit()

            if col_name in df_underlying.columns and col_weight in df_underlying.columns:
                w_series = df_underlying[col_weight]
                n_series = df_underlying[col_name]

                w_series_clean = pd.to_numeric(w_series, errors='coerce')
                mask = w_series_clean.notna() & (w_series_clean > 0) & n_series.notna()

                valid_w = w_series_clean[mask]
                valid_n = n_series[mask]

                for w, name in zip(valid_w, valid_n):
                    contrib = w * weight_in_port
                    aggregated_holdings[name] = aggregated_holdings.get(name, 0.0) + contrib

        all_portfolio_holdings[col] = aggregated_holdings
        sorted_port_holdings = sorted(aggregated_holdings.items(), key=lambda x: x[1], reverse=True)
        top_10 = [x[0] for x in sorted_port_holdings[:10]]
        top_10_union_names.update(top_10)

    df_holdings_res = pd.DataFrame(all_portfolio_holdings).fillna(0)

    # df_top_holdings = df_holdings_res.loc[list(top_10_union_names)]
    # df_top_holdings = df_top_holdings.sort_values(by=sort_by_col, ascending=False)
    df_top_holdings = df_holdings_res.sort_values(by=sort_by_col, ascending=False)

    if config.mode == 'CORE':
        out_file = os.path.join(OUTPUT_PATH_CORE, f'monthly_update_core_{TARGET_DATE}.xlsx')
    elif config.mode == 'PORTFOLIO_TEMPLATES':
        out_file = os.path.join(OUTPUT_PATH_PORTFOLIO_TEMPLATES, f'monthly_update_portfolio_templates_{TARGET_DATE}.xlsx')
    else:
        out_file = os.path.join(OUTPUT_PATH_THEMATIC, f'monthly_update_thematic_{TARGET_DATE}.xlsx')

    print(f"Saving output to {out_file}")
    # same saving methodology as our old code
    with pd.ExcelWriter(out_file) as writer:
        tgt_alloc.to_excel(writer, sheet_name='Composition', index=True)
        df_div_yld_res.T.to_excel(writer, sheet_name='Dividend Yield', index=True)
        if not df_sim_res.empty:
            df_sim_res.to_excel(writer, sheet_name='forecastMonthly', index=True)
        df_top_holdings.to_excel(writer, sheet_name='top10holdings', index=True)
        df_sector.to_excel(writer, sheet_name='sector', index=True)
        df_country_final.to_excel(writer, sheet_name='country', index=True)
        df_country_extra.to_excel(writer, sheet_name='country_extra', index=True)


### For doc string explanation of code
# print(PortfolioConfig.__doc__)
# print(load_target_portfolio_series.__doc__)
# print(calculate_weighted_exposure.__doc__)
# print(preprocess_country_data.__doc__)


# NOTE: Core and Thematic updates are independent. You can comment out one or the other if you only want to run a specific update.

# Run Core
run_monthly_update(CORE_CONFIG)

# Run Thematic
run_monthly_update(THEMATIC_CONFIG)

# Run Portfolio Templates
run_monthly_update(PORTFOLIO_TEMPLATES_CONFIG)
