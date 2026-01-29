"""pdf parsing and enrichment for client portfolio upload"""

import google.generativeai as genai
import json
import re
import pandas as pd


def parse_pdf_with_gemini(api_key, pdf_bytes, filename="portfolio.pdf"):
    """parse pdf with gemini to extract holdings"""
    try:
        genai.configure(api_key=api_key)
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name
        
        try:
            uploaded_file = genai.upload_file(tmp_path, mime_type="application/pdf")
            
            prompt = """
Analyze this PDF document which contains a client's investment portfolio or brokerage statement.

**Your Task:**
Extract ALL stock/equity holdings from the document. For each holding, identify:
1. The yfinance ticker symbol (use proper exchange suffixes like .SS for Shanghai, .SZ for Shenzhen, .HK for Hong Kong)
2. The company name
3. The market value (as a number)
4. The currency of the value as shown in the PDF (e.g., USD, CNH, CNY, SGD, HKD, EUR, GBP)

**Output Format:**
Return ONLY a valid JSON object in this exact format:
```json
{
    "holdings": [
        {"ticker": "AAPL", "name": "Apple Inc", "value": 15000.50, "currency": "USD"},
        {"ticker": "600887.SS", "name": "Inner Mongolia Yili", "value": 50000.00, "currency": "CNH"},
        {"ticker": "9988.HK", "name": "Alibaba Group", "value": 25000.00, "currency": "HKD"}
    ],
    "notes": "Any relevant notes about the parsing (optional)"
}
```

**Important Rules:**
- For US stocks: use standard ticker (AAPL, GOOGL, MSFT)
- For Shanghai stocks: add .SS suffix (600887.SS, 601318.SS)
- For Shenzhen stocks: add .SZ suffix (000001.SZ, 002594.SZ)
- For Hong Kong stocks: add .HK suffix (9988.HK, 0700.HK)
- For Singapore stocks: add .SI suffix
- Currency must match what is shown in the PDF for each holding's value
- Values should be numbers (no currency symbols or commas)
- If no holdings are found, return {"holdings": [], "notes": "No stock holdings found"}

Parse the document now and return ONLY the JSON object.
"""
            
            # Try models in order
            models_to_try = [
                'gemini-2.5-pro',
                'gemini-2.5-flash',
                'gemini-2.0-flash',
                'gemini-2.0-flash-exp'
            ]
            
            last_error = None
            response_text = None
            
            for model_name in models_to_try:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content([uploaded_file, prompt])
                    response_text = response.text
                    break
                except Exception as e:
                    last_error = e
                    continue
            
            if response_text is None:
                return {}, f"Error: All models failed. Last error: {str(last_error)}"
            
            holdings_dict = {}
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*"holdings".*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    return {}, response_text
            
            try:
                parsed = json.loads(json_str)
                for h in parsed.get('holdings', []):
                    ticker = h.get('ticker', '').upper().strip()
                    if ticker:
                        holdings_dict[ticker] = {
                            'name': h.get('name', ticker),
                            'value': float(h.get('value', 0)),
                            'currency': h.get('currency', 'USD')  # Get currency from AI
                        }
                return holdings_dict, response_text
            except json.JSONDecodeError as e:
                return {}, f"JSON Parse Error: {e}\n\nRaw Response:\n{response_text}"
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        return {}, f"Error: {str(e)}"


def parse_pdf_parallel(api_key, pdf_bytes, filename, num_passes=2):
    """run pdf parsing multiple times in parallel, merge results"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import numpy as np
    
    raw_responses = []
    all_results = []
    
    with ThreadPoolExecutor(max_workers=num_passes) as executor:
        futures = [
            executor.submit(parse_pdf_with_gemini, api_key, pdf_bytes, filename)
            for _ in range(num_passes)
        ]
        
        for future in as_completed(futures):
            holdings, raw_response = future.result()
            raw_responses.append(raw_response)
            if holdings:
                all_results.append(holdings)
    
    if not all_results:
        return {}, {'accuracy_score': 0, 'ticker_agreement': 0}, raw_responses
    
    if len(all_results) == 1:
        return all_results[0], {'accuracy_score': 100, 'ticker_agreement': 100}, raw_responses
    
    all_tickers = set()
    for result in all_results:
        all_tickers.update(result.keys())
    
    ticker_counts = {}
    for ticker in all_tickers:
        ticker_counts[ticker] = sum(1 for r in all_results if ticker in r)
    
    tickers_in_all = sum(1 for t, c in ticker_counts.items() if c == len(all_results))
    ticker_agreement = (tickers_in_all / len(all_tickers)) * 100 if all_tickers else 100
    merged = {}
    value_deviations = []
    
    for ticker in all_tickers:
        values = []
        currencies = []
        names = []
        
        for result in all_results:
            if ticker in result:
                values.append(result[ticker]['value'])
                currencies.append(result[ticker].get('currency', 'USD'))
                names.append(result[ticker].get('name', ticker))
        
        if values:
            avg_value = sum(values) / len(values)
            
            if len(values) > 1:
                max_dev = max(abs(v - avg_value) / avg_value * 100 for v in values) if avg_value > 0 else 0
                value_deviations.append(max_dev)
            
            from collections import Counter
            most_common_currency = Counter(currencies).most_common(1)[0][0]
            
            merged[ticker] = {
                'name': names[0],
                'value': avg_value,
                'currency': most_common_currency
            }
    
    avg_deviation = sum(value_deviations) / len(value_deviations) if value_deviations else 0
    value_consistency = max(0, 100 - avg_deviation * 2)
    accuracy_score = (value_consistency * 0.5 + ticker_agreement * 0.5)
    accuracy_score = min(100, max(0, accuracy_score))
    
    accuracy_metrics = {
        'accuracy_score': round(accuracy_score, 1),
        'ticker_agreement': round(ticker_agreement, 1),
        'avg_deviation': round(avg_deviation, 2),
        'num_passes': len(all_results)
    }
    
    return merged, accuracy_metrics, raw_responses


def enrich_with_yfinance(holdings_dict):
    """enrich holdings with country/sector from yfinance"""
    import yfinance as yf
    import time
    
    exchange_suffixes = ['', '.SS', '.SZ', '.HK', '.L', '.T', '.SI']
    max_retries = 3
    
    enriched = {}
    failed = []
    
    for ticker, data in holdings_dict.items():
        ai_currency = data.get('currency', None)
        
        found = False
        best_info = None
        best_ticker = ticker
        
        for suffix in exchange_suffixes:
            try_ticker = ticker + suffix if not ticker.endswith(suffix) else ticker
            
            for retry in range(max_retries):
                print(f"  Trying yfinance: {try_ticker} (attempt {retry + 1}/{max_retries})")
                
                try:
                    stock = yf.Ticker(try_ticker)
                    info = stock.info
                    if info.get('country') or info.get('sector') or info.get('currency'):
                        found = True
                        best_info = info
                        best_ticker = try_ticker
                        print(f"    SUCCESS: {try_ticker}")
                        break
                        
                except Exception as e:
                    print(f"    FAILED: {try_ticker} -> {e}")
                    if retry < max_retries - 1:
                        time.sleep(1)
                    continue
            
            if found:
                break
        
        if found and best_info:
            country = best_info.get('country', 'Unknown')
            sector = best_info.get('sector', 'Unknown')
            industry = best_info.get('industry', 'Unknown')
            
            currency = ai_currency if ai_currency else best_info.get('currency', 'Unknown')
            
            # infer country from exchange if needed
            if country == 'Unknown' or not country:
                exchange = best_info.get('exchange', '')
                if 'NYSE' in exchange or 'NASDAQ' in exchange:
                    country = 'United States'
                elif 'HKSE' in exchange or 'HKG' in exchange:
                    country = 'Hong Kong'
                elif 'Shanghai' in exchange or 'SSE' in exchange:
                    country = 'China'
                elif 'Shenzhen' in exchange or 'SZSE' in exchange:
                    country = 'China'
            
            enriched[ticker] = {
                'name': data.get('name', best_info.get('shortName', ticker)),
                'value': data['value'],
                'country': country if country else 'Unknown',
                'sector': sector if sector else 'Unknown',
                'industry': industry if industry else 'Unknown',
                'currency': currency if currency else 'Unknown'
            }
            print(f"    Currency: {currency} (AI: {ai_currency}, yfinance: {best_info.get('currency')})")
        else:
            failed.append(ticker)
            enriched[ticker] = {
                'name': data.get('name', ticker),
                'value': data['value'],
                'country': 'Unknown',
                'sector': 'Unknown',
                'industry': 'Unknown',
                'currency': ai_currency if ai_currency else 'Unknown'
            }
            print(f"  FAILED: Could not find {ticker} in yfinance, using AI currency: {ai_currency}")
    
    return enriched, failed


def calculate_aggregate_exposure(enriched_holdings):
    """calc country/sector exposure weighted by value"""
    total_value = sum(h['value'] for h in enriched_holdings.values())
    
    if total_value == 0:
        return {}, {}
    
    country_exposure = {}
    sector_exposure = {}
    
    for ticker, data in enriched_holdings.items():
        weight = data['value'] / total_value
        
        country = data.get('country', 'Unknown')
        sector = data.get('sector', 'Unknown')
        
        country_exposure[country] = country_exposure.get(country, 0) + weight
        sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
    
    country_exposure = dict(sorted(country_exposure.items(), key=lambda x: x[1], reverse=True))
    sector_exposure = dict(sorted(sector_exposure.items(), key=lambda x: x[1], reverse=True))
    
    return country_exposure, sector_exposure


def format_holdings_dataframe(enriched_holdings):
    """format holdings as df for display"""
    total_value = sum(h['value'] for h in enriched_holdings.values())
    
    rows = []
    for ticker, data in enriched_holdings.items():
        weight = data['value'] / total_value if total_value > 0 else 0
        rows.append({
            'Ticker': ticker,
            'Name': data.get('name', ticker),
            'Value ($)': data['value'],
            'Weight (%)': weight,
            'Country': data.get('country', 'Unknown'),
            'Sector': data.get('sector', 'Unknown'),
            'Currency': data.get('currency', 'USD')
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Value ($)', ascending=False)
    
    return df


def parse_pdf_multipass(api_key, pdf_bytes, filename="portfolio.pdf", num_passes=2):
    """parse pdf multiple times, compare for accuracy scoring"""
    import numpy as np
    
    all_results = []
    raw_responses = []
    
    for i in range(num_passes):
        holdings, raw = parse_pdf_with_gemini(api_key, pdf_bytes, filename)
        all_results.append(holdings)
        raw_responses.append(f"--- Pass {i+1} ---\n{raw}")
    
    if not any(all_results):
        return {}, {
            'min_deviation': 0,
            'max_deviation': 0,
            'avg_discrepancy': 0,
            'std_deviation': 0,
            'accuracy_score': 0,
            'ticker_agreement': 0,
            'num_passes': num_passes
        }, raw_responses
    
    all_tickers = set()
    for result in all_results:
        all_tickers.update(result.keys())
    
    ticker_metrics = {}
    value_deviations = []
    
    for ticker in all_tickers:
        values = []
        names = []
        presence_count = 0
        
        for result in all_results:
            if ticker in result:
                values.append(result[ticker]['value'])
                names.append(result[ticker]['name'])
                presence_count += 1
            else:
                values.append(0)
        values_arr = np.array(values)
        non_zero_values = values_arr[values_arr > 0]
        
        if len(non_zero_values) > 0:
            avg_value = np.mean(non_zero_values)
            std_value = np.std(non_zero_values) if len(non_zero_values) > 1 else 0
            min_val = np.min(non_zero_values)
            max_val = np.max(non_zero_values)
            
            if avg_value > 0:
                pct_deviation = (std_value / avg_value) * 100
            else:
                pct_deviation = 0
                
            value_deviations.append(pct_deviation)
            
            ticker_metrics[ticker] = {
                'avg_value': avg_value,
                'std_value': std_value,
                'min_value': min_val,
                'max_value': max_val,
                'pct_deviation': pct_deviation,
                'presence_count': presence_count,
                'most_common_name': max(set(names), key=names.count) if names else ticker
            }
    if value_deviations:
        min_deviation = min(value_deviations)
        max_deviation = max(value_deviations)
        avg_discrepancy = np.mean(value_deviations)
        std_deviation = np.std(value_deviations) if len(value_deviations) > 1 else 0
    else:
        min_deviation = max_deviation = avg_discrepancy = std_deviation = 0
    
    tickers_in_all_passes = sum(1 for t, m in ticker_metrics.items() if m['presence_count'] == num_passes)
    ticker_agreement = (tickers_in_all_passes / len(all_tickers) * 100) if all_tickers else 0
    
    value_consistency_score = max(0, 100 - avg_discrepancy * 10)
    accuracy_score = (value_consistency_score * 0.5) + (ticker_agreement * 0.5)
    accuracy_score = min(100, max(0, accuracy_score))
    merged_holdings = {}
    for ticker, metrics in ticker_metrics.items():
        merged_holdings[ticker] = {
            'name': metrics['most_common_name'],
            'value': metrics['avg_value']
        }
    
    accuracy_metrics = {
        'min_deviation': min_deviation,
        'max_deviation': max_deviation,
        'avg_discrepancy': avg_discrepancy,
        'std_deviation': std_deviation,
        'accuracy_score': accuracy_score,
        'ticker_agreement': ticker_agreement,
        'num_passes': num_passes,
        'ticker_details': ticker_metrics
    }
    
    return merged_holdings, accuracy_metrics, raw_responses


def generate_portfolio_insights(api_key, country_exposure, sector_exposure, top_holdings, total_value):
    """generate AI portfolio insights via gemini"""
    import google.generativeai as genai
    from utils.llm_helper import SYFE_CONTEXT
    
    genai.configure(api_key=api_key)
    
    country_str = "\n".join([f"- {k}: {v*100:.1f}%" for k, v in sorted(country_exposure.items(), key=lambda x: -x[1])[:10]])
    sector_str = "\n".join([f"- {k}: {v*100:.1f}%" for k, v in sorted(sector_exposure.items(), key=lambda x: -x[1])[:10]])
    
    holdings_str = ""
    for h in top_holdings[:5]:
        value_key = [k for k in h.keys() if 'Value' in k][0] if any('Value' in k for k in h.keys()) else 'value'
        val = h.get(value_key, 0)
        pct = (val / total_value * 100) if total_value > 0 else 0
        holdings_str += f"- {h.get('Ticker', 'N/A')}: {h.get('Name', 'Unknown')} ({pct:.1f}%)\n"
    
    prompt = f"""
{SYFE_CONTEXT}

**Client Portfolio Analysis:**

**Country Exposure:**
{country_str}

**Sector Exposure:**
{sector_str}

**Top 5 Holdings by Value:**
{holdings_str}

**Total Portfolio Value:** ${total_value:,.0f}

---

**Your Task:**
Analyze this portfolio and generate 2-3 actionable insights to help the client improve diversification.

For each insight, consider:
1. **Concentration Risk**: Single country >50%, single sector >40%, or single stock >15%
2. **Missing Asset Classes**: No bonds, no REITs, no international exposure
3. **Geographic Bias**: Over-reliance on one region

**Output Format:**
Return ONLY a valid JSON array with 2-3 insights:
```json
[
    {{
        "title": "High US Concentration",
        "description": "Your portfolio has 65% exposure to US markets, which creates geographic concentration risk.",
        "recommendation": "Consider adding Syfe Core Global or Core Balanced for broader international diversification."
    }},
    {{
        "title": "Tech-Heavy Portfolio",
        "description": "Technology represents 45% of your holdings, well above typical diversified allocations of 20-25%.",
        "recommendation": "REIT+ or Income+ can add real estate and fixed income exposure to balance sector allocation."
    }}
]
```

Keep insights concise, actionable, and recommend specific Syfe products where relevant.
Generate the JSON now.
"""
    
    try:
        models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']
        response_text = None
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                response_text = response.text
                break
            except Exception:
                continue
        
        if not response_text:
            return [{"title": "Analysis Unavailable", "description": "Could not generate insights at this time.", "recommendation": "Please try again later."}]
        
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return [{"title": "Analysis Complete", "description": response_text[:500], "recommendation": "See above for details."}]
        
        try:
            insights = json.loads(json_str)
            return insights
        except json.JSONDecodeError:
            return [{"title": "Analysis Complete", "description": response_text[:500], "recommendation": "See above for details."}]
            
    except Exception as e:
        return [{"title": "Error", "description": f"Could not generate insights: {str(e)}", "recommendation": "Please try again."}]
