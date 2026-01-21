#!/usr/bin/env python3
"""
Sample Portfolio Report Generator

This script demonstrates how to generate a Portfolio Summary PDF
similar to the Syfe "Client 585455760 Portfolio Summary" format.

Usage:
    python generate_sample_report.py

Output:
    Generates a PDF file in the current directory.
"""

import pandas as pd
import numpy as np
from datetime import date
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.portfolio_report_generator import (
    PortfolioReportGenerator,
    prepare_portfolio_summary_df,
    prepare_income_portfolio_df
)


def create_sample_portfolio_data():
    """
    Creates sample portfolio data matching the Client 585455760 PDF structure.
    
    Returns:
        dict: Contains 'summary', 'growth', 'income' DataFrames
    """
    
    # ==========================================
    # PORTFOLIO SUMMARY DATA
    # ==========================================
    
    df_summary = pd.DataFrame([
        # Growth Portfolios
        {
            'Portfolio ID': '644643869',
            'Portfolio Type': 'Core Equity100',
            'Classification': 'Core',
            'Risk rating (0-5)': 4.0,
            'Activated date': 'April 9, 2025',
            'Value (SGD)': 103466.88,
            'Allocation': 0.0584,
            'Portfolio Returns': 0.0004,
            'Returns (Q1)': 0.0125
        },
        # Growth Portfolios Subtotal
        {
            'Portfolio ID': 'Growth Portfolios',
            'Portfolio Type': '',
            'Classification': '',
            'Risk rating (0-5)': 4.0,
            'Activated date': '',
            'Value (SGD)': 103466.88,
            'Allocation': 0.0584,
            'Portfolio Returns': 0.0004,
            'Returns (Q1)': 0.0125
        },
        # Income & Preservation Portfolios
        {
            'Portfolio ID': '1821329309',
            'Portfolio Type': 'Income Enhance',
            'Classification': 'Income+',
            'Risk rating (0-5)': 2.0,
            'Activated date': 'November 1, 2024',
            'Value (SGD)': 247778.16,
            'Allocation': 0.1400,
            'Portfolio Returns': -0.0001,
            'Returns (Q1)': 0.0125
        },
        {
            'Portfolio ID': '1817852875',
            'Portfolio Type': 'Income Enhance',
            'Classification': 'Income+',
            'Risk rating (0-5)': 2.0,
            'Activated date': 'August 31, 2023',
            'Value (SGD)': 10999.32,
            'Allocation': 0.0062,
            'Portfolio Returns': 0.0010,
            'Returns (Q1)': 0.0125
        },
        {
            'Portfolio ID': '1854996497',
            'Portfolio Type': 'Income Enhance',
            'Classification': 'Income+',
            'Risk rating (0-5)': 2.0,
            'Activated date': 'January 29, 2024',
            'Value (SGD)': 51945.87,
            'Allocation': 0.0293,
            'Portfolio Returns': 0.0005,
            'Returns (Q1)': 0.0125
        },
        {
            'Portfolio ID': '374443838',
            'Portfolio Type': 'REIT 100',
            'Classification': 'REITs',
            'Risk rating (0-5)': 4.0,
            'Activated date': 'October 29, 2024',
            'Value (SGD)': 280900.48,
            'Allocation': 0.1587,
            'Portfolio Returns': -0.0011,
            'Returns (Q1)': 0.0125
        },
        {
            'Portfolio ID': '478419426',
            'Portfolio Type': 'Cash SGD flexi',
            'Classification': 'Cash / MMF',
            'Risk rating (0-5)': 0.5,
            'Activated date': 'September 25, 2024',
            'Value (SGD)': 1075349.14,
            'Allocation': 0.6074,
            'Portfolio Returns': 0.0002,
            'Returns (Q1)': 0.0125
        },
        # Income & Preservation Subtotal
        {
            'Portfolio ID': 'Income & Preservation Portfolios',
            'Portfolio Type': '',
            'Classification': '',
            'Risk rating (0-5)': 1.4,
            'Activated date': '',
            'Value (SGD)': 1666972.97,
            'Allocation': 0.9416,
            'Portfolio Returns': 0.0005,
            'Returns (Q1)': 0.0125
        },
        # Total
        {
            'Portfolio ID': 'Total',
            'Portfolio Type': '',
            'Classification': '',
            'Risk rating (0-5)': 1.5,
            'Activated date': '',
            'Value (SGD)': 1770439.85,
            'Allocation': 1.0000,
            'Portfolio Returns': 0.0009,
            'Returns (Q1)': 0.0125
        },
    ])
    
    # ==========================================
    # GROWTH PORTFOLIOS DETAIL DATA
    # ==========================================
    
    df_growth = pd.DataFrame([
        {
            'Portfolio ID': '644643869',
            'Portfolio Type': 'Core Equity100',
            'Risk rating (0-5)': 4.0,
            'Value (SGD)': 103466.88,
            'Allocation': 1.0000,
            'Portfolio Returns': 0.0004,
            'Returns (Q1)': 0.0125
        },
        # Core Subtotal
        {
            'Portfolio ID': 'Core',
            'Portfolio Type': '',
            'Risk rating (0-5)': 4.0,
            'Value (SGD)': 103466.88,
            'Allocation': 1.0000,
            'Portfolio Returns': 0.0004,
            'Returns (Q1)': '-'
        },
        # Total
        {
            'Portfolio ID': 'Total',
            'Portfolio Type': '',
            'Risk rating (0-5)': 4.0,
            'Value (SGD)': 103466.88,
            'Allocation': 1.0000,
            'Portfolio Returns': 0.0004,
            'Returns (Q1)': '-'
        },
    ])
    
    # ==========================================
    # INCOME & PRESERVATION PORTFOLIOS DETAIL DATA
    # ==========================================
    
    df_income = pd.DataFrame([
        # Income+ Portfolios
        {
            'Portfolio ID': '1821329309',
            'Portfolio Type': 'Income Enhance',
            'Risk rating (0-5)': 2.0,
            'Value (SGD)': 247778.16,
            'Est. Yield (%)*': 5.50,
            'Ann. Proj. Dividend (SGD)**': 13627.80,
            'Dividend Option': 'Reinvest',
            'Maturity period': '-',
            'Allocation': 0.1486
        },
        {
            'Portfolio ID': '1817852875',
            'Portfolio Type': 'Income Enhance',
            'Risk rating (0-5)': 2.0,
            'Value (SGD)': 10999.32,
            'Est. Yield (%)*': 5.50,
            'Ann. Proj. Dividend (SGD)**': 604.96,
            'Dividend Option': 'Reinvest',
            'Maturity period': '-',
            'Allocation': 0.0066
        },
        {
            'Portfolio ID': '1854996497',
            'Portfolio Type': 'Income Enhance',
            'Risk rating (0-5)': 2.0,
            'Value (SGD)': 51945.87,
            'Est. Yield (%)*': 5.50,
            'Ann. Proj. Dividend (SGD)**': 2857.02,
            'Dividend Option': 'Reinvest',
            'Maturity period': '-',
            'Allocation': 0.0312
        },
        # Income+ Subtotal
        {
            'Portfolio ID': 'Income+',
            'Portfolio Type': '',
            'Risk rating (0-5)': 2.0,
            'Value (SGD)': 310723.35,
            'Est. Yield (%)*': 5.50,
            'Ann. Proj. Dividend (SGD)**': 17089.78,
            'Dividend Option': '-',
            'Maturity period': '-',
            'Allocation': 0.1864
        },
        # REITs Portfolio
        {
            'Portfolio ID': '374443838',
            'Portfolio Type': 'REIT 100',
            'Risk rating (0-5)': 4.0,
            'Value (SGD)': 280900.48,
            'Est. Yield (%)*': 6.12,
            'Ann. Proj. Dividend (SGD)**': 17191.11,
            'Dividend Option': 'Reinvest',
            'Maturity period': '-',
            'Allocation': 0.1685
        },
        # REITs Subtotal
        {
            'Portfolio ID': 'REITs',
            'Portfolio Type': '',
            'Risk rating (0-5)': 4.0,
            'Value (SGD)': 280900.48,
            'Est. Yield (%)*': 6.12,
            'Ann. Proj. Dividend (SGD)**': 17191.11,
            'Dividend Option': '-',
            'Maturity period': '-',
            'Allocation': 0.1685
        },
        # Cash Portfolio
        {
            'Portfolio ID': '478419426',
            'Portfolio Type': 'Cash SGD flexi',
            'Risk rating (0-5)': 0.5,
            'Value (SGD)': 1075349.14,
            'Est. Yield (%)*': 3.30,
            'Ann. Proj. Dividend (SGD)**': 35486.52,
            'Dividend Option': 'Accumulating',
            'Maturity period': '-',
            'Allocation': 0.6451
        },
        # Cash Subtotal
        {
            'Portfolio ID': 'Cash / MMF',
            'Portfolio Type': '',
            'Risk rating (0-5)': 0.5,
            'Value (SGD)': 1075349.14,
            'Est. Yield (%)*': 3.30,
            'Ann. Proj. Dividend (SGD)**': 35486.52,
            'Dividend Option': '-',
            'Maturity period': '-',
            'Allocation': 0.6451
        },
        # Total
        {
            'Portfolio ID': 'Total',
            'Portfolio Type': '',
            'Risk rating (0-5)': 1.4,
            'Value (SGD)': 1666972.97,
            'Est. Yield (%)*': 4.19,
            'Ann. Proj. Dividend (SGD)**': 69767.42,
            'Dividend Option': '-',
            'Maturity period': '-',
            'Allocation': 1.0000
        },
    ])
    
    return {
        'summary': df_summary,
        'growth': df_growth,
        'income': df_income
    }


def create_sample_exposure_data():
    """
    Creates sample exposure data for the charts.
    
    Returns:
        dict: Contains exposure data for 'summary', 'growth', 'income' pages
    """
    
    # Overall Portfolio Exposure
    summary_exposure = {
        'asset_class': pd.Series({
            'Cash & MMF': 0.6074,
            'Fixed Income': 0.1755,
            'REITs': 0.1587,
            'Equity': 0.0584
        }),
        'country': pd.Series({
            'Singapore': 0.6774,
            'United States': 0.2050,
            'Europe': 0.0520,
            'Asia Pacific': 0.0400,
            'Others': 0.0256
        }),
        'sector': pd.Series({
            'Cash & Equivalents': 0.6074,
            'Real Estate': 0.1587,
            'Government Bonds': 0.0900,
            'Corporate Bonds': 0.0855,
            'Technology': 0.0200,
            'Financial Services': 0.0180,
            'Healthcare': 0.0120,
            'Others': 0.0084
        })
    }
    
    # Growth Portfolios Exposure
    growth_exposure = {
        'asset_class': pd.Series({
            'Global Equity': 0.65,
            'US Equity': 0.20,
            'Emerging Markets': 0.10,
            'Cash': 0.05
        }),
        'country': pd.Series({
            'United States': 0.55,
            'Europe': 0.15,
            'Japan': 0.10,
            'China': 0.08,
            'Others': 0.12
        }),
        'sector': pd.Series({
            'Technology': 0.28,
            'Financial Services': 0.15,
            'Healthcare': 0.12,
            'Consumer Discretionary': 0.10,
            'Industrials': 0.08,
            'Communication Services': 0.08,
            'Others': 0.19
        })
    }
    
    # Income & Preservation Portfolios Exposure
    income_exposure = {
        'asset_class': pd.Series({
            'Cash & MMF': 0.6451,
            'Fixed Income': 0.1864,
            'REITs': 0.1685
        }),
        'country': pd.Series({
            'Singapore': 0.72,
            'United States': 0.15,
            'Asia Pacific': 0.08,
            'Others': 0.05
        }),
        'sector': pd.Series({
            'Cash & Equivalents': 0.6451,
            'Real Estate': 0.1685,
            'Government Bonds': 0.10,
            'Corporate Bonds': 0.0864
        })
    }
    
    return {
        'summary': summary_exposure,
        'growth': growth_exposure,
        'income': income_exposure
    }


def main():
    """Main function to generate the sample report."""
    
    print("=" * 60)
    print("PORTFOLIO REPORT GENERATOR - SAMPLE EXECUTION")
    print("=" * 60)
    
    # Configuration
    CLIENT_ID = '585455760'
    REPORT_DATE = date(2025, 6, 27)  # Match the sample PDF date
    DATA_PATH = 'Data/'
    
    print(f"\nClient ID: {CLIENT_ID}")
    print(f"Report Date: {REPORT_DATE.strftime('%B %d, %Y')}")
    print(f"Data Path: {DATA_PATH}")
    
    # Create sample data
    print("\n[1/4] Creating sample portfolio data...")
    portfolio_data = create_sample_portfolio_data()
    
    print("[2/4] Creating sample exposure data...")
    exposure_data = create_sample_exposure_data()
    
    # Prepare DataFrames
    print("[3/4] Preparing DataFrames for PDF generation...")
    df_summary = prepare_portfolio_summary_df(portfolio_data['summary'])
    df_growth = portfolio_data['growth']  # Already in correct format
    df_income = portfolio_data['income']  # Already has income-specific columns
    
    # Generate the report
    print("[4/4] Generating PDF report...")
    
    generator = PortfolioReportGenerator(
        client_id=CLIENT_ID,
        report_date=REPORT_DATE,
        data_path=DATA_PATH
    )
    
    output_filename = f"Client_{CLIENT_ID}_{CLIENT_ID}_Portfolio_Summary_V1_[Date].pdf"
    
    try:
        generated_file = generator.generate_report(
            df_portfolio_summary=df_summary,
            df_growth_portfolios=df_growth,
            df_income_portfolios=df_income,
            exposure_data=exposure_data['summary'],
            growth_exposure_data=exposure_data['growth'],
            income_exposure_data=exposure_data['income'],
            output_filename=output_filename,
            include_cover=True,
            include_disclaimers=True
        )
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"\n📄 Report generated: {generated_file}")
        print(f"📁 Location: {os.path.abspath(generated_file)}")
        
        # Print file size
        file_size = os.path.getsize(generated_file)
        print(f"📦 File size: {file_size / 1024:.1f} KB")
        
        print("\nThe PDF contains the following sections:")
        print("  ✓ Cover page with Syfe branding")
        print("  ✓ Portfolio Summary with table and exposure charts")
        print("  ✓ Growth Portfolios detail page")
        print("  ✓ Income & Preservation Portfolios detail page")
        print("  ✓ Disclaimers page with company information")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR GENERATING REPORT")
        print("=" * 60)
        print(f"\n❌ Error: {str(e)}")
        
        import traceback
        traceback.print_exc()
        
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

