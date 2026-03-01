"""
portfolio report generator - creates PDF reports matching Syfe Portfolio Summary format

generates professional PDF reports with:
- cover page from template with Client ID
- portfolio summary table with grouping (Section 1)
- portfolio returns bar chart
- asset class, country, sector exposure charts (Section 2)
- appendix/disclaimers page from template
- uses Template_Landscape.pdf as background for each content page
"""

import os
import io
import pandas as pd
import numpy as np
from datetime import datetime, date

# ReportLab Imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Spacer, 
    Paragraph, PageBreak, KeepTogether, Image
)
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, String, Rect, Wedge, Line
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend

# PyPDF Imports
from pypdf import PdfReader, PdfWriter

# ==========================================
# COLOR DEFINITIONS (Syfe Navy/Blue Theme)
# ==========================================

SYFE_COLORS = {
    'dark_navy': colors.HexColor("#1a2744"),      # dark navy (header bg)
    'navy': colors.HexColor("#263159"),           # navy 
    'medium_blue': colors.HexColor("#4a5568"),    # medium blue-grey
    'light_blue': colors.HexColor("#c5cfe8"),     # light blue (subtotal rows)
    'lighter_blue': colors.HexColor("#e8ecf5"),   # lighter blue (alt rows)
    'white': colors.white,
    'light_grey': colors.HexColor("#f7fafc"),     # very light grey
    'medium_grey': colors.HexColor("#e2e8f0"),    # border grey
    'text_dark': colors.HexColor("#2d3748"),      # dark text
    'text_grey': colors.HexColor("#718096"),      # grey text
}

# donut chart colors (navy theme - matching the screenshot)
DONUT_COLORS = [
    colors.HexColor("#1a2744"),  # dark navy
    colors.HexColor("#2d4a7c"),  # navy blue
    colors.HexColor("#4a6fa5"),  # medium blue
    colors.HexColor("#7b9cc7"),  # light blue
    colors.HexColor("#a8c5e2"),  # lighter blue
    colors.HexColor("#d1e3f5"),  # very light blue
    colors.HexColor("#4a5568"),  # grey-blue
    colors.HexColor("#718096"),  # medium grey
]

# bar chart color
BAR_COLOR = colors.HexColor("#7b9cc7")  # light blue for bars

# ==========================================
# PARAGRAPH STYLES
# ==========================================

def get_syfe_styles():
    """returns dictionary of styled ParagraphStyles for our syfe branding"""
    styles = {}
    
    styles['section_header'] = ParagraphStyle(
        name='SectionHeader',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=14,
        textColor=SYFE_COLORS['text_dark'],
        spaceBefore=8,
        spaceAfter=8
    )
    
    styles['chart_title'] = ParagraphStyle(
        name='ChartTitle',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=SYFE_COLORS['text_dark'],
        spaceBefore=4,
        spaceAfter=4
    )
    
    styles['footnote'] = ParagraphStyle(
        name='Footnote',
        fontName='Helvetica',
        fontSize=7,
        textColor=SYFE_COLORS['text_grey'],
        spaceBefore=2,
        spaceAfter=2
    )
    
    return styles

# ==========================================
# DONUT CHART (matches screenshot style)
# ==========================================

def create_donut_chart_drawing(data, title="", width=280, height=220):
    """
    creates a native ReportLab donut chart matching the screenshot style,
    uses Wedge shapes to create the donut effect
    
    Args:
        data: pd.Series with labels as index and values as data
        title: Chart title
        width, height: Drawing dimensions
    
    Returns:
        Drawing object or None
    """
    if data is None:
        return None
    
    if isinstance(data, dict):
        data = pd.Series(data)
    
    if data.empty or data.sum() == 0:
        return None
    
    # filter small values and normalize
    data = data[data > 0.001]
    total = data.sum()
    data = data / total
    data = data.sort_values(ascending=False)
    
    # show ALL items from the data - no truncation
    
    d = Drawing(width, height)
    
    # create donut chart using Wedges
    center_x = 100
    center_y = height // 2 + 10
    outer_radius = 60
    inner_radius = 33  # creates the donut hole
    
    start_angle = 90  # start from top
    
    for i, (label, value) in enumerate(data.items()):
        # calculate sweep angle (in degrees)
        sweep_angle = value * 360
        
        # create outer wedge
        wedge = Wedge(
            center_x, center_y,
            outer_radius,
            start_angle,
            start_angle + sweep_angle,
            radius1=inner_radius,  # inner radius for donut
            fillColor=DONUT_COLORS[i % len(DONUT_COLORS)],
            strokeColor=colors.white,
            strokeWidth=1
        )
        d.add(wedge)
        
        start_angle += sweep_angle
    
    # add title at bottom center
    if title:
        d.add(String(width // 2, 10, title,
                     fontName='Helvetica-Bold', fontSize=13,
                     fillColor=SYFE_COLORS['text_dark'],
                     textAnchor='middle'))
    
    # add legend on the right side - dynamic spacing based on number of items
    legend_x = 180
    legend_y = height - 20
    num_items = len(data)
    # adjust line height based on number of items (min 12, max 18)
    legend_line_height = min(18, max(12, (height - 40) / max(num_items, 1)))
    
    for i, (label, value) in enumerate(data.items()):
        y_pos = legend_y - (i * legend_line_height)
        
        # color box
        d.add(Rect(legend_x, y_pos - 4, 12, 12, 
                   fillColor=DONUT_COLORS[i % len(DONUT_COLORS)],
                   strokeColor=None))
        
        # label - show full name (up to 28 chars)
        pct = value * 100
        label_text = str(label)[:28]
        d.add(String(legend_x + 18, y_pos, label_text,
                     fontName='Helvetica', fontSize=7,
                     fillColor=SYFE_COLORS['text_dark']))
        
        # percentage on the right
        d.add(String(width - 10, y_pos, f"{pct:.1f}%",
                     fontName='Helvetica', fontSize=7,
                     fillColor=SYFE_COLORS['text_grey'],
                     textAnchor='end'))
    
    return d

# ==========================================
# BAR CHART (matches screenshot style)
# ==========================================

def create_horizontal_bar_chart_drawing(data, title="", width=700, height=300, show_legend=True):
    """
    creates a native ReportLab horizontal bar chart with multi-colored bars,
    better for handling many portfolios and long names
    
    Args:
        data: pd.Series with labels as index and values as data
        title: Chart title
        width, height: Drawing dimensions
    
    Returns:
        Drawing object or None
    """
    if data is None:
        return None
    
    if isinstance(data, dict):
        data = pd.Series(data)
    
    if data.empty:
        return None
    
    # sort descending and show all portfolios (up to 25)
    data = data.sort_values(ascending=False).head(25)
    # reverse for horizontal chart (top item at top)
    data = data.iloc[::-1]
    
    num_bars = len(data)
    
    # adjust height based on number of bars (minimum 20px per bar)
    min_height = num_bars * 20 + 60
    height = max(height, min_height)
    
    d = Drawing(width, height)
    
    # chart area dimensions
    label_width = 200  # more space for portfolio names on left
    chart_x = label_width + 10
    chart_y = 35
    chart_width = width - label_width - 50
    chart_height = height - 70
    
    # calculate bar dimensions
    bar_height = min(18, (chart_height - 10) / num_bars - 4)
    bar_spacing = (chart_height - (bar_height * num_bars)) / (num_bars + 1)
    
    # convert to percentages
    max_val = max([v * 100 for v in data.values]) if len(data) > 0 else 10
    max_val = max(max_val, 1)  # prevent division by zero
    
    # draw grid lines
    for i in range(6):
        grid_x = chart_x + (i / 5) * chart_width
        d.add(Line(grid_x, chart_y, grid_x, chart_y + chart_height,
                   strokeColor=SYFE_COLORS['medium_grey'], strokeWidth=0.5))
        # grid labels
        d.add(String(grid_x, chart_y - 12, f"{int(max_val * 1.2 * i / 5)}%",
                     fontName='Helvetica', fontSize=7,
                     fillColor=SYFE_COLORS['text_grey'],
                     textAnchor='middle'))
    
    # draw bars and labels
    for i, (label, value) in enumerate(data.items()):
        bar_y = chart_y + bar_spacing + i * (bar_height + bar_spacing)
        bar_width = (value * 100 / (max_val * 1.2)) * chart_width if max_val > 0 else 0
        
        # draw colored bar
        d.add(Rect(chart_x, bar_y, max(bar_width, 2), bar_height,
                   fillColor=EXTENDED_COLORS[i % len(EXTENDED_COLORS)],
                   strokeColor=None))
        
        # portfolio name on the left (show full name up to 32 chars)
        label_text = str(label)[:32]
        d.add(String(chart_x - 8, bar_y + bar_height / 2 - 2, label_text,
                     fontName='Helvetica', fontSize=6,
                     fillColor=SYFE_COLORS['text_dark'],
                     textAnchor='end'))
        
        # value label at end of bar
        d.add(String(chart_x + bar_width + 5, bar_y + bar_height / 2 - 2, f"{value*100:.2f}%",
                     fontName='Helvetica', fontSize=7,
                     fillColor=SYFE_COLORS['text_dark'],
                     textAnchor='start'))
    
    # add title at bottom center
    if title:
        d.add(String(width // 2, 8, title,
                     fontName='Helvetica-Bold', fontSize=10,
                     fillColor=SYFE_COLORS['text_dark'],
                     textAnchor='middle'))
    
    return d


# extended color palette for more bars (defined before functions that use it)
EXTENDED_COLORS = DONUT_COLORS + [
    colors.HexColor("#3d5a80"),
    colors.HexColor("#5c7a99"),
    colors.HexColor("#8fa3b8"),
    colors.HexColor("#2c3e50"),
    colors.HexColor("#34495e"),
    colors.HexColor("#5d6d7e"),
    colors.HexColor("#85929e"),
]


def create_dividends_bar_chart_drawing(data, title="", width=380, height=220):
    """
    creates a horizontal bar chart for dividends (shows dollar amounts)
    
    Args:
        data: pd.Series with portfolio names as index and dividend amounts as values
        title: Chart title (will be auto-generated if empty)
        width, height: Drawing dimensions
    
    Returns:
        Drawing object or None
    """
    if data is None:
        return None
    
    if isinstance(data, dict):
        data = pd.Series(data)
    
    if data.empty:
        return None
    
    # sort descending and show top 10 max
    data = data.sort_values(ascending=False).head(10)
    num_portfolios = len(data)
    
    # reverse for horizontal chart (top item at top)
    data = data.iloc[::-1]
    
    num_bars = len(data)
    
    # keep fixed height to align with sector exposure chart
    # don't adjust height dynamically - use the same height as donut charts
    d = Drawing(width, height)
    
    # chart area dimensions - aligned with donut chart legend
    # donut chart legend starts at height - 25 = 195, so bars should start there too
    label_width = 160
    chart_x = label_width + 10
    chart_y = 45  # more space for title at bottom
    chart_width = width - label_width - 60
    chart_height = height - 70  # top aligned with donut legend (starts at ~195)
    
    # calculate bar dimensions
    bar_height = min(15, (chart_height - 10) / num_bars - 2)
    bar_spacing = (chart_height - (bar_height * num_bars)) / (num_bars + 1)
    
    # find max value for scaling
    max_val = max(data.values) if len(data) > 0 else 1
    max_val = max(max_val, 1)  # prevent division by zero
    
    # no x-axis grid labels to keep it clean and avoid crowding the title
    # just draw subtle grid lines
    for i in range(5):
        grid_x = chart_x + (i / 4) * chart_width
        d.add(Line(grid_x, chart_y, grid_x, chart_y + chart_height,
                   strokeColor=SYFE_COLORS['medium_grey'], strokeWidth=0.5))
    
    # draw bars and labels
    for i, (label, value) in enumerate(data.items()):
        bar_y = chart_y + bar_spacing + i * (bar_height + bar_spacing)
        bar_width = (value / (max_val * 1.2)) * chart_width if max_val > 0 else 0
        
        # draw colored bar
        d.add(Rect(chart_x, bar_y, max(bar_width, 2), bar_height,
                   fillColor=EXTENDED_COLORS[i % len(EXTENDED_COLORS)],
                   strokeColor=None))
        
        # portfolio name on the left - larger font size
        label_text = str(label)[:26]
        d.add(String(chart_x - 5, bar_y + bar_height / 2 - 2, label_text,
                     fontName='Helvetica', fontSize=7,
                     fillColor=SYFE_COLORS['text_dark'],
                     textAnchor='end'))
        
        # value label at end of bar (dollar amount)
        d.add(String(chart_x + bar_width + 3, bar_y + bar_height / 2 - 2, f"${value:,.0f}",
                     fontName='Helvetica', fontSize=7,
                     fillColor=SYFE_COLORS['text_dark'],
                     textAnchor='start'))
    
    # title: "Portfolio Dividends (Reinvested)"
    if not title:
        title = "Portfolio Dividends (Reinvested)"
    
    # add title at bottom center - same y position and font size as Sector Exposure
    d.add(String(width // 2, 10, title,
                 fontName='Helvetica-Bold', fontSize=13,
                 fillColor=SYFE_COLORS['text_dark'],
                 textAnchor='middle'))
    
    return d


# legacy alias for backwards compatibility
extended_colors = EXTENDED_COLORS + [
    colors.HexColor("#3d5a80"),
    colors.HexColor("#5c7a99"),
    colors.HexColor("#8fa3b8"),
    colors.HexColor("#2c3e50"),
    colors.HexColor("#34495e"),
    colors.HexColor("#5d6d7e"),
    colors.HexColor("#85929e"),
]


def create_cashflow_chart_drawing(cashflow_data, currency="SGD", width=760, height=280):
    """
    creates a stacked bar chart for monthly cash flow projections
    
    Args:
        cashflow_data: dict with keys like "Portfolio - Ticker" and values as arrays of 12 monthly amounts
        currency: Currency symbol to display
        width, height: Drawing dimensions
    
    Returns:
        Drawing object or None
    """
    if cashflow_data is None or len(cashflow_data) == 0:
        return None
    
    d = Drawing(width, height)
    
    items = list(cashflow_data.items())
    num_items = len(items)
    
    # layout dimensions
    y_axis_values_width = 55  # space for "S$17.4k" etc
    chart_bars_width = 420  # width of the actual bar chart
    legend_width = 110
    
    # center the chart bars visually on the page (legend hangs to the right)
    # the visual center is based on y_axis + chart_bars only
    visual_content_width = y_axis_values_width + chart_bars_width
    left_pad = (width - visual_content_width - legend_width) / 2
    
    chart_x = left_pad + y_axis_values_width
    chart_y = 55  # space for title at bottom and month labels
    chart_width = chart_bars_width
    chart_height = height - 100  # leave more room at top and bottom
    
    # get max value for scaling - add 15% headroom so bars don't touch top
    monthly_totals = [0] * 12
    for key, values in cashflow_data.items():
        for i, v in enumerate(values):
            monthly_totals[i] += v
    
    max_val = max(monthly_totals) if monthly_totals else 1
    max_val = max(max_val * 1.15, 1)  # add 15% headroom at top
    
    bar_width = (chart_width - 20) / 12 - 4
    
    # draw y-axis grid lines and labels
    for i in range(6):
        y_pos = chart_y + (i / 5) * chart_height
        d.add(Line(chart_x, y_pos, chart_x + chart_width, y_pos,
                   strokeColor=SYFE_COLORS['medium_grey'], strokeWidth=0.5))
        # y-axis value labels (S$17.4k, etc) - right-aligned before chart
        grid_val = max_val * i / 5
        ccy_symbol = "S$" if currency == "SGD" else "$"
        d.add(String(chart_x - 8, y_pos - 2, f"{ccy_symbol}{grid_val/1000:.1f}k",
                     fontName='Helvetica', fontSize=7,
                     fillColor=SYFE_COLORS['text_grey'],
                     textAnchor='end'))
    
    # draw stacked bars for each month
    for month in range(12):
        bar_x = chart_x + 10 + month * (bar_width + 4)
        cumulative_y = chart_y
        
        for i, (key, values) in enumerate(items):
            value = values[month] if month < len(values) else 0
            if value > 0:
                bar_height = (value / max_val) * chart_height
                
                # draw bar segment
                d.add(Rect(bar_x, cumulative_y, bar_width, bar_height,
                           fillColor=EXTENDED_COLORS[i % len(EXTENDED_COLORS)],
                           strokeColor=colors.white, strokeWidth=0.5))
                
                cumulative_y += bar_height
        
        # month label
        d.add(String(bar_x + bar_width / 2, chart_y - 12, str(month + 1),
                     fontName='Helvetica', fontSize=7,
                     fillColor=SYFE_COLORS['text_dark'],
                     textAnchor='middle'))
    
    # x-axis label "Month" - centered on page
    d.add(String(width / 2, chart_y - 30, "Month",
                 fontName='Helvetica', fontSize=8,
                 fillColor=SYFE_COLORS['text_dark'],
                 textAnchor='middle'))
    
    # title at bottom center - centered on page
    d.add(String(width / 2, 10, f"Monthly Cash Flow Projections ({currency})",
                 fontName='Helvetica-Bold', fontSize=13,
                 fillColor=SYFE_COLORS['text_dark'],
                 textAnchor='middle'))
    
    # legend on the right - show all items
    legend_x = chart_x + chart_width + 25
    legend_start_y = chart_y + chart_height - 5
    line_height = min(14, (chart_height - 10) / max(num_items, 1))  # adjust spacing based on items
    
    for i, (key, values) in enumerate(items):
        y_pos = legend_start_y - (i * line_height)
        # color box
        d.add(Rect(legend_x, y_pos - 4, 10, 10,
                   fillColor=EXTENDED_COLORS[i % len(EXTENDED_COLORS)],
                   strokeColor=None))
        # label - show portfolio type, truncate if needed
        label = key.split(' - ')[-1] if ' - ' in key else key
        label = label[:25] if len(label) > 25 else label
        d.add(String(legend_x + 14, y_pos - 2, label,
                     fontName='Helvetica', fontSize=6,
                     fillColor=SYFE_COLORS['text_dark']))
    
    return d


def create_cashflow_table(cashflow_data, currency="SGD", page_width=750):
    """
    creates a table showing monthly cash flows by portfolio type,
    uses month names (Jan, Feb, etc.) instead of "Month 1", "Month 2"
    
    Args:
        cashflow_data: dict with keys as portfolio types and values as arrays of 12 monthly amounts
        currency: Currency symbol
        page_width: Page width for table sizing
    
    Returns:
        ReportLab Table object or None
    """
    if cashflow_data is None or len(cashflow_data) == 0:
        return None
    
    # month names
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # currency symbol
    ccy_symbol = "S$" if currency == "SGD" else ("HK$" if currency == "HKD" else "$")
    
    # build table data
    header = ['Portfolio Type'] + months + ['Total Annual']
    table_data = [header]
    
    grand_totals = [0.0] * 12
    
    for port_type, monthly_values in cashflow_data.items():
        # clean up portfolio type name
        display_name = port_type.split(' - ')[-1] if ' - ' in port_type else port_type
        display_name = display_name[:30] if len(display_name) > 30 else display_name
        
        row = [display_name]
        annual_total = 0.0
        
        for i, val in enumerate(monthly_values):
            row.append(f"{ccy_symbol}{val:,.0f}" if val >= 1 else "-")
            annual_total += val
            grand_totals[i] += val
        
        row.append(f"{ccy_symbol}{annual_total:,.0f}")
        table_data.append(row)
    
    # add total row
    total_row = ['Total']
    grand_annual = 0.0
    for val in grand_totals:
        total_row.append(f"{ccy_symbol}{val:,.0f}" if val >= 1 else "-")
        grand_annual += val
    total_row.append(f"{ccy_symbol}{grand_annual:,.0f}")
    table_data.append(total_row)
    
    # calculate column widths
    first_col_width = 130  # wider for longer portfolio names
    month_col_width = (page_width - first_col_width - 60) / 13  # 12 months + total
    col_widths = [first_col_width] + [month_col_width] * 13
    
    # create table
    table = Table(table_data, colWidths=col_widths)
    
    # style the table
    style_cmds = [
        # header row
        ('BACKGROUND', (0, 0), (-1, 0), SYFE_COLORS['dark_navy']),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 6),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        
        # data rows
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 6),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),  # center-align all data
        
        # total row
        ('BACKGROUND', (0, -1), (-1, -1), SYFE_COLORS['light_blue']),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        
        # grid
        ('GRID', (0, 0), (-1, -1), 0.5, SYFE_COLORS['medium_grey']),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]
    
    # alternate row colors for data rows
    for i in range(1, len(table_data) - 1):
        if i % 2 == 0:
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), colors.Color(0.97, 0.97, 0.97)))
    
    table.setStyle(TableStyle(style_cmds))
    
    return table


# ==========================================
# PORTFOLIO SUMMARY TABLE (grouped format)
# ==========================================

def create_grouped_portfolio_table(df_raw, page_width, df_asset_class=None, df_risk_rating=None):
    """
    creates a grouped portfolio summary table matching the screenshot format
    
    Format:
    - Portfolio ID, Portfolio Type, Classification, Risk rating, Activated date, Value, Allocation, Returns, Returns Q1
    - Grouped by Growth Portfolios and Income & Preservation Portfolios
    - Subtotals per group
    - Total row at bottom
    """
    if df_raw is None or df_raw.empty:
        return None
    
    df = df_raw.copy()
    
    # normalize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # helper to safely get float
    def safe_float(val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace(',', ''))
        except:
            return 0.0
    
    # convert numeric columns
    if 'nav_in_sgd' in df.columns:
        df['nav_in_sgd'] = df['nav_in_sgd'].apply(safe_float)
    
    if 'pnl_inception' in df.columns:
        df['pnl_inception'] = df['pnl_inception'].apply(safe_float)
    
    # calculate allocation
    total_nav = df['nav_in_sgd'].sum() if 'nav_in_sgd' in df.columns else 0
    
    # portfolio type to ticker mapping for asset class lookup
    PORTFOLIO_TICKER_MAP = {
        'INCOME_ENHANCE': 'Income Enhance',
        'INCOME_ENHANCED': 'Income Enhance',
        'INCOME_PRESERVE': 'Income Preserve',
        'REIT': 'REIT 100',
        'REIT_100': 'REIT 100',
        'REIT_RISK_MANAGED': 'REIT Risk Managed',
        'BLACKROCK': 'Blackrock Private Credit Fund',
        'BLACKROCK_PRIVATE_CREDIT': 'Blackrock Private Credit Fund',
        'MONROE': 'Monroe Income Plus Fund',
        'MONROE_CAPITAL': 'Monroe Income Plus Fund',
        'MONROE_INCOME_PLUS': 'Monroe Income Plus Fund',
        'CORE_EQUITY100': 'Core Equity100',
        'CORE_EQUITY_100': 'Core Equity100',
        'GLOBAL_EQUITY_100': 'Core Equity100',
        'GLOBAL_EQUITY': 'Core Equity100',
        'CORE_GROWTH': 'Core Growth',
        'CORE_BALANCED': 'Core Balanced',
        'CORE_DEFENSIVE': 'Core Defensive',
        'CHINA_GROWTH': 'China Growth',
        'CHINA': 'China Growth',
        'ESG_AND_CLEAN_ENERGY': 'ESG & Clean Energy',
        'ESG_CLEAN_ENERGY': 'ESG & Clean Energy',
        'ESG': 'ESG & Clean Energy',
        'HEALTHCARE_INNOVATION': 'Healthcare Innovation',
        'HEALTHCARE': 'Healthcare Innovation',
        'DISRUPTIVE_TECHNOLOGY': 'Disruptive Technology',
        'DISRUPTIVE_TECH': 'Disruptive Technology',
        'DISRUPTIVE': 'Disruptive Technology',
        'CASH_PLUS_SGD': 'Cash SGD flexi',
        'CASH_SGD': 'Cash SGD flexi',
        'CASH_PLUS': 'Cash SGD flexi',
        'CASH_PLUS_USD': 'Cash USD flexi',
        'CASH_USD': 'Cash USD flexi',
        'CASH_PLUS_GUARANTEED': 'Cash SGD guaranteed',
        'CASH_GUARANTEED': 'Cash SGD guaranteed',
        'SRS_INCOME_ENHANCE': 'SRS Income Enhance',
        'SRS_INCOME_ENHANCED': 'SRS Income Enhance',
        'SRS_INCOME_PRESERVE': 'SRS Income Preserve',
        'SRS_EQUITY100': 'SRS Equity100',
        'SRS_EQUITY_100': 'SRS Equity100',
        'SRS_CASH_PLUS': 'SRS Cash SGD flexi',
        'SRS_CASH': 'SRS Cash SGD flexi',
        'DOWNSIDE_PROTECTED': 'Downside Protected',
        'DOWNSIDE': 'Downside Protected',
        'CUSTOM_USD': 'Custom USD',
        'CUSTOM': 'Custom USD',
        # silverdale funds - various maturity dates
        'SILVERDALE': 'Silverdale Fund',
        'SILVER_DALE': 'Silverdale Fund',
    }
    
    def get_ticker(port_type):
        """map portfolio type to ticker for asset class lookup"""
        upper = str(port_type).upper().replace(' ', '_').replace('-', '_')
        # remove common suffixes
        for suffix in ['_ACTIVE_NEW', '_ACTIVE', '_INACTIVE', '_CLOSED']:
            if upper.endswith(suffix):
                upper = upper[:-len(suffix)]
        
        # try exact match first
        if upper in PORTFOLIO_TICKER_MAP:
            return PORTFOLIO_TICKER_MAP[upper]
        
        # try contains match
        if 'BLACKROCK' in upper:
            return 'Blackrock Private Credit Fund'
        if 'MONROE' in upper:
            return 'Monroe Income Plus Fund'
        if 'DOWNSIDE' in upper:
            return 'Downside Protected'
        if 'SILVER' in upper or 'SILVERDALE' in upper:
            # extract date from name (e.g., "Silver Dale Usd Feb2029" -> "Silverdale Fund Feb2029")
            import re
            # try to find month and year pattern
            date_match = re.search(r'(NOV|DEC|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT)\s*(\d{4})', upper)
            if date_match:
                month = date_match.group(1).title()
                year = date_match.group(2)
                return f'Silverdale Fund {month}{year}'  # match excel format: "Silverdale Fund Feb2029"
            # default to the original name if no date pattern found
            return str(portfolio_type).replace('_', ' ').title()
        if 'REIT' in upper:
            return 'REIT 100'
        if 'INCOME_ENHANCE' in upper:
            return 'Income Enhance'
        if 'INCOME_PRESERVE' in upper:
            return 'Income Preserve'
        if 'INCOME' in upper:
            return 'Income Enhance'
        if 'CHINA' in upper:
            return 'China Growth'
        if 'ESG' in upper:
            return 'ESG & Clean Energy'
        if 'HEALTHCARE' in upper:
            return 'Healthcare Innovation'
        if 'DISRUPTIVE' in upper:
            return 'Disruptive Technology'
        if 'EQUITY100' in upper or 'EQUITY_100' in upper:
            return 'Core Equity100'
        if 'CORE' in upper:
            return 'Core Equity100'
        if 'CASH' in upper:
            return 'Cash SGD flexi'
        
        # try the original type as ticker name
        return port_type.replace('_', ' ').title()
    
    def get_classification(portfolio_type):
        """look up asset class from df_asset_class excel sheet, returns 'Unknown' if not found"""
        if df_asset_class is None or df_asset_class.empty:
            return 'Unknown'
        
        # get ticker name for lookup
        ticker = get_ticker(portfolio_type)
        
        # check if ticker exists in asset class columns (exact match)
        if ticker in df_asset_class.columns:
            # find which asset class has value 1 for this ticker
            asset_classes = df_asset_class.index[df_asset_class[ticker] == 1].tolist()
            if asset_classes:
                return asset_classes[0]  # return the first matching asset class
        
        # try original type name directly
        orig_name = str(portfolio_type).replace('_', ' ').title()
        if orig_name in df_asset_class.columns:
            asset_classes = df_asset_class.index[df_asset_class[orig_name] == 1].tolist()
            if asset_classes:
                return asset_classes[0]
        
        # try case-insensitive and partial matching
        ticker_lower = ticker.lower() if ticker else ''
        orig_lower = orig_name.lower()
        
        for col in df_asset_class.columns:
            col_lower = col.lower()
            # check if ticker or orig_name matches column (case-insensitive)
            if col_lower == ticker_lower or col_lower == orig_lower:
                asset_classes = df_asset_class.index[df_asset_class[col] == 1].tolist()
                if asset_classes:
                    return asset_classes[0]
            # check if column contains or is contained by the ticker (partial match)
            if ticker_lower and (ticker_lower in col_lower or col_lower in ticker_lower):
                asset_classes = df_asset_class.index[df_asset_class[col] == 1].tolist()
                if asset_classes:
                    return asset_classes[0]
        
        # fallback for types not in excel: CUSTOM should be Specialized / alternatives
        upper = str(portfolio_type).upper()
        if any(x in upper for x in ['CUSTOM', 'STRATEGIC', 'ALTERNATIVE']):
            return 'Specialized / alternatives'
        
        # growth portfolios default to Equity
        if any(x in upper for x in ['GROWTH', 'CORE', 'EQUITY', 'BALANCED', 'DEFENSIVE', 
                                     'CHINA', 'ESG', 'HEALTHCARE', 'DISRUPTIVE', 'TECHNOLOGY']):
            return 'Equity'
        
        # REIT portfolios
        if 'REIT' in upper:
            return 'REITs'
        
        # income portfolios
        if 'INCOME' in upper:
            return 'Fixed income'
        
        return 'Unknown'
    
    # risk rating - look up from df_risk_rating (Risk sheet in excel)
    def get_risk_rating(portfolio_type):
        """look up risk rating from df_risk_rating excel sheet, returns 2.0 if not found"""
        # special case for CUSTOM_USD - hardcoded to 3.0
        upper = str(portfolio_type).upper()
        if 'CUSTOM_USD' in upper or 'CUSTOM USD' in upper:
            return 3.0
        
        if df_risk_rating is None or df_risk_rating.empty:
            return 2.0  # default fallback
        
        # get ticker name for lookup (same mapping as asset class)
        ticker = get_ticker(portfolio_type)
        
        # check if ticker exists in risk rating index
        if ticker in df_risk_rating.index:
            try:
                risk_val = float(df_risk_rating.loc[ticker, 'Risk'])
                return round(risk_val, 1)
            except:
                pass
        
        # try original type name directly (with title case)
        orig_name = str(portfolio_type).replace('_', ' ').title()
        if orig_name in df_risk_rating.index:
            try:
                risk_val = float(df_risk_rating.loc[orig_name, 'Risk'])
                return round(risk_val, 1)
            except:
                pass
        
        # try lowercase matching
        for idx in df_risk_rating.index:
            if idx.lower() == ticker.lower() or idx.lower() == orig_name.lower():
                try:
                    risk_val = float(df_risk_rating.loc[idx, 'Risk'])
                    return round(risk_val, 1)
                except:
                    pass
        
        return 0.0  # return 0 if not found
    
    # categorize portfolios into 3 groups
    def get_portfolio_category(ptype, asset_class=None):
        """categorise portfolio based on asset class (from excel) or portfolio type name"""
        upper = str(ptype).upper()
        
        # first check asset class - if it's Specialized/alternatives, always put in alternatives
        if asset_class and 'alternative' in str(asset_class).lower():
            return 'alternatives'
        
        # growth: Equity, Core, Thematic
        if any(x in upper for x in ['EQUITY', 'GROWTH', 'BALANCED', 'DEFENSIVE', 
                                    'TECHNOLOGY', 'HEALTHCARE', 'ESG', 'CHINA', 'GLOBAL', 'DISRUPTIVE']):
            return 'growth'
        
        # alternatives: Blackrock, Monroe, Custom, Strategic, Downside
        if any(x in upper for x in ['BLACKROCK', 'MONROE', 'CUSTOM', 'STRATEGIC', 'ALTERNATIVE', 'DOWNSIDE']):
            return 'alternatives'
        
        # income & preservation: REITs, Income, Cash/MMF
        return 'income'
    
    # build table data - P&L with returns in brackets
    header = ['Portfolio ID', 'Portfolio Type', 'Asset Class', 'Risk rating (0-5)', 
              'Inception date', 'Current Value (SGD)', 'Allocation', 'P&L Inception']
    
    table_data = [header]
    
    growth_rows = []
    income_rows = []
    alternatives_rows = []
    
    for _, row in df.iterrows():
        # convert portfolio_id to int first to remove decimal, then to string
        raw_port_id = row.get('portfolio_id', '')
        try:
            port_id = str(int(float(raw_port_id)))[:12]
        except (ValueError, TypeError):
            port_id = str(raw_port_id)[:12]
        port_type = row.get('type', row.get('name', ''))
        
        # clean up portfolio type name
        port_type_display = str(port_type).replace('_', ' ').title()
        
        classification = get_classification(port_type)
        risk = get_risk_rating(port_type)
        
        # get inception date
        inc_date = row.get('activated_date_sgt', row.get('activated_date', ''))
        if pd.notna(inc_date):
            try:
                if isinstance(inc_date, str):
                    inc_date = pd.to_datetime(inc_date).strftime('%b %d, %Y')
                else:
                    inc_date = inc_date.strftime('%b %d, %Y')
            except:
                inc_date = str(inc_date)[:12]
        else:
            inc_date = ''
        
        nav = safe_float(row.get('nav_in_sgd', 0))
        allocation = (nav / total_nav * 100) if total_nav > 0 else 0
        
        # P&L inception - the actual dollar amount from CSV
        pnl = safe_float(row.get('pnl_inception', 0))
        
        # calculate simple returns: PnL / (NAV - PnL) * 100
        cost_basis = nav - pnl
        if cost_basis > 0:
            returns_pct = (pnl / cost_basis) * 100
        else:
            returns_pct = 0
        
        # combine P&L dollar and returns % in one column
        pnl_with_returns = f"${pnl:,.2f} ({returns_pct:.2f}%)"
        
        row_data = [
            port_id,
            port_type_display,
            classification,
            f"{risk:.1f}",
            inc_date,
            f"{nav:,.2f}",
            f"{allocation:.2f}%",
            pnl_with_returns
        ]
        
        # pass asset class to categorization so Specialized/alternatives goes to Alternatives section
        category = get_portfolio_category(port_type, classification)
        if category == 'growth':
            growth_rows.append((row_data, nav, allocation, pnl, returns_pct))
        elif category == 'alternatives':
            alternatives_rows.append((row_data, nav, allocation, pnl, returns_pct))
        else:
            income_rows.append((row_data, nav, allocation, pnl, returns_pct))
    
    # sort each group by value
    growth_rows.sort(key=lambda x: -x[1])
    income_rows.sort(key=lambda x: -x[1])
    alternatives_rows.sort(key=lambda x: -x[1])
    
    # calculate growth subtotals
    growth_total_nav = sum(r[1] for r in growth_rows)
    growth_total_alloc = sum(r[2] for r in growth_rows)
    growth_avg_risk = 4.0
    growth_total_pnl = sum(r[3] for r in growth_rows)
    growth_cost = growth_total_nav - growth_total_pnl
    growth_returns = (growth_total_pnl / growth_cost * 100) if growth_cost > 0 else 0
    
    # calculate income & preservation subtotals
    income_total_nav = sum(r[1] for r in income_rows)
    income_total_alloc = sum(r[2] for r in income_rows)
    income_avg_risk = 1.4
    income_total_pnl = sum(r[3] for r in income_rows)
    income_cost = income_total_nav - income_total_pnl
    income_returns = (income_total_pnl / income_cost * 100) if income_cost > 0 else 0
    
    # calculate alternatives subtotals
    alt_total_nav = sum(r[1] for r in alternatives_rows)
    alt_total_alloc = sum(r[2] for r in alternatives_rows)
    alt_avg_risk = 2.0
    alt_total_pnl = sum(r[3] for r in alternatives_rows)
    alt_cost = alt_total_nav - alt_total_pnl
    alt_returns = (alt_total_pnl / alt_cost * 100) if alt_cost > 0 else 0
    
    # === GROWTH PORTFOLIOS SECTION ===
    # add subtotal row first (at top of section)
    if growth_rows:
        growth_pnl_combined = f"${growth_total_pnl:,.2f} ({growth_returns:.2f}%)"
        table_data.append([
            'Growth Portfolios', '', '', f"{growth_avg_risk:.1f}", '',
            f"{growth_total_nav:,.2f}", f"{growth_total_alloc:.2f}%",
            growth_pnl_combined
        ])
    # then add individual growth portfolios
    for row_data, _, _, _, _ in growth_rows:
        table_data.append(row_data)
    
    # === INCOME & PRESERVATION SECTION ===
    # add subtotal row first (at top of section)
    if income_rows:
        income_pnl_combined = f"${income_total_pnl:,.2f} ({income_returns:.2f}%)"
        table_data.append([
            'Income & Preservation', '', '', f"{income_avg_risk:.1f}", '',
            f"{income_total_nav:,.2f}", f"{income_total_alloc:.2f}%",
            income_pnl_combined
        ])
    # then add individual income portfolios
    for row_data, _, _, _, _ in income_rows:
        table_data.append(row_data)
    
    # === ALTERNATIVES SECTION ===
    # add subtotal row first (at top of section)
    if alternatives_rows:
        alt_pnl_combined = f"${alt_total_pnl:,.2f} ({alt_returns:.2f}%)"
        table_data.append([
            'Alternatives', '', '', f"{alt_avg_risk:.1f}", '',
            f"{alt_total_nav:,.2f}", f"{alt_total_alloc:.2f}%",
            alt_pnl_combined
        ])
    # then add individual alternatives portfolios
    for row_data, _, _, _, _ in alternatives_rows:
        table_data.append(row_data)
    
    # add total row (includes all 3 sections)
    grand_total = growth_total_nav + income_total_nav + alt_total_nav
    grand_total_pnl = growth_total_pnl + income_total_pnl + alt_total_pnl
    grand_cost = grand_total - grand_total_pnl
    grand_returns = (grand_total_pnl / grand_cost * 100) if grand_cost > 0 else 0
    # weighted average risk
    risk_weighted = (growth_avg_risk * growth_total_nav + income_avg_risk * income_total_nav + alt_avg_risk * alt_total_nav)
    avg_risk = risk_weighted / grand_total if grand_total > 0 else 0
    
    grand_pnl_combined = f"${grand_total_pnl:,.2f} ({grand_returns:.2f}%)"
    table_data.append([
        'Total', '', '', f"{avg_risk:.1f}", '',
        f"{grand_total:,.2f}", '100.00%',
        grand_pnl_combined
    ])
    
    # calculate column widths (8 columns - combined P&L with returns)
    col_widths = [
        0.10 * page_width,  # portfolio ID
        0.15 * page_width,  # portfolio type
        0.10 * page_width,  # classification
        0.10 * page_width,  # risk rating
        0.12 * page_width,  # inception date
        0.12 * page_width,  # value
        0.10 * page_width,  # allocation
        0.21 * page_width,  # P&L inception (with returns %)
    ]
    
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # build style commands
    style_cmds = [
        # header styling
        ('BACKGROUND', (0, 0), (-1, 0), SYFE_COLORS['dark_navy']),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        
        # cell padding
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        
        # data rows styling
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), SYFE_COLORS['text_dark']),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        
        # center align all columns
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        
        # grid
        ('GRID', (0, 0), (-1, -1), 0.5, SYFE_COLORS['medium_grey']),
    ]
    
    # find and style subtotal/total rows (section headers)
    for i, row in enumerate(table_data[1:], start=1):
        first_cell = str(row[0])
        # section headers: Growth, Alternatives, Income & Preservation - all get light blue
        if first_cell in ['Growth Portfolios', 'Alternatives', 'Income & Preservation']:
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), SYFE_COLORS['light_blue']))
            style_cmds.append(('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold'))
            style_cmds.append(('FONTSIZE', (0, i), (-1, i), 7))
        elif first_cell == 'Total':
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), SYFE_COLORS['dark_navy']))
            style_cmds.append(('TEXTCOLOR', (0, i), (-1, i), colors.white))
            style_cmds.append(('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold'))
    
    table.setStyle(TableStyle(style_cmds))
    return table

# ==========================================
# PDF UTILITIES
# ==========================================

def overlay_content_on_template(content_pdf_path, template_pdf_path, output_path):
    """
    overlay content PDF pages onto template background
    """
    if not os.path.exists(template_pdf_path):
        # no template, just copy content
        import shutil
        shutil.copy(content_pdf_path, output_path)
        return
    
    template_reader = PdfReader(template_pdf_path)
    content_reader = PdfReader(content_pdf_path)
    writer = PdfWriter()
    
    for content_page in content_reader.pages:
        # re-read template for each page to avoid mutation issues
        template_reader_fresh = PdfReader(template_pdf_path)
        template_page = template_reader_fresh.pages[0]
        # merge content on top of template
        template_page.merge_page(content_page)
        writer.add_page(template_page)
    
    with open(output_path, "wb") as f:
        writer.write(f)

def merge_pdfs(pdf_list, output_path):
    """merge multiple PDFs into one"""
    writer = PdfWriter()
    
    for pdf_path in pdf_list:
        if pdf_path and os.path.exists(pdf_path):
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                writer.add_page(page)
    
    with open(output_path, "wb") as f:
        writer.write(f)

def add_page_numbers(input_pdf, output_pdf, page_size=landscape(A4), skip_first=True, skip_last=False, client_id=None):
    """
    adds 'Page X of Y' to the bottom right, date/time and Client ID to top right of each page,
    now includes appendix page (skip_last=False by default)
    """
    page_width, page_height = page_size
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    total_pages = len(reader.pages)
    
    # calculate how many content pages (excluding cover only)
    content_pages = total_pages
    if skip_first:
        content_pages -= 1
    if skip_last:
        content_pages -= 1
    
    # track which content page we're on
    content_page_num = 0
    
    # get current date/time for generation timestamp in SGT (Singapore Time, UTC+8)
    from zoneinfo import ZoneInfo
    sgt = ZoneInfo('Asia/Singapore')
    generated_datetime = datetime.now(sgt).strftime("%d %b %Y, %H:%M")
    
    for page_idx, page in enumerate(reader.pages):
        # skip cover page (first) only
        if skip_first and page_idx == 0:
            writer.add_page(page)
            continue
        
        # skip appendix if skip_last is True
        if skip_last and page_idx == total_pages - 1:
            writer.add_page(page)
            continue
        
        # increment content page counter
        content_page_num += 1
        page_num = content_page_num
        
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=page_size)
        can.setFont("Helvetica", 7)
        can.setFillColor(SYFE_COLORS['text_grey'])
        
        # page number at bottom right
        text = f"Page {page_num} of {content_pages}"
        text_width = can.stringWidth(text, "Helvetica", 7)
        can.drawString(page_width - 50 - text_width, 15, text)
        
        # date/time at top right with "Generated at:" prefix
        can.setFont("Helvetica", 6)
        gen_text = f"Generated at: {generated_datetime}"
        gen_text_width = can.stringWidth(gen_text, "Helvetica", 6)
        can.drawString(page_width - 50 - gen_text_width, page_height - 25, gen_text)
        
        # client ID below date/time
        if client_id:
            client_text = f"Client ID: {client_id}"
            client_text_width = can.stringWidth(client_text, "Helvetica", 6)
            can.drawString(page_width - 50 - client_text_width, page_height - 35, client_text)
        
        can.save()
        
        packet.seek(0)
        number_pdf = PdfReader(packet)
        page.merge_page(number_pdf.pages[0])
        writer.add_page(page)
    
    with open(output_pdf, "wb") as f:
        writer.write(f)

# ==========================================
# EXPOSURE CALCULATION
# ==========================================

def calculate_portfolio_exposure(client_weights, df_exposure_matrix):
    """
    calculate weighted exposure using matrix multiplication
    """
    if isinstance(client_weights, dict):
        weights = pd.Series(client_weights)
    else:
        weights = client_weights
    
    if weights.empty or df_exposure_matrix.empty:
        return pd.Series()
    
    # find common tickers
    common_tickers = weights.index.intersection(df_exposure_matrix.columns)
    
    if len(common_tickers) == 0:
        return pd.Series()
    
    # align and calculate
    weights_aligned = weights[common_tickers]
    matrix_aligned = df_exposure_matrix[common_tickers]
    
    # ensure numeric
    matrix_aligned = matrix_aligned.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # matrix multiplication
    exposure = matrix_aligned.dot(weights_aligned)
    
    # normalize
    if exposure.sum() > 0:
        exposure = exposure / exposure.sum()
    
    # sort descending
    exposure = exposure.sort_values(ascending=False)
    
    return exposure

# ==========================================
# MAIN REPORT GENERATOR CLASS
# ==========================================

class PortfolioReportGenerator:
    """
    main class for generating portfolio summary PDF reports
    """
    
    def __init__(self, client_id, report_date=None, data_path='Data/'):
        self.client_id = str(client_id)
        self.report_date = report_date or date.today()
        self.data_path = data_path
        
    def _get_template_path(self, filename):
        return os.path.join(self.data_path, filename)
    
    def _add_client_id_to_cover(self, cover_path, output_path, page_size):
        """add client ID text to the cover page"""
        page_width, page_height = page_size
        
        # read the cover PDF
        reader = PdfReader(cover_path)
        writer = PdfWriter()
        
        # create overlay with client ID
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=page_size)
        
        # add client ID text - positioned below where "Portfolio Summary" would be
        # adjust position based on your cover page layout
        can.setFont("Helvetica", 10)
        can.setFillColor(SYFE_COLORS['text_dark'])
        client_text = f"Client ID: {self.client_id}"
        # center the text horizontally, position it in lower third of page
        text_width = can.stringWidth(client_text, "Helvetica", 10)
        can.drawString((page_width - text_width) / 2, page_height * 0.35, client_text)
        
        can.save()
        packet.seek(0)
        
        # merge overlay with cover page
        overlay_pdf = PdfReader(packet)
        cover_page = reader.pages[0]
        cover_page.merge_page(overlay_pdf.pages[0])
        writer.add_page(cover_page)
        
        # write output
        with open(output_path, "wb") as f:
            writer.write(f)
    
    def generate_report(
        self,
        df_holdings,
        exposure_data=None,
        returns_data=None,
        dividends_data=None,
        cashflow_data=None,
        cashflow_currency="SGD",
        output_filename=None,
        include_cover=True,
        include_appendix=True,
        df_asset_class=None,
        df_risk_rating=None
    ):
        """
        generate the complete portfolio report PDF
        
        Args:
            df_holdings: DataFrame with portfolio holdings data
            exposure_data: Dict with 'asset_class', 'country', 'sector' Series
            returns_data: Series with portfolio returns for bar chart
            dividends_data: Series with dividends earned by portfolio
            cashflow_data: Dict with monthly cash flow projections
            cashflow_currency: Currency for cash flow display
            output_filename: Output PDF filename
            include_cover: Whether to include cover page
            include_appendix: Whether to include appendix page
            df_asset_class: DataFrame with asset class matrix from Excel
            df_risk_rating: DataFrame with risk ratings from Excel
        
        Returns:
            str: Path to generated PDF file
        """
        if output_filename is None:
            output_filename = f"Client_{self.client_id}_Portfolio_Summary_{self.report_date.strftime('%Y%m%d')}.pdf"
        
        # page setup
        page_size = landscape(A4)
        page_width = page_size[0] - 0.8 * inch
        
        all_elements = []
        styles = get_syfe_styles()
        
        # ==========================================
        # PAGE 1: PORTFOLIO SUMMARY + EXPOSURE CHARTS
        # ==========================================
        
        all_elements.append(Paragraph("Portfolio Summary", styles['section_header']))
        
        # holdings table (grouped format)
        if df_holdings is not None and not df_holdings.empty:
            table = create_grouped_portfolio_table(df_holdings, page_width, df_asset_class, df_risk_rating)
            if table:
                all_elements.append(table)
        
        all_elements.append(Spacer(1, 0.25 * inch))
        
        # ==========================================
        # EXPOSURE CHARTS (Asset Class + Country side by side)
        # ==========================================
        
        chart_row = []
        
        # asset class exposure (donut)
        if exposure_data and 'asset_class' in exposure_data:
            asset_data = exposure_data['asset_class']
            if isinstance(asset_data, pd.Series) and not asset_data.empty:
                asset_chart = create_donut_chart_drawing(
                    asset_data, 
                    title="Asset Class Exposure",
                    width=380, 
                    height=180
                )
                if asset_chart:
                    chart_row.append(asset_chart)
        
        # country exposure (donut - same style as sector)
        if exposure_data and 'country' in exposure_data:
            country_data = exposure_data['country']
            if isinstance(country_data, pd.Series) and not country_data.empty:
                country_chart = create_donut_chart_drawing(
                    country_data,
                    title="Country Exposure",
                    width=380,
                    height=180
                )
                if country_chart:
                    chart_row.append(country_chart)
        
        if chart_row:
            if len(chart_row) == 2:
                chart_table = Table([chart_row], colWidths=[380, 380])
            else:
                chart_table = Table([chart_row], colWidths=[page_width])
            
            chart_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ]))
            all_elements.append(chart_table)
        
        all_elements.append(PageBreak())
        
        # ==========================================
        # PAGE 2: SECTOR EXPOSURE + DIVIDENDS EARNED
        # ==========================================
        
        page2_chart_row = []
        
        # sector exposure (donut)
        if exposure_data and 'sector' in exposure_data:
            sector_data = exposure_data['sector']
            if isinstance(sector_data, pd.Series) and not sector_data.empty:
                sector_chart = create_donut_chart_drawing(
                    sector_data,
                    title="Sector Exposure",
                    width=380,
                    height=220
                )
                if sector_chart:
                    page2_chart_row.append(sector_chart)
        
        # dividends bar chart (X Portfolios Contributing to Your Dividends)
        if dividends_data is not None and len(dividends_data) > 0:
            div_chart = create_dividends_bar_chart_drawing(
                dividends_data,
                title="",  # auto-generates "X Portfolios Contributing to Your Dividends"
                width=380,  # same width as sector exposure
                height=220  # same height as sector exposure for alignment
            )
            if div_chart:
                page2_chart_row.append(div_chart)
        
        if page2_chart_row:
            if len(page2_chart_row) == 2:
                page2_table = Table([page2_chart_row], colWidths=[380, 380])
            else:
                page2_table = Table([page2_chart_row], colWidths=[page_width])
            
            page2_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ]))
            all_elements.append(page2_table)
            all_elements.append(Spacer(1, 0.2 * inch))
        
        # cash flow projections chart - centered on page
        if cashflow_data is not None and len(cashflow_data) > 0:
            all_elements.append(Spacer(1, 0.1 * inch))
            cashflow_chart = create_cashflow_chart_drawing(
                cashflow_data,
                currency=cashflow_currency,
                width=page_width,  # use full page width
                height=220
            )
            if cashflow_chart:
                # wrap in a table to center it on the page
                cashflow_table = Table([[cashflow_chart]], colWidths=[page_width])
                cashflow_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                all_elements.append(cashflow_table)
            
            # add cash flow summary table below the chart
            all_elements.append(Spacer(1, 0.15 * inch))
            
            # table title
            cashflow_table_title = ParagraphStyle(
                name='CashflowTableTitle',
                fontName='Helvetica-Bold',
                fontSize=10,
                textColor=SYFE_COLORS['text_dark'],
                alignment=1,  # center
                spaceBefore=5,
                spaceAfter=8
            )
            all_elements.append(Paragraph("Monthly Cash Flow Summary by Portfolio", cashflow_table_title))
            
            # create the table
            cf_summary_table = create_cashflow_table(
                cashflow_data,
                currency=cashflow_currency,
                page_width=page_width - 20  # slight margin
            )
            if cf_summary_table:
                all_elements.append(cf_summary_table)
        else:
            # no cash flow generating portfolios - show message (left-aligned)
            all_elements.append(Spacer(1, 0.2 * inch))
            no_cashflow_style = ParagraphStyle(
                name='NoCashflow',
                fontName='Helvetica',
                fontSize=11,
                textColor=SYFE_COLORS['text_grey'],
                alignment=0,  # left-aligned
                spaceBefore=20,
                spaceAfter=20
            )
            all_elements.append(Paragraph("Client has no cashflow generating portfolios", no_cashflow_style))
        
        # ==========================================
        # BUILD PDF
        # ==========================================
        
        content_pdf = "temp_content.pdf"
        doc = SimpleDocTemplate(
            content_pdf,
            pagesize=page_size,
            leftMargin=0.4 * inch,
            rightMargin=0.4 * inch,
            topMargin=1.0 * inch,  # increased to leave room for Syfe logo watermark
            bottomMargin=0.4 * inch
        )
        
        try:
            doc.build(all_elements)
        except Exception as e:
            print(f"Error building PDF: {e}")
            raise
        
        # overlay content on template background
        template_path = self._get_template_path("Portfolio_Visualiser_Template_Landscape.pdf")
        templated_pdf = "temp_templated.pdf"
        
        if os.path.exists(template_path):
            overlay_content_on_template(content_pdf, template_path, templated_pdf)
        else:
            templated_pdf = content_pdf
        
        # merge with cover and appendix
        pdfs_to_merge = []
        
        cover_path = self._get_template_path("Portfolio_Visualiser_Cover.pdf")
        if include_cover and os.path.exists(cover_path):
            # add client ID to cover page
            cover_with_client = "temp_cover_with_client.pdf"
            self._add_client_id_to_cover(cover_path, cover_with_client, page_size)
            pdfs_to_merge.append(cover_with_client)
        
        pdfs_to_merge.append(templated_pdf)
        
        # use the new appendix from PDF Format folder
        appendix_path = "PDF Format/Portfolio_Visualiser_Appendix.pdf"
        if include_appendix and os.path.exists(appendix_path):
            pdfs_to_merge.append(appendix_path)
        
        merged_pdf = "temp_merged.pdf"
        merge_pdfs(pdfs_to_merge, merged_pdf)
        
        # add page numbers and client ID (include appendix in page numbering)
        add_page_numbers(
            merged_pdf, 
            output_filename, 
            page_size=page_size,
            skip_first=include_cover and os.path.exists(cover_path),
            skip_last=False,  # include appendix in page numbering
            client_id=self.client_id
        )
        
        # cleanup
        temp_files = [content_pdf, templated_pdf, merged_pdf, "temp_cover_with_client.pdf"]
        for f in temp_files:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        
        return output_filename


# ==========================================
# CONVENIENCE FUNCTIONS
# ==========================================

def prepare_holdings_dataframe(df_raw):
    """
    prepare holdings DataFrame for PDF display,
    returns the raw dataframe - formatting is done in create_grouped_portfolio_table
    """
    return df_raw.copy() if df_raw is not None else pd.DataFrame()


def prepare_returns_data(df_raw):
    """
    prepare returns data for bar chart,
    uses pnl_inception and nav_in_sgd to calculate returns percentage
    returns = pnl_inception / (nav_in_sgd - pnl_inception) * 100
    """
    if df_raw is None or df_raw.empty:
        return pd.Series()
    
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.strip()
    
    # helper to safely convert to float
    def safe_float(val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace(',', ''))
        except:
            return 0.0
    
    # get name column
    name_col = None
    for col in ['name', 'type', 'portfolio_type']:
        if col in df.columns:
            name_col = col
            break
    
    if name_col is None:
        return pd.Series()
    
    # calculate returns from pnl_inception and nav
    if 'pnl_inception' in df.columns and 'nav_in_sgd' in df.columns:
        returns_dict = {}
        for _, row in df.iterrows():
            name = str(row[name_col]).replace('_', ' ').title()
            pnl = safe_float(row.get('pnl_inception', 0))
            nav = safe_float(row.get('nav_in_sgd', 0))
            
            # cost basis = NAV - PnL
            cost_basis = nav - pnl
            
            # returns = PnL / cost basis (as decimal, e.g., 0.05 = 5%)
            if cost_basis > 0:
                returns_pct = pnl / cost_basis
            else:
                returns_pct = 0
            
            returns_dict[name] = returns_pct
        
        return pd.Series(returns_dict)
    
    return pd.Series()


def prepare_dividends_data(df_raw):
    """
    prepare dividends data for bar chart,
    uses dividends_reinvested column from CSV,
    aggregates by portfolio type (sums dividends for same type)
    """
    if df_raw is None or df_raw.empty:
        return pd.Series()
    
    df = df_raw.copy()
    # normalize column names - lowercase, strip whitespace, replace spaces with underscores
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # helper to safely convert to float
    def safe_float(val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace(',', ''))
        except:
            return 0.0
    
    # get name/type column (prefer 'type' for aggregation)
    name_col = None
    for col in ['type', 'name', 'portfolio_type']:
        if col in df.columns:
            name_col = col
            break
    
    if name_col is None:
        return pd.Series()
    
    # find dividends column - check for various naming conventions
    div_col = None
    possible_div_cols = [
        'dividend_reinvested',  # this is the actual column name in the CSV
        'dividends_reinvested', 'dividends', 'dividend', 'div_reinvested',
        'reinvested_dividends', 'total_dividends',
        'dividends_earned', 'dividend_earned'
    ]
    for col in possible_div_cols:
        if col in df.columns:
            div_col = col
            break
    
    # also check for partial matches
    if div_col is None:
        for col in df.columns:
            if 'dividend' in col.lower():
                div_col = col
                break
    
    if div_col is None:
        print(f"[DEBUG] No dividends column found. Available columns: {list(df.columns)}")
        return pd.Series()
    
    print(f"[DEBUG] Found dividends column: {div_col}")
    
    # convert dividends column to numeric
    df['div_numeric'] = df[div_col].apply(safe_float)
    
    # aggregate dividends by portfolio type
    df['type_clean'] = df[name_col].apply(lambda x: str(x).replace('_', ' ').title())
    aggregated = df.groupby('type_clean')['div_numeric'].sum()
    
    # filter out zero or negative values
    aggregated = aggregated[aggregated > 0]
    
    print(f"[DEBUG] Aggregated dividends data: {aggregated.to_dict()}")
    return aggregated


def prepare_cashflow_data(df_raw, df_fi_metrics=None):
    """
    prepare monthly cash flow projections based on portfolio yields from df_fi_metrics
    
    INCLUDES: REITs, Income portfolios, Alternatives (Blackrock, Monroe)
    EXCLUDES: Cash/MMF, ALL Growth portfolios, CUSTOM portfolios
    
    Returns:
        dict: keys are portfolio types, values are arrays of 12 monthly amounts
    """
    if df_raw is None or df_raw.empty:
        return None
    
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    def safe_float(val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace(',', '').replace('%', ''))
        except:
            return 0.0
    
    # get name/type column
    name_col = None
    for col in ['type', 'name', 'portfolio_type']:
        if col in df.columns:
            name_col = col
            break
    
    if name_col is None:
        return None
    
    # find NAV column
    nav_col = None
    for col in ['nav_in_sgd', 'nav', 'value', 'balance']:
        if col in df.columns:
            nav_col = col
            break
    
    if nav_col is None:
        return None
    
    # portfolio type to ticker mapping for yield lookup
    PORTFOLIO_TICKER_MAP = {
        'INCOME_ENHANCE': 'Income Enhance',
        'INCOME_ENHANCED': 'Income Enhance',
        'INCOME_PRESERVE': 'Income Preserve',
        'INCOME': 'Income Enhance',
        'REIT': 'REIT 100',
        'REIT_100': 'REIT 100',
        'REIT_RISK_MANAGED': 'REIT Risk Managed',
        'BLACKROCK': 'Blackrock Private Credit Fund',
        'BLACKROCK_PRIVATE_CREDIT': 'Blackrock Private Credit Fund',
        'MONROE': 'Monroe Income Plus Fund',
        'MONROE_CAPITAL': 'Monroe Income Plus Fund',
        'MONROE_INCOME_PLUS': 'Monroe Income Plus Fund',
        'SRS_INCOME_ENHANCE': 'SRS Income Enhance',
        'SRS_INCOME_ENHANCED': 'SRS Income Enhance',
    }
    
    # portfolio types to exclude from cash flow (no cash flow analysis)
    EXCLUDED_PATTERNS = [
        'CASH', 'MMF', 'FLEXI', 'GUARANTEED',  # cash/MMF
        'CORE', 'EQUITY', 'GROWTH',  # growth portfolios
        'CHINA', 'ESG', 'HEALTHCARE', 'DISRUPTIVE', 'TECHNOLOGY', 'THEMATIC',  # more growth
        'CUSTOM', 'STRATEGIC',  # custom portfolios excluded
    ]
    
    def should_exclude(port_type):
        """check if portfolio should be excluded from cash flow analysis"""
        upper = str(port_type).upper()
        for pattern in EXCLUDED_PATTERNS:
            if pattern in upper:
                return True
        return False
    
    def should_include(port_type):
        """check if portfolio should be included in cash flow analysis"""
        upper = str(port_type).upper()
        # include: REITs, Income, Blackrock, Monroe, Silverdale
        include_patterns = ['REIT', 'INCOME', 'BLACKROCK', 'MONROE', 'SILVER', 'SILVERDALE']
        for pattern in include_patterns:
            if pattern in upper:
                return True
        return False
    
    def get_ticker(port_type):
        """map portfolio type to ticker for yield lookup"""
        upper = str(port_type).upper().replace(' ', '_').replace('-', '_')
        # remove common suffixes
        for suffix in ['_ACTIVE_NEW', '_ACTIVE', '_INACTIVE', '_CLOSED']:
            if upper.endswith(suffix):
                upper = upper[:-len(suffix)]
        
        # try exact match first
        if upper in PORTFOLIO_TICKER_MAP:
            return PORTFOLIO_TICKER_MAP[upper]
        
        # try contains match for Blackrock/Monroe/Silverdale
        if 'BLACKROCK' in upper:
            return 'Blackrock Private Credit Fund'
        if 'MONROE' in upper:
            return 'Monroe Income Plus Fund'
        if 'SILVER' in upper or 'SILVERDALE' in upper:
            # extract date from name (e.g., "Silver Dale Usd Feb2029" -> "Silverdale Fund Feb2029")
            import re
            date_match = re.search(r'(NOV|DEC|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT)\s*(\d{4})', upper)
            if date_match:
                month = date_match.group(1).title()
                year = date_match.group(2)
                return f'Silverdale Fund {month}{year}'  # match excel format: "Silverdale Fund Feb2029"
            return 'Silverdale Fund'
        if 'REIT' in upper:
            return 'REIT 100'
        if 'INCOME_ENHANCE' in upper or 'INCOME ENHANCE' in upper:
            return 'Income Enhance'
        if 'INCOME' in upper:
            return 'Income Enhance'
        
        return None
    
    def get_yield_from_fi_metrics(ticker):
        """get annual yield from df_fi_metrics (yields are in % format)"""
        if df_fi_metrics is None or ticker is None:
            return 0.0
        
        if ticker not in df_fi_metrics.columns:
            return 0.0
        
        annual_yield = 0.0
        
        # first try T12M dividend yield
        if 'T12M dividend yield' in df_fi_metrics.index:
            yield_val = df_fi_metrics.loc['T12M dividend yield', ticker]
            if pd.notna(yield_val) and yield_val != 0:
                annual_yield = float(yield_val) / 100  # convert % to decimal
        
        # fall back to YTM / est. yield if no dividend yield
        if annual_yield == 0 and 'YTM / est. yield' in df_fi_metrics.index:
            yield_val = df_fi_metrics.loc['YTM / est. yield', ticker]
            if pd.notna(yield_val) and yield_val != 0:
                annual_yield = float(yield_val) / 100  # convert % to decimal
        
        return annual_yield
    
    def get_payment_freq(port_type):
        """determine payment frequency"""
        upper = str(port_type).upper()
        if 'REIT' in upper:
            return 'quarterly'
        return 'monthly'
    
    # calculate cash flows
    cashflow_data = {}
    
    for _, row in df.iterrows():
        raw_type = str(row[name_col])
        port_type = raw_type.replace('_', ' ').title()
        nav = safe_float(row.get(nav_col, 0))
        
        # skip portfolios with no value
        if nav < 100:
            continue
        
        # check if portfolio should be included first (takes priority)
        included = should_include(raw_type)
        
        # if not explicitly included, check if excluded
        if not included:
            if should_exclude(raw_type):
                print(f"[DEBUG] Excluding {raw_type} from cash flow (excluded pattern)")
                continue
            print(f"[DEBUG] Skipping {raw_type} from cash flow (not in include list)")
            continue
        
        # get ticker for yield lookup
        ticker = get_ticker(raw_type)
        
        # get yield from FI metrics
        annual_yield = get_yield_from_fi_metrics(ticker)
        
        print(f"[DEBUG] Portfolio {raw_type} -> ticker {ticker} -> yield {annual_yield:.4f}")
        
        # skip zero-yield portfolios
        if annual_yield <= 0:
            continue
        
        # initialize monthly array for this portfolio
        if port_type not in cashflow_data:
            cashflow_data[port_type] = [0.0] * 12
        
        # calculate monthly cash flow
        payment_freq = get_payment_freq(raw_type)
        
        if payment_freq == 'monthly':
            monthly_amount = (nav * annual_yield) / 12
            for month in range(12):
                cashflow_data[port_type][month] += monthly_amount
        else:  # quarterly
            quarterly_amount = (nav * annual_yield) / 4
            for quarter in range(4):
                month = (quarter * 3) + 2  # months 3, 6, 9, 12 (index 2, 5, 8, 11)
                cashflow_data[port_type][month] += quarterly_amount
    
    if not cashflow_data:
        return None
    
    print(f"[DEBUG] Cash flow data portfolios: {list(cashflow_data.keys())}")
    return cashflow_data


def generate_client_report(
    client_id,
    df_csv,
    df_sector=None,
    df_country=None,
    df_asset_class=None,
    report_date=None,
    data_path='Data/',
    output_path=None
):
    """
    convenience function to generate a client report from CSV data
    """
    # prepare holdings table
    df_holdings = prepare_holdings_dataframe(df_csv)
    
    # prepare returns data
    returns_data = prepare_returns_data(df_csv)
    
    # calculate exposures if matrices provided
    exposure_data = {}
    
    df = df_csv.copy()
    df.columns = df.columns.str.lower().str.strip()
    
    # build weights from CSV
    if 'nav_in_sgd' in df.columns and 'type' in df.columns:
        def safe_float(val):
            if pd.isna(val):
                return 0.0
            if isinstance(val, (int, float)):
                return float(val)
            try:
                return float(str(val).replace(',', ''))
            except:
                return 0.0
        
        df['nav_clean'] = df['nav_in_sgd'].apply(safe_float)
        total_nav = df['nav_clean'].sum()
        
        if total_nav > 0:
            weights = df.groupby('type')['nav_clean'].sum() / total_nav
            
            # use portfolio types as asset class exposure for now
            exposure_data['asset_class'] = weights.copy()
    
    # generate report
    generator = PortfolioReportGenerator(
        client_id=client_id,
        report_date=report_date,
        data_path=data_path
    )
    
    return generator.generate_report(
        df_holdings=df_holdings,
        exposure_data=exposure_data if exposure_data else None,
        returns_data=returns_data,
        output_filename=output_path
    )
