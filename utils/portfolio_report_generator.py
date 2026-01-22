"""
Portfolio Report Generator - Creates PDF reports matching Syfe Portfolio Summary format.

Generates professional PDF reports with:
- Cover page from template with Client ID
- Portfolio Summary table with grouping (Section 1)
- Portfolio Returns bar chart
- Asset Class, Country, Sector exposure charts (Section 2)
- Appendix/Disclaimers page from template
- Uses Template_Landscape.pdf as background for each content page
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
    'dark_navy': colors.HexColor("#1a2744"),      # Dark navy (header bg)
    'navy': colors.HexColor("#263159"),           # Navy 
    'medium_blue': colors.HexColor("#4a5568"),    # Medium blue-grey
    'light_blue': colors.HexColor("#c5cfe8"),     # Light blue (subtotal rows)
    'lighter_blue': colors.HexColor("#e8ecf5"),   # Lighter blue (alt rows)
    'white': colors.white,
    'light_grey': colors.HexColor("#f7fafc"),     # Very light grey
    'medium_grey': colors.HexColor("#e2e8f0"),    # Border grey
    'text_dark': colors.HexColor("#2d3748"),      # Dark text
    'text_grey': colors.HexColor("#718096"),      # Grey text
}

# Donut chart colors (navy theme - matching the screenshot)
DONUT_COLORS = [
    colors.HexColor("#1a2744"),  # Dark Navy
    colors.HexColor("#2d4a7c"),  # Navy Blue
    colors.HexColor("#4a6fa5"),  # Medium Blue
    colors.HexColor("#7b9cc7"),  # Light Blue
    colors.HexColor("#a8c5e2"),  # Lighter Blue
    colors.HexColor("#d1e3f5"),  # Very Light Blue
    colors.HexColor("#4a5568"),  # Grey-Blue
    colors.HexColor("#718096"),  # Medium Grey
]

# Bar chart color
BAR_COLOR = colors.HexColor("#7b9cc7")  # Light blue for bars

# ==========================================
# PARAGRAPH STYLES
# ==========================================

def get_syfe_styles():
    """Returns dictionary of styled ParagraphStyles for Syfe branding."""
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
    Creates a native ReportLab donut chart matching the screenshot style.
    Uses Wedge shapes to create the donut effect.
    
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
    
    # Filter small values and normalize
    data = data[data > 0.001]
    total = data.sum()
    data = data / total
    data = data.sort_values(ascending=False)
    
    # Limit to top 7 + Others
    if len(data) > 7:
        top_data = data.head(6)
        others = data.iloc[6:].sum()
        if others > 0:
            top_data = pd.concat([top_data, pd.Series({'Others': others})])
        data = top_data
    
    d = Drawing(width, height)
    
    # Create donut chart using Wedges
    center_x = 100
    center_y = height // 2 + 10
    outer_radius = 60
    inner_radius = 33  # Creates the donut hole
    
    start_angle = 90  # Start from top
    
    for i, (label, value) in enumerate(data.items()):
        # Calculate sweep angle (in degrees)
        sweep_angle = value * 360
        
        # Create outer wedge
        wedge = Wedge(
            center_x, center_y,
            outer_radius,
            start_angle,
            start_angle + sweep_angle,
            radius1=inner_radius,  # Inner radius for donut
            fillColor=DONUT_COLORS[i % len(DONUT_COLORS)],
            strokeColor=colors.white,
            strokeWidth=1
        )
        d.add(wedge)
        
        start_angle += sweep_angle
    
    # Add title at bottom center
    if title:
        d.add(String(width // 2, 10, title,
                     fontName='Helvetica-Bold', fontSize=13,
                     fillColor=SYFE_COLORS['text_dark'],
                     textAnchor='middle'))
    
    # Add legend on the right side
    legend_x = 180
    legend_y = height - 25
    legend_line_height = 18
    
    for i, (label, value) in enumerate(data.items()):
        y_pos = legend_y - (i * legend_line_height)
        
        # Color box
        d.add(Rect(legend_x, y_pos - 4, 12, 12, 
                   fillColor=DONUT_COLORS[i % len(DONUT_COLORS)],
                   strokeColor=None))
        
        # Label - show full name (up to 28 chars)
        pct = value * 100
        label_text = str(label)[:28]
        d.add(String(legend_x + 18, y_pos, label_text,
                     fontName='Helvetica', fontSize=7,
                     fillColor=SYFE_COLORS['text_dark']))
        
        # Percentage on the right
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
    Creates a native ReportLab horizontal bar chart with multi-colored bars.
    Better for handling many portfolios and long names.
    
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
    
    # Sort descending and show all portfolios (up to 25)
    data = data.sort_values(ascending=False).head(25)
    # Reverse for horizontal chart (top item at top)
    data = data.iloc[::-1]
    
    num_bars = len(data)
    
    # Adjust height based on number of bars (minimum 20px per bar)
    min_height = num_bars * 20 + 60
    height = max(height, min_height)
    
    d = Drawing(width, height)
    
    # Chart area dimensions
    label_width = 200  # More space for portfolio names on left
    chart_x = label_width + 10
    chart_y = 35
    chart_width = width - label_width - 50
    chart_height = height - 70
    
    # Calculate bar dimensions
    bar_height = min(18, (chart_height - 10) / num_bars - 4)
    bar_spacing = (chart_height - (bar_height * num_bars)) / (num_bars + 1)
    
    # Convert to percentages
    max_val = max([v * 100 for v in data.values]) if len(data) > 0 else 10
    max_val = max(max_val, 1)  # Prevent division by zero
    
    # Draw grid lines
    for i in range(6):
        grid_x = chart_x + (i / 5) * chart_width
        d.add(Line(grid_x, chart_y, grid_x, chart_y + chart_height,
                   strokeColor=SYFE_COLORS['medium_grey'], strokeWidth=0.5))
        # Grid labels
        d.add(String(grid_x, chart_y - 12, f"{int(max_val * 1.2 * i / 5)}%",
                     fontName='Helvetica', fontSize=7,
                     fillColor=SYFE_COLORS['text_grey'],
                     textAnchor='middle'))
    
    # Draw bars and labels
    for i, (label, value) in enumerate(data.items()):
        bar_y = chart_y + bar_spacing + i * (bar_height + bar_spacing)
        bar_width = (value * 100 / (max_val * 1.2)) * chart_width if max_val > 0 else 0
        
        # Draw colored bar
        d.add(Rect(chart_x, bar_y, max(bar_width, 2), bar_height,
                   fillColor=EXTENDED_COLORS[i % len(EXTENDED_COLORS)],
                   strokeColor=None))
        
        # Portfolio name on the left (show full name up to 32 chars)
        label_text = str(label)[:32]
        d.add(String(chart_x - 8, bar_y + bar_height / 2 - 2, label_text,
                     fontName='Helvetica', fontSize=6,
                     fillColor=SYFE_COLORS['text_dark'],
                     textAnchor='end'))
        
        # Value label at end of bar
        d.add(String(chart_x + bar_width + 5, bar_y + bar_height / 2 - 2, f"{value*100:.2f}%",
                     fontName='Helvetica', fontSize=7,
                     fillColor=SYFE_COLORS['text_dark'],
                     textAnchor='start'))
    
    # Add title at bottom center
    if title:
        d.add(String(width // 2, 8, title,
                     fontName='Helvetica-Bold', fontSize=10,
                     fillColor=SYFE_COLORS['text_dark'],
                     textAnchor='middle'))
    
    return d


# Extended color palette for more bars (defined before functions that use it)
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
    Creates a horizontal bar chart for dividends (shows dollar amounts).
    
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
    
    # Sort descending and show top 10 max
    data = data.sort_values(ascending=False).head(10)
    num_portfolios = len(data)
    
    # Reverse for horizontal chart (top item at top)
    data = data.iloc[::-1]
    
    num_bars = len(data)
    
    # Keep fixed height to align with sector exposure chart
    # Don't adjust height dynamically - use the same height as donut charts
    d = Drawing(width, height)
    
    # Chart area dimensions - aligned with donut chart legend
    # Donut chart legend starts at height - 25 = 195, so bars should start there too
    label_width = 160
    chart_x = label_width + 10
    chart_y = 45  # More space for title at bottom
    chart_width = width - label_width - 60
    chart_height = height - 70  # Top aligned with donut legend (starts at ~195)
    
    # Calculate bar dimensions
    bar_height = min(15, (chart_height - 10) / num_bars - 2)
    bar_spacing = (chart_height - (bar_height * num_bars)) / (num_bars + 1)
    
    # Find max value for scaling
    max_val = max(data.values) if len(data) > 0 else 1
    max_val = max(max_val, 1)  # Prevent division by zero
    
    # No x-axis grid labels to keep it clean and avoid crowding the title
    # Just draw subtle grid lines
    for i in range(5):
        grid_x = chart_x + (i / 4) * chart_width
        d.add(Line(grid_x, chart_y, grid_x, chart_y + chart_height,
                   strokeColor=SYFE_COLORS['medium_grey'], strokeWidth=0.5))
    
    # Draw bars and labels
    for i, (label, value) in enumerate(data.items()):
        bar_y = chart_y + bar_spacing + i * (bar_height + bar_spacing)
        bar_width = (value / (max_val * 1.2)) * chart_width if max_val > 0 else 0
        
        # Draw colored bar
        d.add(Rect(chart_x, bar_y, max(bar_width, 2), bar_height,
                   fillColor=EXTENDED_COLORS[i % len(EXTENDED_COLORS)],
                   strokeColor=None))
        
        # Portfolio name on the left - larger font size
        label_text = str(label)[:26]
        d.add(String(chart_x - 5, bar_y + bar_height / 2 - 2, label_text,
                     fontName='Helvetica', fontSize=7,
                     fillColor=SYFE_COLORS['text_dark'],
                     textAnchor='end'))
        
        # Value label at end of bar (dollar amount)
        d.add(String(chart_x + bar_width + 3, bar_y + bar_height / 2 - 2, f"${value:,.0f}",
                     fontName='Helvetica', fontSize=7,
                     fillColor=SYFE_COLORS['text_dark'],
                     textAnchor='start'))
    
    # Title: "Portfolio Dividends"
    if not title:
        title = "Portfolio Dividends"
    
    # Add title at bottom center - same y position and font size as Sector Exposure
    d.add(String(width // 2, 10, title,
                 fontName='Helvetica-Bold', fontSize=13,
                 fillColor=SYFE_COLORS['text_dark'],
                 textAnchor='middle'))
    
    return d


# Legacy alias for backwards compatibility
extended_colors = EXTENDED_COLORS + [
    colors.HexColor("#3d5a80"),
    colors.HexColor("#5c7a99"),
    colors.HexColor("#8fa3b8"),
    colors.HexColor("#2c3e50"),
    colors.HexColor("#34495e"),
    colors.HexColor("#5d6d7e"),
    colors.HexColor("#85929e"),
]

# ==========================================
# PORTFOLIO SUMMARY TABLE (grouped format)
# ==========================================

def create_grouped_portfolio_table(df_raw, page_width):
    """
    Creates a grouped portfolio summary table matching the screenshot format.
    
    Format:
    - Portfolio ID, Portfolio Type, Classification, Risk rating, Activated date, Value, Allocation, Returns, Returns Q1
    - Grouped by Growth Portfolios and Income & Preservation Portfolios
    - Subtotals per group
    - Total row at bottom
    """
    if df_raw is None or df_raw.empty:
        return None
    
    df = df_raw.copy()
    
    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Helper to safely get float
    def safe_float(val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace(',', ''))
        except:
            return 0.0
    
    # Convert numeric columns
    if 'nav_in_sgd' in df.columns:
        df['nav_in_sgd'] = df['nav_in_sgd'].apply(safe_float)
    
    if 'pnl_inception' in df.columns:
        df['pnl_inception'] = df['pnl_inception'].apply(safe_float)
    
    # Calculate allocation
    total_nav = df['nav_in_sgd'].sum() if 'nav_in_sgd' in df.columns else 0
    
    # Classification mapping
    def get_classification(portfolio_type):
        ptype = str(portfolio_type).upper()
        if any(x in ptype for x in ['EQUITY', 'GROWTH', 'BALANCED', 'DEFENSIVE', 'TECHNOLOGY', 'HEALTHCARE', 'ESG', 'CHINA']):
            return 'Core'
        if any(x in ptype for x in ['INCOME', 'ENHANCE', 'BLACKROCK', 'MONROE']):
            return 'Income+'
        if 'REIT' in ptype:
            return 'REITs'
        if any(x in ptype for x in ['CASH', 'MMF', 'FLEXI']):
            return 'Cash / MMF'
        return 'Core'
    
    # Risk rating mapping
    def get_risk_rating(portfolio_type):
        ptype = str(portfolio_type).upper()
        if any(x in ptype for x in ['EQUITY100', 'GLOBAL_EQUITY', 'CHINA', 'TECHNOLOGY', 'HEALTHCARE', 'ESG']):
            return 4.0
        if 'GROWTH' in ptype:
            return 3.5
        if 'BALANCED' in ptype:
            return 3.0
        if 'DEFENSIVE' in ptype:
            return 2.5
        if any(x in ptype for x in ['INCOME', 'REIT']):
            return 2.0
        if any(x in ptype for x in ['CASH', 'MMF', 'FLEXI', 'PRESERVE']):
            return 0.5
        return 2.0
    
    # Categorize portfolios
    def is_growth_portfolio(ptype):
        upper = str(ptype).upper()
        return any(x in upper for x in ['EQUITY', 'GROWTH', 'BALANCED', 'DEFENSIVE', 
                                        'TECHNOLOGY', 'HEALTHCARE', 'ESG', 'CHINA', 'GLOBAL'])
    
    # Build table data (removed Returns Q1 column)
    header = ['Portfolio ID', 'Portfolio Type', 'Classification', 'Risk rating (0-5)', 
              'Activated date', 'Value (SGD)', 'Allocation', 'P&L Inception']
    
    table_data = [header]
    
    growth_rows = []
    income_rows = []
    
    for _, row in df.iterrows():
        port_id = str(row.get('portfolio_id', ''))[:12]
        port_type = row.get('type', row.get('name', ''))
        
        # Clean up portfolio type name
        port_type_display = str(port_type).replace('_', ' ').title()
        
        classification = get_classification(port_type)
        risk = get_risk_rating(port_type)
        
        # Get activated date
        act_date = row.get('activated_date_sgt', row.get('activated_date', ''))
        if pd.notna(act_date):
            try:
                if isinstance(act_date, str):
                    act_date = pd.to_datetime(act_date).strftime('%B %d, %Y')
                else:
                    act_date = act_date.strftime('%B %d, %Y')
            except:
                act_date = str(act_date)[:15]
        else:
            act_date = ''
        
        nav = safe_float(row.get('nav_in_sgd', 0))
        allocation = (nav / total_nav * 100) if total_nav > 0 else 0
        
        # P&L Inception - the actual dollar amount from CSV
        pnl = safe_float(row.get('pnl_inception', 0))
        
        row_data = [
            port_id,
            port_type_display,
            classification,
            f"{risk:.1f}",
            act_date,
            f"{nav:,.2f}",
            f"{allocation:.2f}%",
            f"{pnl:,.2f}"  # Show actual P&L in dollars
        ]
        
        if is_growth_portfolio(port_type):
            growth_rows.append((row_data, nav, allocation, pnl))
        else:
            income_rows.append((row_data, nav, allocation, pnl))
    
    # Sort by value
    growth_rows.sort(key=lambda x: -x[1])
    income_rows.sort(key=lambda x: -x[1])
    
    # Calculate subtotals
    growth_total_nav = sum(r[1] for r in growth_rows)
    growth_total_alloc = sum(r[2] for r in growth_rows)
    growth_avg_risk = 4.0
    growth_total_pnl = sum(r[3] for r in growth_rows)
    
    income_total_nav = sum(r[1] for r in income_rows)
    income_total_alloc = sum(r[2] for r in income_rows)
    income_avg_risk = 1.4
    income_total_pnl = sum(r[3] for r in income_rows)
    
    # Add growth portfolios
    for row_data, _, _, _ in growth_rows:
        table_data.append(row_data)
    
    # Add growth subtotal row
    if growth_rows:
        table_data.append([
            'Growth Portfolios', '', '', f"{growth_avg_risk:.1f}", '',
            f"{growth_total_nav:,.2f}", f"{growth_total_alloc:.2f}%",
            f"{growth_total_pnl:,.2f}"
        ])
    
    # Add income portfolios
    for row_data, _, _, _ in income_rows:
        table_data.append(row_data)
    
    # Add income subtotal row
    if income_rows:
        table_data.append([
            'Income & Preservation', '', '', f"{income_avg_risk:.1f}", '',
            f"{income_total_nav:,.2f}", f"{income_total_alloc:.2f}%",
            f"{income_total_pnl:,.2f}"
        ])
    
    # Add total row
    grand_total = growth_total_nav + income_total_nav
    grand_total_pnl = growth_total_pnl + income_total_pnl
    avg_risk = (growth_avg_risk * growth_total_nav + income_avg_risk * income_total_nav) / grand_total if grand_total > 0 else 0
    
    table_data.append([
        'Total', '', '', f"{avg_risk:.1f}", '',
        f"{grand_total:,.2f}", '100.00%',
        f"{grand_total_pnl:,.2f}"
    ])
    
    # Calculate column widths (8 columns now)
    col_widths = [
        0.12 * page_width,  # Portfolio ID
        0.16 * page_width,  # Portfolio Type
        0.11 * page_width,  # Classification
        0.11 * page_width,  # Risk rating
        0.15 * page_width,  # Activated date
        0.12 * page_width,  # Value
        0.11 * page_width,  # Allocation
        0.12 * page_width,  # P&L Inception
    ]
    
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Build style commands
    style_cmds = [
        # Header styling
        ('BACKGROUND', (0, 0), (-1, 0), SYFE_COLORS['dark_navy']),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 7),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        
        # Cell padding
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
        
        # Data rows styling
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), SYFE_COLORS['text_dark']),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('ALIGN', (0, 1), (1, -1), 'LEFT'),
        ('ALIGN', (2, 1), (-1, -1), 'CENTER'),
        
        # Right align numeric columns
        ('ALIGN', (5, 1), (8, -1), 'RIGHT'),
        
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.5, SYFE_COLORS['medium_grey']),
    ]
    
    # Find and style subtotal/total rows
    for i, row in enumerate(table_data[1:], start=1):
        first_cell = str(row[0])
        if 'Growth Portfolios' in first_cell or 'Income & Preservation' in first_cell:
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
    Overlay content PDF pages onto template background.
    """
    if not os.path.exists(template_pdf_path):
        # No template, just copy content
        import shutil
        shutil.copy(content_pdf_path, output_path)
        return
    
    template_reader = PdfReader(template_pdf_path)
    content_reader = PdfReader(content_pdf_path)
    writer = PdfWriter()
    
    for content_page in content_reader.pages:
        # Re-read template for each page to avoid mutation issues
        template_reader_fresh = PdfReader(template_pdf_path)
        template_page = template_reader_fresh.pages[0]
        # Merge content on top of template
        template_page.merge_page(content_page)
        writer.add_page(template_page)
    
    with open(output_path, "wb") as f:
        writer.write(f)

def merge_pdfs(pdf_list, output_path):
    """Merge multiple PDFs into one."""
    writer = PdfWriter()
    
    for pdf_path in pdf_list:
        if pdf_path and os.path.exists(pdf_path):
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                writer.add_page(page)
    
    with open(output_path, "wb") as f:
        writer.write(f)

def add_page_numbers(input_pdf, output_pdf, page_size=landscape(A4), skip_first=True, skip_last=True, client_id=None):
    """
    Adds 'Page X of Y' to the bottom right, date/time and Client ID to top right of each page.
    """
    page_width, page_height = page_size
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    total_pages = len(reader.pages)
    
    # Calculate how many content pages (excluding cover and appendix)
    content_pages = total_pages
    if skip_first:
        content_pages -= 1
    if skip_last:
        content_pages -= 1
    
    # Track which content page we're on
    content_page_num = 0
    
    # Get current date/time for generation timestamp
    generated_datetime = datetime.now().strftime("%d %b %Y, %H:%M")
    
    for page_idx, page in enumerate(reader.pages):
        # Skip cover page (first) and appendix (last)
        if (skip_first and page_idx == 0) or (skip_last and page_idx == total_pages - 1):
            writer.add_page(page)
            continue
        
        # Increment content page counter
        content_page_num += 1
        page_num = content_page_num
        
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=page_size)
        can.setFont("Helvetica", 7)
        can.setFillColor(SYFE_COLORS['text_grey'])
        
        # Page number at bottom right
        text = f"Page {page_num} of {content_pages}"
        text_width = can.stringWidth(text, "Helvetica", 7)
        can.drawString(page_width - 50 - text_width, 15, text)
        
        # Date/time at top right
        can.setFont("Helvetica", 6)
        gen_text = generated_datetime
        gen_text_width = can.stringWidth(gen_text, "Helvetica", 6)
        can.drawString(page_width - 50 - gen_text_width, page_height - 25, gen_text)
        
        # Client ID below date/time
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
    Calculate weighted exposure using matrix multiplication.
    """
    if isinstance(client_weights, dict):
        weights = pd.Series(client_weights)
    else:
        weights = client_weights
    
    if weights.empty or df_exposure_matrix.empty:
        return pd.Series()
    
    # Find common tickers
    common_tickers = weights.index.intersection(df_exposure_matrix.columns)
    
    if len(common_tickers) == 0:
        return pd.Series()
    
    # Align and calculate
    weights_aligned = weights[common_tickers]
    matrix_aligned = df_exposure_matrix[common_tickers]
    
    # Ensure numeric
    matrix_aligned = matrix_aligned.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Matrix multiplication
    exposure = matrix_aligned.dot(weights_aligned)
    
    # Normalize
    if exposure.sum() > 0:
        exposure = exposure / exposure.sum()
    
    # Sort descending
    exposure = exposure.sort_values(ascending=False)
    
    return exposure

# ==========================================
# MAIN REPORT GENERATOR CLASS
# ==========================================

class PortfolioReportGenerator:
    """
    Main class for generating portfolio summary PDF reports.
    """
    
    def __init__(self, client_id, report_date=None, data_path='Data/'):
        self.client_id = str(client_id)
        self.report_date = report_date or date.today()
        self.data_path = data_path
        
    def _get_template_path(self, filename):
        return os.path.join(self.data_path, filename)
    
    def _add_client_id_to_cover(self, cover_path, output_path, page_size):
        """Add client ID text to the cover page."""
        page_width, page_height = page_size
        
        # Read the cover PDF
        reader = PdfReader(cover_path)
        writer = PdfWriter()
        
        # Create overlay with client ID
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=page_size)
        
        # Add Client ID text - positioned below where "Portfolio Summary" would be
        # Adjust position based on your cover page layout
        can.setFont("Helvetica", 10)
        can.setFillColor(SYFE_COLORS['text_dark'])
        client_text = f"Client ID: {self.client_id}"
        # Center the text horizontally, position it in lower third of page
        text_width = can.stringWidth(client_text, "Helvetica", 10)
        can.drawString((page_width - text_width) / 2, page_height * 0.35, client_text)
        
        can.save()
        packet.seek(0)
        
        # Merge overlay with cover page
        overlay_pdf = PdfReader(packet)
        cover_page = reader.pages[0]
        cover_page.merge_page(overlay_pdf.pages[0])
        writer.add_page(cover_page)
        
        # Write output
        with open(output_path, "wb") as f:
            writer.write(f)
    
    def generate_report(
        self,
        df_holdings,
        exposure_data=None,
        returns_data=None,
        dividends_data=None,
        output_filename=None,
        include_cover=True,
        include_appendix=True
    ):
        """
        Generate the complete portfolio report PDF.
        
        Args:
            df_holdings: DataFrame with portfolio holdings data
            exposure_data: Dict with 'asset_class', 'country', 'sector' Series
            returns_data: Series with portfolio returns for bar chart
            dividends_data: Series with dividends earned by portfolio
            output_filename: Output PDF filename
            include_cover: Whether to include cover page
            include_appendix: Whether to include appendix page
        
        Returns:
            str: Path to generated PDF file
        """
        if output_filename is None:
            output_filename = f"Client_{self.client_id}_Portfolio_Summary_{self.report_date.strftime('%Y%m%d')}.pdf"
        
        # Page setup
        page_size = landscape(A4)
        page_width = page_size[0] - 0.8 * inch
        
        all_elements = []
        styles = get_syfe_styles()
        
        # ==========================================
        # PAGE 1: PORTFOLIO SUMMARY + EXPOSURE CHARTS
        # ==========================================
        
        all_elements.append(Paragraph("Portfolio Summary", styles['section_header']))
        
        # Add Client ID below Portfolio Summary
        client_id_style = ParagraphStyle(
            name='ClientID',
            fontName='Helvetica',
            fontSize=8,
            textColor=SYFE_COLORS['text_grey'],
            spaceBefore=2,
            spaceAfter=8
        )
        all_elements.append(Paragraph(f"Client ID: {self.client_id}", client_id_style))
        
        # Holdings Table (grouped format)
        if df_holdings is not None and not df_holdings.empty:
            table = create_grouped_portfolio_table(df_holdings, page_width)
            if table:
                all_elements.append(table)
        
        all_elements.append(Spacer(1, 0.25 * inch))
        
        # ==========================================
        # EXPOSURE CHARTS (Asset Class + Country side by side)
        # ==========================================
        
        chart_row = []
        
        # Asset Class Exposure (Donut)
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
        
        # Country Exposure (Donut - same style as sector)
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
        
        # Sector Exposure (Donut)
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
        
        # Dividends Bar Chart (X Portfolios Contributing to Your Dividends)
        if dividends_data is not None and len(dividends_data) > 0:
            div_chart = create_dividends_bar_chart_drawing(
                dividends_data,
                title="",  # Auto-generates "X Portfolios Contributing to Your Dividends"
                width=380,  # Same width as sector exposure
                height=220  # Same height as sector exposure for alignment
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
        
        # ==========================================
        # BUILD PDF
        # ==========================================
        
        content_pdf = "temp_content.pdf"
        doc = SimpleDocTemplate(
            content_pdf,
            pagesize=page_size,
            leftMargin=0.4 * inch,
            rightMargin=0.4 * inch,
            topMargin=1.0 * inch,  # Increased to leave room for Syfe logo watermark
            bottomMargin=0.4 * inch
        )
        
        try:
            doc.build(all_elements)
        except Exception as e:
            print(f"Error building PDF: {e}")
            raise
        
        # Overlay content on template background
        template_path = self._get_template_path("Portfolio_Visualiser_Template_Landscape.pdf")
        templated_pdf = "temp_templated.pdf"
        
        if os.path.exists(template_path):
            overlay_content_on_template(content_pdf, template_path, templated_pdf)
        else:
            templated_pdf = content_pdf
        
        # Merge with cover and appendix
        pdfs_to_merge = []
        
        cover_path = self._get_template_path("Portfolio_Visualiser_Cover.pdf")
        if include_cover and os.path.exists(cover_path):
            # Add client ID to cover page
            cover_with_client = "temp_cover_with_client.pdf"
            self._add_client_id_to_cover(cover_path, cover_with_client, page_size)
            pdfs_to_merge.append(cover_with_client)
        
        pdfs_to_merge.append(templated_pdf)
        
        appendix_path = self._get_template_path("Portfolio_Visualiser_Appendix.pdf")
        if include_appendix and os.path.exists(appendix_path):
            pdfs_to_merge.append(appendix_path)
        
        merged_pdf = "temp_merged.pdf"
        merge_pdfs(pdfs_to_merge, merged_pdf)
        
        # Add page numbers and client ID
        add_page_numbers(
            merged_pdf, 
            output_filename, 
            page_size=page_size,
            skip_first=include_cover and os.path.exists(cover_path),
            skip_last=include_appendix and os.path.exists(appendix_path),
            client_id=self.client_id
        )
        
        # Cleanup
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
    Prepare holdings DataFrame for PDF display.
    Returns the raw dataframe - formatting is done in create_grouped_portfolio_table.
    """
    return df_raw.copy() if df_raw is not None else pd.DataFrame()


def prepare_returns_data(df_raw):
    """
    Prepare returns data for bar chart.
    Uses pnl_inception and nav_in_sgd to calculate returns percentage.
    Returns = pnl_inception / (nav_in_sgd - pnl_inception) * 100
    """
    if df_raw is None or df_raw.empty:
        return pd.Series()
    
    df = df_raw.copy()
    df.columns = df.columns.str.lower().str.strip()
    
    # Helper to safely convert to float
    def safe_float(val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace(',', ''))
        except:
            return 0.0
    
    # Get name column
    name_col = None
    for col in ['name', 'type', 'portfolio_type']:
        if col in df.columns:
            name_col = col
            break
    
    if name_col is None:
        return pd.Series()
    
    # Calculate returns from pnl_inception and nav
    if 'pnl_inception' in df.columns and 'nav_in_sgd' in df.columns:
        returns_dict = {}
        for _, row in df.iterrows():
            name = str(row[name_col]).replace('_', ' ').title()
            pnl = safe_float(row.get('pnl_inception', 0))
            nav = safe_float(row.get('nav_in_sgd', 0))
            
            # Cost basis = NAV - PnL
            cost_basis = nav - pnl
            
            # Returns = PnL / Cost Basis (as decimal, e.g., 0.05 = 5%)
            if cost_basis > 0:
                returns_pct = pnl / cost_basis
            else:
                returns_pct = 0
            
            returns_dict[name] = returns_pct
        
        return pd.Series(returns_dict)
    
    return pd.Series()


def prepare_dividends_data(df_raw):
    """
    Prepare dividends data for bar chart.
    Uses dividends_reinvested column from CSV.
    Aggregates by portfolio type (sums dividends for same type).
    """
    if df_raw is None or df_raw.empty:
        return pd.Series()
    
    df = df_raw.copy()
    # Normalize column names - lowercase, strip whitespace, replace spaces with underscores
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Helper to safely convert to float
    def safe_float(val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace(',', ''))
        except:
            return 0.0
    
    # Get name/type column (prefer 'type' for aggregation)
    name_col = None
    for col in ['type', 'name', 'portfolio_type']:
        if col in df.columns:
            name_col = col
            break
    
    if name_col is None:
        return pd.Series()
    
    # Find dividends column - check for various naming conventions
    div_col = None
    possible_div_cols = [
        'dividend_reinvested',  # This is the actual column name in the CSV
        'dividends_reinvested', 'dividends', 'dividend', 'div_reinvested',
        'reinvested_dividends', 'total_dividends',
        'dividends_earned', 'dividend_earned'
    ]
    for col in possible_div_cols:
        if col in df.columns:
            div_col = col
            break
    
    # Also check for partial matches
    if div_col is None:
        for col in df.columns:
            if 'dividend' in col.lower():
                div_col = col
                break
    
    if div_col is None:
        print(f"[DEBUG] No dividends column found. Available columns: {list(df.columns)}")
        return pd.Series()
    
    print(f"[DEBUG] Found dividends column: {div_col}")
    
    # Convert dividends column to numeric
    df['div_numeric'] = df[div_col].apply(safe_float)
    
    # Aggregate dividends by portfolio type
    df['type_clean'] = df[name_col].apply(lambda x: str(x).replace('_', ' ').title())
    aggregated = df.groupby('type_clean')['div_numeric'].sum()
    
    # Filter out zero or negative values
    aggregated = aggregated[aggregated > 0]
    
    print(f"[DEBUG] Aggregated dividends data: {aggregated.to_dict()}")
    return aggregated


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
    Convenience function to generate a client report from CSV data.
    """
    # Prepare holdings table
    df_holdings = prepare_holdings_dataframe(df_csv)
    
    # Prepare returns data
    returns_data = prepare_returns_data(df_csv)
    
    # Calculate exposures if matrices provided
    exposure_data = {}
    
    df = df_csv.copy()
    df.columns = df.columns.str.lower().str.strip()
    
    # Build weights from CSV
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
            
            # Use portfolio types as asset class exposure for now
            exposure_data['asset_class'] = weights.copy()
    
    # Generate report
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
