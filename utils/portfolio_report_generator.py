"""
Portfolio Report Generator - Creates PDF reports matching Syfe Portfolio Summary format.

This module generates professional PDF reports with:
- Cover page from template
- Portfolio Summary tables with conditional formatting
- Native ReportLab pie charts for Asset Class, Country, and Sector exposure
- Growth Portfolios section
- Income & Preservation Portfolios section  
- Appendix/Disclaimers page from template
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
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Spacer, 
    Paragraph, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, String, Rect
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics import renderPDF

# PyPDF Imports
from pypdf import PdfReader, PdfWriter

# ==========================================
# COLOR DEFINITIONS (Syfe Branding)
# ==========================================

SYFE_COLORS = {
    'primary_blue': colors.HexColor("#0052CC"),
    'dark_blue': colors.HexColor("#263159"),
    'light_blue': colors.HexColor("#879be4"),
    'navy': colors.HexColor("#172B4D"),
    'white': colors.white,
    'light_grey': colors.HexColor("#F4F5F7"),
    'medium_grey': colors.HexColor("#DFE1E6"),
    'dark_grey': colors.HexColor("#5E6C84"),
    'black': colors.black,
    'green': colors.HexColor("#36B37E"),
    'red': colors.HexColor("#FF5630"),
    'yellow': colors.HexColor("#FFAB00"),
    'teal': colors.HexColor("#00B8D9"),
}

# Chart color palette for pie charts (matching Syfe brand)
PIE_COLORS = [
    colors.HexColor("#0052CC"),  # Primary Blue
    colors.HexColor("#36B37E"),  # Green
    colors.HexColor("#FFAB00"),  # Yellow/Gold
    colors.HexColor("#FF5630"),  # Red
    colors.HexColor("#00B8D9"),  # Teal
    colors.HexColor("#6554C0"),  # Purple
    colors.HexColor("#FF8B00"),  # Orange
    colors.HexColor("#2684FF"),  # Light Blue
    colors.HexColor("#4C9AFF"),  # Lighter Blue
    colors.HexColor("#172B4D"),  # Dark Navy
]

# ==========================================
# PARAGRAPH STYLES
# ==========================================

def get_syfe_styles():
    """Returns dictionary of styled ParagraphStyles for Syfe branding."""
    styles = {}
    
    styles['brand_header'] = ParagraphStyle(
        name='BrandHeader',
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=SYFE_COLORS['primary_blue'],
        spaceAfter=6
    )
    
    styles['page_header'] = ParagraphStyle(
        name='PageHeader',
        fontName='Helvetica',
        fontSize=10,
        textColor=SYFE_COLORS['dark_grey'],
        spaceAfter=4
    )
    
    styles['section_header'] = ParagraphStyle(
        name='SectionHeader',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=SYFE_COLORS['navy'],
        spaceBefore=15,
        spaceAfter=8
    )
    
    styles['chart_title'] = ParagraphStyle(
        name='ChartTitle',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=SYFE_COLORS['navy'],
        alignment=TA_CENTER,
        spaceAfter=4
    )
    
    styles['footnote'] = ParagraphStyle(
        name='Footnote',
        fontName='Helvetica-Oblique',
        fontSize=7,
        textColor=SYFE_COLORS['dark_grey'],
        spaceBefore=4,
        spaceAfter=2
    )
    
    styles['footer'] = ParagraphStyle(
        name='Footer',
        fontName='Helvetica',
        fontSize=8,
        textColor=SYFE_COLORS['dark_grey'],
        alignment=TA_CENTER,
    )
    
    return styles

# ==========================================
# NATIVE REPORTLAB PIE CHART
# ==========================================

def create_pie_chart_drawing(data, title="", width=220, height=180):
    """
    Creates a native ReportLab pie chart drawing.
    
    Args:
        data: pd.Series or dict with labels as keys and values as data
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
    data = data / total  # Normalize to percentages
    
    # Limit to top 8 + Others
    if len(data) > 8:
        top_data = data.head(7)
        others = data.iloc[7:].sum()
        if others > 0:
            top_data = pd.concat([top_data, pd.Series({'Others': others})])
        data = top_data
    
    # Create drawing
    d = Drawing(width, height)
    
    # Create pie chart
    pie = Pie()
    pie.x = width // 2 - 50
    pie.y = 25
    pie.width = 100
    pie.height = 100
    pie.data = list(data.values)
    pie.labels = None  # We'll use legend instead
    
    # Set colors
    for i in range(len(data)):
        pie.slices[i].fillColor = PIE_COLORS[i % len(PIE_COLORS)]
        pie.slices[i].strokeColor = colors.white
        pie.slices[i].strokeWidth = 1
    
    # Create legend
    legend = Legend()
    legend.x = width - 90
    legend.y = height - 30
    legend.dx = 8
    legend.dy = 8
    legend.fontName = 'Helvetica'
    legend.fontSize = 7
    legend.boxAnchor = 'nw'
    legend.columnMaximum = 10
    legend.strokeWidth = 0.5
    legend.strokeColor = colors.white
    legend.deltax = 5
    legend.deltay = 3
    legend.autoXPadding = 3
    legend.yGap = 0
    legend.dxTextSpace = 4
    legend.alignment = 'right'
    legend.dividerLines = 1|2|4
    legend.dividerOffsY = 4.5
    legend.subCols.rpad = 15
    
    # Legend items with percentages
    legend_items = []
    for i, (label, value) in enumerate(data.items()):
        color = PIE_COLORS[i % len(PIE_COLORS)]
        pct = value * 100
        legend_items.append((color, f"{label[:15]}{'...' if len(str(label)) > 15 else ''} ({pct:.1f}%)"))
    
    legend.colorNamePairs = legend_items
    
    # Add title
    title_str = String(width // 2, height - 10, title,
                       fontName='Helvetica-Bold', fontSize=9,
                       fillColor=SYFE_COLORS['navy'], textAnchor='middle')
    
    d.add(pie)
    d.add(legend)
    d.add(title_str)
    
    return d

# ==========================================
# TABLE CREATION
# ==========================================

def create_portfolio_table(df, highlight_groups=True, col_widths=None):
    """
    Creates a styled ReportLab Table from a DataFrame.
    
    Args:
        df: pandas DataFrame with portfolio data
        highlight_groups: Whether to highlight aggregation rows
        col_widths: Optional list of column widths
    
    Returns:
        Table object
    """
    if df.empty:
        return None
    
    # Convert DataFrame to list of lists
    table_data = [list(df.columns)]  # Header row
    for _, row in df.iterrows():
        table_data.append([str(v) if pd.notna(v) else '-' for v in row])
    
    # Create table
    table = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    # Build style commands
    style_cmds = [
        # Header styling
        ('BACKGROUND', (0, 0), (-1, 0), SYFE_COLORS['dark_blue']),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        
        # Cell padding
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        
        # Data rows styling
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.5, SYFE_COLORS['medium_grey']),
        
        # Alternating row colors
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, SYFE_COLORS['light_grey']]),
    ]
    
    # Highlight special group rows
    if highlight_groups:
        special_groups = [
            'Growth Portfolios', 'Income & Preservation Portfolios',
            'Core', 'Specialised', 'Income+', 'REITs', 'Cash / MMF', 'Total'
        ]
        
        first_col = df.columns[0]
        for i, row in df.iterrows():
            row_idx = df.index.get_loc(i) + 1  # +1 for header
            first_val = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ''
            
            if first_val in special_groups:
                style_cmds.append(('BACKGROUND', (0, row_idx), (-1, row_idx), SYFE_COLORS['light_blue']))
                style_cmds.append(('FONTNAME', (0, row_idx), (-1, row_idx), 'Helvetica-Bold'))
            
            if first_val == 'Total':
                style_cmds.append(('BACKGROUND', (0, row_idx), (-1, row_idx), SYFE_COLORS['dark_blue']))
                style_cmds.append(('TEXTCOLOR', (0, row_idx), (-1, row_idx), colors.white))
    
    table.setStyle(TableStyle(style_cmds))
    
    return table

# ==========================================
# PAGE BUILDING FUNCTIONS
# ==========================================

def build_page_header(elements, client_name, report_date):
    """Add standard page header with Syfe branding."""
    styles = get_syfe_styles()
    
    elements.append(Paragraph("Syfe", styles['brand_header']))
    elements.append(Spacer(1, 0.05 * inch))
    elements.append(Paragraph(f"Client Name: {client_name}", styles['page_header']))
    elements.append(Paragraph(f"Date: {report_date.strftime('%B %d, %Y')}", styles['page_header']))
    elements.append(Spacer(1, 0.15 * inch))

def build_page_footer(elements):
    """Add standard page footer."""
    styles = get_syfe_styles()
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("[FOR INTERNAL PURPOSES ONLY - V1]", styles['footer']))

def build_charts_row(elements, charts_data, page_width):
    """
    Build a row of pie charts from exposure data.
    
    Args:
        elements: List to append to
        charts_data: Dict with keys like 'asset_class', 'country', 'sector'
        page_width: Available page width
    """
    styles = get_syfe_styles()
    chart_drawings = []
    
    chart_configs = [
        ('asset_class', 'Asset Class Exposure'),
        ('country', 'Country Exposure'),
        ('sector', 'Sector Exposure'),
    ]
    
    for key, title in chart_configs:
        if key in charts_data and charts_data[key] is not None:
            data = charts_data[key]
            if isinstance(data, pd.Series) and not data.empty:
                drawing = create_pie_chart_drawing(data, title=title)
                if drawing:
                    chart_drawings.append(drawing)
    
    if not chart_drawings:
        return
    
    # Arrange charts in a table row
    elements.append(Spacer(1, 0.1 * inch))
    
    # Create a table with charts
    if len(chart_drawings) <= 3:
        col_width = page_width / len(chart_drawings)
        chart_table = Table([chart_drawings], colWidths=[col_width] * len(chart_drawings))
        chart_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(chart_table)
    else:
        # Multiple rows if more than 3 charts
        for i in range(0, len(chart_drawings), 3):
            row_charts = chart_drawings[i:i+3]
            col_width = page_width / len(row_charts)
            chart_table = Table([row_charts], colWidths=[col_width] * len(row_charts))
            chart_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            elements.append(chart_table)
            elements.append(Spacer(1, 0.1 * inch))

# ==========================================
# PAGE CREATION FUNCTIONS
# ==========================================

def create_portfolio_summary_page(df_summary, charts_data, client_name, report_date, page_width):
    """Creates the Portfolio Summary page elements."""
    elements = []
    styles = get_syfe_styles()
    
    # Header
    build_page_header(elements, client_name, report_date)
    
    # Section title
    elements.append(Paragraph("Portfolio Summary", styles['section_header']))
    
    # Table
    if not df_summary.empty:
        table = create_portfolio_table(df_summary, highlight_groups=True)
        if table:
            elements.append(table)
    
    # Charts
    if charts_data:
        build_charts_row(elements, charts_data, page_width)
    
    # Footer
    build_page_footer(elements)
    elements.append(PageBreak())
    
    return elements

def create_growth_portfolios_page(df_growth, charts_data, client_name, report_date, page_width):
    """Creates the Growth Portfolios page elements."""
    elements = []
    styles = get_syfe_styles()
    
    # Header
    build_page_header(elements, client_name, report_date)
    
    # Section title
    elements.append(Paragraph("Growth Portfolios", styles['section_header']))
    
    # Table
    if not df_growth.empty:
        table = create_portfolio_table(df_growth, highlight_groups=True)
        if table:
            elements.append(table)
    
    # Charts
    if charts_data:
        build_charts_row(elements, charts_data, page_width)
    
    # Footer
    build_page_footer(elements)
    elements.append(PageBreak())
    
    return elements

def create_income_preservation_page(df_income, charts_data, client_name, report_date, page_width):
    """Creates the Income & Preservation Portfolios page elements."""
    elements = []
    styles = get_syfe_styles()
    
    # Header
    build_page_header(elements, client_name, report_date)
    
    # Section title
    elements.append(Paragraph("Income & Preservation Portfolios", styles['section_header']))
    
    # Table
    if not df_income.empty:
        table = create_portfolio_table(df_income, highlight_groups=True)
        if table:
            elements.append(table)
        
        # Footnotes
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph(
            "*Est. Yield (%): References latest annualised distribution yield for Income+ portfolios, "
            "estimated dividend yield for REIT+ portfolios, projected returns for Cash+ Flexi portfolios, "
            "and guaranteed rates for Cash+ Guaranteed portfolios.",
            styles['footnote']
        ))
        elements.append(Paragraph(
            "**Annualised Projected Dividend: Estimated dividend yield multiplied by current portfolio value.",
            styles['footnote']
        ))
    
    # Charts
    if charts_data:
        build_charts_row(elements, charts_data, page_width)
    
    # Footer
    build_page_footer(elements)
    elements.append(PageBreak())
    
    return elements

# ==========================================
# PDF UTILITIES
# ==========================================

def add_page_numbers(input_pdf, output_pdf, page_size=landscape(A4), skip_first=True, skip_last=True):
    """
    Adds 'Page X of Y' to the bottom right of each page.
    Skips first page (cover) and last page (appendix) by default.
    """
    page_width, page_height = page_size
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    total_pages = len(reader.pages)
    
    # Calculate content pages (excluding cover and appendix)
    content_pages = total_pages
    if skip_first:
        content_pages -= 1
    if skip_last:
        content_pages -= 1
    
    for page_idx, page in enumerate(reader.pages):
        # Skip numbering for cover (first) and appendix (last)
        if (skip_first and page_idx == 0) or (skip_last and page_idx == total_pages - 1):
            writer.add_page(page)
            continue
        
        # Calculate page number (adjusted for skipped pages)
        page_num = page_idx if skip_first else page_idx + 1
        
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=page_size)
        can.setFont("Helvetica", 8)
        can.setFillColor(SYFE_COLORS['dark_grey'])
        
        text = f"Page {page_num} of {content_pages}"
        text_width = can.stringWidth(text, "Helvetica", 8)
        
        x_pos = page_width - 50 - text_width
        y_pos = 20
        can.drawString(x_pos, y_pos, text)
        can.save()
        
        packet.seek(0)
        number_pdf = PdfReader(packet)
        page.merge_page(number_pdf.pages[0])
        writer.add_page(page)
    
    with open(output_pdf, "wb") as f:
        writer.write(f)

def merge_pdfs(pdf_list, output_path):
    """Merge multiple PDFs into one."""
    writer = PdfWriter()
    
    for pdf_path in pdf_list:
        if os.path.exists(pdf_path):
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                writer.add_page(page)
    
    with open(output_path, "wb") as f:
        writer.write(f)

# ==========================================
# MAIN REPORT GENERATOR CLASS
# ==========================================

class PortfolioReportGenerator:
    """
    Main class for generating portfolio summary PDF reports.
    """
    
    def __init__(self, client_id, report_date=None, data_path='Data/'):
        """
        Initialize the report generator.
        
        Args:
            client_id: Client identifier
            report_date: Report date (defaults to today)
            data_path: Path to data folder with templates
        """
        self.client_id = str(client_id)
        self.report_date = report_date or date.today()
        self.data_path = data_path
        
    def _get_template_path(self, filename):
        """Get full path to template file."""
        return os.path.join(self.data_path, filename)
    
    def generate_report(
        self,
        df_portfolio_summary,
        df_growth_portfolios=None,
        df_income_portfolios=None,
        exposure_data=None,
        growth_exposure_data=None,
        income_exposure_data=None,
        output_filename=None,
        include_cover=True,
        include_appendix=True
    ):
        """
        Generate the complete portfolio report PDF.
        
        Args:
            df_portfolio_summary: Main portfolio summary DataFrame
            df_growth_portfolios: Growth portfolios detail DataFrame
            df_income_portfolios: Income portfolios detail DataFrame
            exposure_data: Dict with 'asset_class', 'country', 'sector' Series for summary page
            growth_exposure_data: Dict with exposure data for growth page
            income_exposure_data: Dict with exposure data for income page
            output_filename: Output PDF filename
            include_cover: Whether to include cover page from template
            include_appendix: Whether to include appendix page from template
        
        Returns:
            str: Path to generated PDF file
        """
        if output_filename is None:
            output_filename = f"Client_{self.client_id}_Portfolio_Summary_{self.report_date.strftime('%Y%m%d')}.pdf"
        
        # Page setup
        page_size = landscape(A4)
        page_width = page_size[0] - 1.0 * inch  # Account for margins
        
        all_elements = []
        
        # Portfolio Summary page
        summary_elements = create_portfolio_summary_page(
            df_portfolio_summary,
            exposure_data,
            self.client_id,
            self.report_date,
            page_width
        )
        all_elements.extend(summary_elements)
        
        # Growth Portfolios page
        if df_growth_portfolios is not None and not df_growth_portfolios.empty:
            growth_elements = create_growth_portfolios_page(
                df_growth_portfolios,
                growth_exposure_data,
                self.client_id,
                self.report_date,
                page_width
            )
            all_elements.extend(growth_elements)
        
        # Income & Preservation Portfolios page
        if df_income_portfolios is not None and not df_income_portfolios.empty:
            income_elements = create_income_preservation_page(
                df_income_portfolios,
                income_exposure_data,
                self.client_id,
                self.report_date,
                page_width
            )
            all_elements.extend(income_elements)
        
        # Build content PDF
        content_pdf = "temp_content.pdf"
        doc = SimpleDocTemplate(
            content_pdf,
            pagesize=page_size,
            leftMargin=0.5 * inch,
            rightMargin=0.5 * inch,
            topMargin=0.4 * inch,
            bottomMargin=0.4 * inch
        )
        
        try:
            doc.build(all_elements)
        except Exception as e:
            print(f"Error building PDF: {e}")
            raise
        
        # Merge with cover and appendix
        pdfs_to_merge = []
        
        # Add cover page
        cover_path = self._get_template_path("Portfolio_Visualiser_Cover.pdf")
        if include_cover and os.path.exists(cover_path):
            pdfs_to_merge.append(cover_path)
        
        # Add content
        pdfs_to_merge.append(content_pdf)
        
        # Add appendix
        appendix_path = self._get_template_path("Portfolio_Visualiser_Appendix.pdf")
        if include_appendix and os.path.exists(appendix_path):
            pdfs_to_merge.append(appendix_path)
        
        # Merge PDFs
        merged_pdf = "temp_merged.pdf"
        merge_pdfs(pdfs_to_merge, merged_pdf)
        
        # Add page numbers (skip cover and appendix)
        add_page_numbers(
            merged_pdf, 
            output_filename, 
            page_size=page_size,
            skip_first=include_cover and os.path.exists(cover_path),
            skip_last=include_appendix and os.path.exists(appendix_path)
        )
        
        # Cleanup temp files
        for f in [content_pdf, merged_pdf]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        
        return output_filename

# ==========================================
# HELPER FUNCTIONS FOR DATA PREPARATION
# ==========================================

def prepare_portfolio_summary_df(portfolios_data):
    """
    Prepares the portfolio summary DataFrame from raw portfolio data.
    
    Args:
        portfolios_data: List of dicts or DataFrame with portfolio information
    
    Returns:
        DataFrame: Formatted for the summary table
    """
    if isinstance(portfolios_data, list):
        df = pd.DataFrame(portfolios_data)
    else:
        df = portfolios_data.copy()
    
    # Column mapping
    col_mapping = {
        'portfolio_id': 'Portfolio ID',
        'portfolio_type': 'Portfolio Type',
        'name': 'Portfolio Type',
        'type': 'Portfolio Type',
        'classification': 'Classification',
        'risk_rating': 'Risk rating (0-5)',
        'activated_date': 'Activated date',
        'value': 'Value (SGD)',
        'nav_in_sgd': 'Value (SGD)',
        'allocation': 'Allocation',
        'returns': 'Portfolio Returns',
        'pnl_inception': 'Portfolio Returns',
        'quarterly_returns': 'Returns (Q1)'
    }
    
    df = df.rename(columns=col_mapping)
    
    # Format numeric columns
    if 'Value (SGD)' in df.columns:
        df['Value (SGD)'] = df['Value (SGD)'].apply(
            lambda x: f"{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x
        )
    
    if 'Allocation' in df.columns:
        df['Allocation'] = df['Allocation'].apply(
            lambda x: f"{x:.2%}" if pd.notna(x) and isinstance(x, (int, float)) and x <= 1 else 
                      (f"{x:.2f}%" if pd.notna(x) and isinstance(x, (int, float)) else x)
        )
    
    if 'Portfolio Returns' in df.columns:
        df['Portfolio Returns'] = df['Portfolio Returns'].apply(
            lambda x: f"{x:.2%}" if pd.notna(x) and isinstance(x, (int, float)) else x
        )
    
    if 'Risk rating (0-5)' in df.columns:
        df['Risk rating (0-5)'] = df['Risk rating (0-5)'].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) and isinstance(x, (int, float)) else x
        )
    
    return df

def prepare_income_portfolio_df(portfolios_data):
    """
    Prepares the income portfolio DataFrame with yield columns.
    """
    if isinstance(portfolios_data, list):
        df = pd.DataFrame(portfolios_data)
    else:
        df = portfolios_data.copy()
    
    col_mapping = {
        'portfolio_id': 'Portfolio ID',
        'portfolio_type': 'Portfolio Type',
        'name': 'Portfolio Type',
        'risk_rating': 'Risk rating (0-5)',
        'value': 'Value (SGD)',
        'nav_in_sgd': 'Value (SGD)',
        'est_yield': 'Est. Yield (%)*',
        'yield': 'Est. Yield (%)*',
        'projected_dividend': 'Ann. Proj. Dividend (SGD)**',
        'dividend_option': 'Dividend Option',
        'maturity': 'Maturity period',
        'allocation': 'Allocation'
    }
    
    df = df.rename(columns=col_mapping)
    
    # Format numeric columns
    if 'Value (SGD)' in df.columns:
        df['Value (SGD)'] = df['Value (SGD)'].apply(
            lambda x: f"{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x
        )
    
    if 'Ann. Proj. Dividend (SGD)**' in df.columns:
        df['Ann. Proj. Dividend (SGD)**'] = df['Ann. Proj. Dividend (SGD)**'].apply(
            lambda x: f"{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x
        )
    
    return df

def generate_client_report(
    client_id,
    portfolio_data,
    exposure_data=None,
    report_date=None,
    data_path='Data/',
    output_path=None
):
    """
    Convenience function to generate a client report.
    """
    generator = PortfolioReportGenerator(
        client_id=client_id,
        report_date=report_date,
        data_path=data_path
    )
    
    df_summary = portfolio_data.get('summary', pd.DataFrame())
    df_growth = portfolio_data.get('growth', pd.DataFrame())
    df_income = portfolio_data.get('income', pd.DataFrame())
    
    summary_exposure = exposure_data.get('summary', {}) if exposure_data else {}
    growth_exposure = exposure_data.get('growth', {}) if exposure_data else {}
    income_exposure = exposure_data.get('income', {}) if exposure_data else {}
    
    return generator.generate_report(
        df_portfolio_summary=df_summary,
        df_growth_portfolios=df_growth,
        df_income_portfolios=df_income,
        exposure_data=summary_exposure,
        growth_exposure_data=growth_exposure,
        income_exposure_data=income_exposure,
        output_filename=output_path
    )


# ==========================================
# EXAMPLE USAGE / MAIN
# ==========================================

if __name__ == "__main__":
    print("Portfolio Report Generator - Example Usage")
    print("=" * 50)
    
    # Sample data matching the PDF structure
    df_summary = pd.DataFrame([
        {'Portfolio ID': '644643869', 'Portfolio Type': 'Core Equity100', 'Classification': 'Core', 
         'Risk rating (0-5)': 4.0, 'Activated date': 'April 9, 2025', 'Value (SGD)': 103466.88, 
         'Allocation': 0.0584, 'Portfolio Returns': 0.0004, 'Returns (Q1)': 0.0125},
        {'Portfolio ID': 'Growth Portfolios', 'Portfolio Type': '', 'Classification': '', 
         'Risk rating (0-5)': 4.0, 'Activated date': '', 'Value (SGD)': 103466.88, 
         'Allocation': 0.0584, 'Portfolio Returns': 0.0004, 'Returns (Q1)': 0.0125},
        {'Portfolio ID': '1821329309', 'Portfolio Type': 'Income Enhance', 'Classification': 'Income+', 
         'Risk rating (0-5)': 2.0, 'Activated date': 'November 1, 2024', 'Value (SGD)': 247778.16, 
         'Allocation': 0.14, 'Portfolio Returns': -0.0001, 'Returns (Q1)': 0.0125},
        {'Portfolio ID': '374443838', 'Portfolio Type': 'REIT 100', 'Classification': 'REITs', 
         'Risk rating (0-5)': 4.0, 'Activated date': 'October 29, 2024', 'Value (SGD)': 280900.48, 
         'Allocation': 0.1587, 'Portfolio Returns': -0.0011, 'Returns (Q1)': 0.0125},
        {'Portfolio ID': '478419426', 'Portfolio Type': 'Cash SGD flexi', 'Classification': 'Cash / MMF', 
         'Risk rating (0-5)': 0.5, 'Activated date': 'September 25, 2024', 'Value (SGD)': 1075349.14, 
         'Allocation': 0.6074, 'Portfolio Returns': 0.0002, 'Returns (Q1)': 0.0125},
        {'Portfolio ID': 'Income & Preservation Portfolios', 'Portfolio Type': '', 'Classification': '', 
         'Risk rating (0-5)': 1.4, 'Activated date': '', 'Value (SGD)': 1666972.97, 
         'Allocation': 0.9416, 'Portfolio Returns': 0.0005, 'Returns (Q1)': 0.0125},
        {'Portfolio ID': 'Total', 'Portfolio Type': '', 'Classification': '', 
         'Risk rating (0-5)': 1.5, 'Activated date': '', 'Value (SGD)': 1770439.85, 
         'Allocation': 1.0, 'Portfolio Returns': 0.0009, 'Returns (Q1)': 0.0125},
    ])
    
    # Sample exposure data
    exposure_data = {
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
        })
    }
    
    # Format the summary dataframe
    df_summary_formatted = prepare_portfolio_summary_df(df_summary)
    
    # Generate report
    generator = PortfolioReportGenerator(
        client_id='585455760',
        report_date=date(2025, 6, 27),
        data_path='Data/'
    )
    
    try:
        output_file = generator.generate_report(
            df_portfolio_summary=df_summary_formatted,
            exposure_data=exposure_data,
            include_cover=True,
            include_appendix=True
        )
        print(f"\nSUCCESS: Report generated at {output_file}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
