"""
Portfolio Report Generator - Creates PDF reports matching Syfe Portfolio Summary format.

This module generates professional PDF reports with:
- Cover page with branding
- Portfolio Summary tables with conditional formatting
- Asset Class, Country, and Sector exposure charts
- Growth Portfolios section
- Income & Preservation Portfolios section  
- Disclaimers page
"""

import os
import io
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.graph_objects as go
import plotly.express as px

# ReportLab Imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Spacer, Image, 
    Paragraph, PageBreak, KeepTogether, Frame, PageTemplate, BaseDocTemplate
)
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
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

# Chart color palette for pie charts
PIE_COLORS = [
    "#0052CC", "#36B37E", "#FFAB00", "#FF5630", "#00B8D9",
    "#6554C0", "#FF8B00", "#2684FF", "#4C9AFF", "#B3D4FF"
]

# ==========================================
# PARAGRAPH STYLES
# ==========================================

def get_syfe_styles():
    """Returns dictionary of styled ParagraphStyles for Syfe branding."""
    styles = {}
    
    styles['title'] = ParagraphStyle(
        name='SyfeTitle',
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        alignment=TA_CENTER,
        textColor=SYFE_COLORS['dark_blue'],
        spaceAfter=20
    )
    
    styles['subtitle'] = ParagraphStyle(
        name='SyfeSubtitle',
        fontName='Helvetica',
        fontSize=14,
        leading=18,
        alignment=TA_CENTER,
        textColor=SYFE_COLORS['dark_grey'],
        spaceAfter=12
    )
    
    styles['section_header'] = ParagraphStyle(
        name='SyfeSectionHeader',
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        alignment=TA_LEFT,
        textColor=SYFE_COLORS['navy'],
        spaceBefore=20,
        spaceAfter=10
    )
    
    styles['body'] = ParagraphStyle(
        name='SyfeBody',
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        textColor=SYFE_COLORS['black'],
    )
    
    styles['small'] = ParagraphStyle(
        name='SyfeSmall',
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        alignment=TA_LEFT,
        textColor=SYFE_COLORS['dark_grey'],
    )
    
    styles['footer'] = ParagraphStyle(
        name='SyfeFooter',
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        textColor=SYFE_COLORS['dark_grey'],
    )
    
    styles['disclaimer'] = ParagraphStyle(
        name='SyfeDisclaimer',
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        alignment=TA_JUSTIFY,
        textColor=SYFE_COLORS['dark_grey'],
        spaceBefore=6,
        spaceAfter=6
    )
    
    return styles

# ==========================================
# TABLE CREATION FUNCTIONS
# ==========================================

def create_portfolio_table(df, title="", highlight_groups=True, col_widths=None):
    """
    Creates a styled ReportLab Table from a DataFrame.
    
    Args:
        df: pandas DataFrame with portfolio data
        title: Optional table title
        highlight_groups: Whether to highlight aggregation rows
        col_widths: Optional list of column widths
    
    Returns:
        list: ReportLab flowable elements (title + table)
    """
    elements = []
    styles = get_syfe_styles()
    
    if df.empty:
        return elements
    
    # Add title if provided
    if title:
        elements.append(Paragraph(title, styles['section_header']))
        elements.append(Spacer(1, 0.1 * inch))
    
    # Convert DataFrame to list of lists for ReportLab
    # Use Paragraph for wrapping text in cells
    header_row = []
    for col in df.columns:
        header_row.append(Paragraph(f"<b>{col}</b>", ParagraphStyle(
            name='HeaderCell',
            fontName='Helvetica-Bold',
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.white
        )))
    
    table_data = [header_row]
    
    for idx, row in df.iterrows():
        row_data = []
        for val in row:
            cell_style = ParagraphStyle(
                name='Cell',
                fontName='Helvetica',
                fontSize=8,
                leading=10,
                alignment=TA_CENTER,
                textColor=colors.black
            )
            row_data.append(Paragraph(str(val) if pd.notna(val) else '-', cell_style))
        table_data.append(row_data)
    
    # Calculate column widths if not provided
    if col_widths is None:
        page_width = landscape(A4)[0] - 1.4 * inch  # Account for margins
        num_cols = len(df.columns)
        col_widths = [page_width / num_cols] * num_cols
    
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
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.5, SYFE_COLORS['medium_grey']),
        
        # Alternating row colors
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, SYFE_COLORS['light_grey']]),
    ]
    
    # Highlight special group rows (e.g., "Growth Portfolios", "Total")
    if highlight_groups and 'Portfolio ID' in df.columns:
        special_groups = [
            'Growth Portfolios', 'Income & Preservation Portfolios',
            'Core', 'Specialised', 'Income+', 'REITs', 'Cash / MMF', 'Total'
        ]
        
        for i, row in df.iterrows():
            row_idx = df.index.get_loc(i) + 1  # +1 for header row
            port_id = str(row.get('Portfolio ID', ''))
            
            if port_id in special_groups:
                style_cmds.append(('BACKGROUND', (0, row_idx), (-1, row_idx), SYFE_COLORS['light_blue']))
                style_cmds.append(('FONTNAME', (0, row_idx), (-1, row_idx), 'Helvetica-Bold'))
            
            if port_id == 'Total':
                style_cmds.append(('BACKGROUND', (0, row_idx), (-1, row_idx), SYFE_COLORS['dark_blue']))
                style_cmds.append(('TEXTCOLOR', (0, row_idx), (-1, row_idx), colors.white))
    
    table.setStyle(TableStyle(style_cmds))
    elements.append(table)
    
    return elements

def create_income_table(df, title=""):
    """
    Creates a specialized table for Income & Preservation portfolios with additional columns.
    """
    return create_portfolio_table(df, title=title, highlight_groups=True)

# ==========================================
# CHART CREATION FUNCTIONS
# ==========================================

def create_pie_chart_image(data, title="", width=350, height=250):
    """
    Creates a pie chart using Plotly and returns as image bytes.
    
    Args:
        data: Series with labels as index and values as data
        title: Chart title
        width, height: Image dimensions
    
    Returns:
        bytes: PNG image data
    """
    if data.empty or data.sum() == 0:
        return None
    
    # Filter out very small values and limit to top 10
    data = data[data > 0.001]
    if len(data) > 10:
        top_data = data.head(9)
        others = data.iloc[9:].sum()
        top_data['Others'] = others
        data = top_data
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=data.index,
        values=data.values,
        hole=0.4,
        marker=dict(colors=PIE_COLORS[:len(data)]),
        textinfo='percent+label',
        textposition='outside',
        textfont=dict(size=8),
        outsidetextfont=dict(size=8),
        insidetextfont=dict(size=7),
    )])
    
    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            font=dict(size=12, color='#172B4D'),
            x=0.5,
            y=0.95
        ),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='right',
            x=1.3,
            font=dict(size=8)
        ),
        margin=dict(l=20, r=100, t=40, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=width,
        height=height
    )
    
    # Export to bytes
    img_bytes = fig.to_image(format='png', scale=2)
    return img_bytes

def create_bar_chart_image(data, title="", width=350, height=250):
    """
    Creates a horizontal bar chart using Plotly and returns as image bytes.
    """
    if data.empty or data.sum() == 0:
        return None
    
    # Limit to top 10
    data = data.head(10)
    
    fig = go.Figure(data=[go.Bar(
        x=data.values,
        y=data.index,
        orientation='h',
        marker_color=PIE_COLORS[0],
        text=[f'{v:.1%}' for v in data.values],
        textposition='outside',
        textfont=dict(size=8)
    )])
    
    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b>',
            font=dict(size=12, color='#172B4D'),
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            tickformat='.0%',
            range=[0, max(data.values) * 1.2],
            tickfont=dict(size=8)
        ),
        yaxis=dict(
            tickfont=dict(size=8),
            autorange='reversed'
        ),
        margin=dict(l=100, r=60, t=40, b=30),
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=width,
        height=height
    )
    
    img_bytes = fig.to_image(format='png', scale=2)
    return img_bytes

def add_charts_to_elements(elements, charts_data, available_width):
    """
    Adds exposure charts (pie charts) to the elements list.
    
    Args:
        elements: List of flowable elements
        charts_data: Dict with 'asset_class', 'country', 'sector' keys containing Series data
        available_width: Available page width for charts
    """
    temp_files = []
    chart_images = []
    
    chart_config = [
        ('asset_class', 'Asset Class Exposure'),
        ('country', 'Country Exposure'),
        ('sector', 'Sector Exposure')
    ]
    
    chart_width = 3.5 * inch
    chart_height = 2.5 * inch
    
    for key, title in chart_config:
        if key in charts_data and charts_data[key] is not None:
            data = charts_data[key]
            if isinstance(data, pd.Series) and not data.empty:
                img_bytes = create_pie_chart_image(data, title=title, width=350, height=250)
                if img_bytes:
                    # Save to temp file
                    temp_file = f"temp_chart_{key}_{datetime.now().strftime('%H%M%S%f')}.png"
                    with open(temp_file, 'wb') as f:
                        f.write(img_bytes)
                    temp_files.append(temp_file)
                    
                    img = Image(temp_file, width=chart_width, height=chart_height)
                    chart_images.append(img)
    
    # Arrange charts in rows of 2
    if chart_images:
        rows = []
        for i in range(0, len(chart_images), 2):
            row = chart_images[i:i+2]
            if len(row) == 1:
                row.append('')  # Empty cell for alignment
            rows.append(row)
        
        if rows:
            col_widths = [available_width / 2] * 2
            chart_table = Table(rows, colWidths=col_widths)
            chart_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(chart_table)
    
    return temp_files

# ==========================================
# PAGE CREATION FUNCTIONS
# ==========================================

def create_cover_page(client_id, report_date):
    """
    Creates cover page elements.
    
    Args:
        client_id: Client identifier
        report_date: Report generation date
    
    Returns:
        list: Flowable elements for cover page
    """
    elements = []
    styles = get_syfe_styles()
    
    # Add vertical spacing to center content
    elements.append(Spacer(1, 2.5 * inch))
    
    # Syfe Logo placeholder (text for now)
    logo_style = ParagraphStyle(
        name='Logo',
        fontName='Helvetica-Bold',
        fontSize=48,
        alignment=TA_CENTER,
        textColor=SYFE_COLORS['primary_blue'],
        spaceAfter=30
    )
    elements.append(Paragraph("Syfe", logo_style))
    
    # Title
    elements.append(Paragraph("PORTFOLIO SUMMARY", styles['title']))
    
    elements.append(Spacer(1, 0.5 * inch))
    
    # Client info
    client_style = ParagraphStyle(
        name='ClientInfo',
        fontName='Helvetica',
        fontSize=14,
        alignment=TA_CENTER,
        textColor=SYFE_COLORS['dark_grey'],
        spaceAfter=10
    )
    elements.append(Paragraph(f"Client {client_id}", client_style))
    elements.append(Paragraph(report_date.strftime("%B %d, %Y"), client_style))
    
    elements.append(PageBreak())
    
    return elements

def create_portfolio_summary_page(df_summary, charts_data, client_name, report_date):
    """
    Creates the main Portfolio Summary page.
    
    Args:
        df_summary: DataFrame with portfolio summary data
        charts_data: Dict with exposure data for charts
        client_name: Client name/ID
        report_date: Report date
    
    Returns:
        tuple: (elements list, temp_files list)
    """
    elements = []
    styles = get_syfe_styles()
    temp_files = []
    
    # Page header
    header_style = ParagraphStyle(
        name='PageHeader',
        fontName='Helvetica-Bold',
        fontSize=10,
        alignment=TA_LEFT,
        textColor=SYFE_COLORS['dark_grey'],
    )
    
    # Client name and date header
    elements.append(Paragraph(f"<b>Syfe</b>", ParagraphStyle(
        name='BrandHeader',
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=SYFE_COLORS['primary_blue']
    )))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph(f"Client Name: {client_name}", header_style))
    elements.append(Paragraph(f"Date: {report_date.strftime('%B %d, %Y')}", header_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Section title
    elements.append(Paragraph("Portfolio Summary", styles['section_header']))
    
    # Portfolio table
    page_width = landscape(A4)[0] - 1.4 * inch
    table_elements = create_portfolio_table(df_summary, highlight_groups=True)
    elements.extend(table_elements)
    
    # Add charts
    if charts_data:
        temp_files = add_charts_to_elements(elements, charts_data, page_width)
    
    # Footer
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("[FOR INTERNAL PURPOSES ONLY - V1]", styles['footer']))
    
    elements.append(PageBreak())
    
    return elements, temp_files

def create_growth_portfolios_page(df_growth, charts_data, client_name, report_date):
    """
    Creates the Growth Portfolios detail page.
    """
    elements = []
    styles = get_syfe_styles()
    temp_files = []
    
    # Header
    elements.append(Paragraph(f"<b>Syfe</b>", ParagraphStyle(
        name='BrandHeader',
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=SYFE_COLORS['primary_blue']
    )))
    elements.append(Spacer(1, 0.1 * inch))
    
    header_style = ParagraphStyle(
        name='PageHeader',
        fontName='Helvetica-Bold',
        fontSize=10,
        alignment=TA_LEFT,
        textColor=SYFE_COLORS['dark_grey'],
    )
    elements.append(Paragraph(f"Client Name: {client_name}", header_style))
    elements.append(Paragraph(f"Date: {report_date.strftime('%B %d, %Y')}", header_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Section title
    elements.append(Paragraph("Growth Portfolios", styles['section_header']))
    
    # Table
    page_width = landscape(A4)[0] - 1.4 * inch
    if not df_growth.empty:
        table_elements = create_portfolio_table(df_growth, highlight_groups=True)
        elements.extend(table_elements)
    
    # Charts
    if charts_data:
        temp_files = add_charts_to_elements(elements, charts_data, page_width)
    
    # Footer
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("[FOR INTERNAL PURPOSES ONLY - V1]", styles['footer']))
    
    elements.append(PageBreak())
    
    return elements, temp_files

def create_income_preservation_page(df_income, charts_data, client_name, report_date):
    """
    Creates the Income & Preservation Portfolios detail page.
    """
    elements = []
    styles = get_syfe_styles()
    temp_files = []
    
    # Header
    elements.append(Paragraph(f"<b>Syfe</b>", ParagraphStyle(
        name='BrandHeader',
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=SYFE_COLORS['primary_blue']
    )))
    elements.append(Spacer(1, 0.1 * inch))
    
    header_style = ParagraphStyle(
        name='PageHeader',
        fontName='Helvetica-Bold',
        fontSize=10,
        alignment=TA_LEFT,
        textColor=SYFE_COLORS['dark_grey'],
    )
    elements.append(Paragraph(f"Client Name: {client_name}", header_style))
    elements.append(Paragraph(f"Date: {report_date.strftime('%B %d, %Y')}", header_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Section title
    elements.append(Paragraph("Income & Preservation Portfolios", styles['section_header']))
    
    # Footnotes for yields
    footnote_style = ParagraphStyle(
        name='Footnote',
        fontName='Helvetica-Oblique',
        fontSize=7,
        textColor=SYFE_COLORS['dark_grey'],
        spaceBefore=4
    )
    
    # Table
    page_width = landscape(A4)[0] - 1.4 * inch
    if not df_income.empty:
        table_elements = create_portfolio_table(df_income, highlight_groups=True)
        elements.extend(table_elements)
        
        # Add footnotes
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph(
            "*Est. Yield (%): References latest annualised distribution yield for Income+ portfolios, "
            "estimated dividend yield for REIT+ portfolios, projected returns for Cash+ Flexi portfolios, "
            "and guaranteed rates for Cash+ Guaranteed portfolios.",
            footnote_style
        ))
        elements.append(Paragraph(
            "**Annualised Projected Dividend: Estimated dividend yield multiplied by current portfolio value.",
            footnote_style
        ))
    
    # Charts
    if charts_data:
        temp_files = add_charts_to_elements(elements, charts_data, page_width)
    
    # Footer
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("[FOR INTERNAL PURPOSES ONLY - V1]", styles['footer']))
    
    elements.append(PageBreak())
    
    return elements, temp_files

def create_disclaimers_page(client_name, report_date):
    """
    Creates the disclaimers page.
    """
    elements = []
    styles = get_syfe_styles()
    
    # Header
    elements.append(Paragraph(f"<b>Syfe</b>", ParagraphStyle(
        name='BrandHeader',
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=SYFE_COLORS['primary_blue']
    )))
    elements.append(Spacer(1, 0.1 * inch))
    
    header_style = ParagraphStyle(
        name='PageHeader',
        fontName='Helvetica-Bold',
        fontSize=10,
        alignment=TA_LEFT,
        textColor=SYFE_COLORS['dark_grey'],
    )
    elements.append(Paragraph(f"Client Name: {client_name}", header_style))
    elements.append(Paragraph(f"Date: {report_date.strftime('%B %d, %Y')}", header_style))
    elements.append(Spacer(1, 0.3 * inch))
    
    # Disclaimers title
    elements.append(Paragraph("Disclaimers", styles['section_header']))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Disclaimer text
    disclaimers = [
        "Please review this statement carefully. If you disagree with any transactions, or if there are any errors or omissions, please notify our Investment Advisory Team at advisory@syfe.com as soon as possible. The statement will otherwise be considered conclusive.",
        
        "All figures are rounded to 2 decimal numbers. Amounts and units less than 0.01 are shown as <0.01.",
        
        "For Cash+ Guaranteed portfolios, the closing value shown in this statement includes the returns to-date and is indicative only. All Cash+ Guaranteed returns will only be available after maturity.",
        
        "Any values, forecast, projection or target is indicative only and is not guaranteed in any way. Past performances are not indicative of future performances. All forms of investments carry risks. The possibility exists that you could lose some or all your investments or deposits. If you have any doubt on this material or any offering documentation, you should seek independent professional advice.",
        
        "This material does not constitute an offer or invitation to anyone in any jurisdiction to invest in any of Syfe's product or use any Syfe's services where such offer or invitation is not lawful, or in which the person making such offer or invitation is not qualified to do so, nor has it been prepared in connection with any such offer or invitation.",
        
        "Syfe Pte. Ltd. is registered with the Monetary Authority of Singapore (\"MAS\") as a Capital Markets Services (\"CMS\") license holder to carry out retail fund management activities issued under the Securities and Futures Act (Singapore Statute Cap. 289)."
    ]
    
    for disclaimer in disclaimers:
        elements.append(Paragraph(disclaimer, styles['disclaimer']))
    
    elements.append(Spacer(1, 0.5 * inch))
    
    # Company info
    company_info_style = ParagraphStyle(
        name='CompanyInfo',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=SYFE_COLORS['navy'],
        spaceBefore=4,
        spaceAfter=2
    )
    
    contact_style = ParagraphStyle(
        name='Contact',
        fontName='Helvetica',
        fontSize=9,
        textColor=SYFE_COLORS['dark_grey'],
        spaceBefore=2,
        spaceAfter=2
    )
    
    elements.append(Paragraph("Syfe Pte. Ltd", company_info_style))
    elements.append(Paragraph("4 Robinson Rd", contact_style))
    elements.append(Paragraph("#11-01", contact_style))
    elements.append(Paragraph("The House Of Eden", contact_style))
    elements.append(Paragraph("Singapore 048543", contact_style))
    elements.append(Spacer(1, 0.1 * inch))
    elements.append(Paragraph("Email: advisory@syfe.com", contact_style))
    elements.append(Paragraph("Phone: +65 6911 6863", contact_style))
    elements.append(Paragraph("MAS License: CMS100837", contact_style))
    
    # Footer
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("[FOR INTERNAL PURPOSES ONLY - V1]", styles['footer']))
    
    return elements

# ==========================================
# PAGE NUMBERING
# ==========================================

def add_page_numbers(input_pdf, output_pdf, page_size=landscape(A4)):
    """
    Adds 'Page X of Y' to the bottom right of each page.
    """
    page_width, page_height = page_size
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    total_pages = len(reader.pages)
    
    for page_num, page in enumerate(reader.pages, start=1):
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=page_size)
        can.setFont("Helvetica", 8)
        can.setFillColor(SYFE_COLORS['dark_grey'])
        
        text = f"Page {page_num} of {total_pages}"
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

def merge_with_template(template_pdf_path, overlay_pdf_path, output_pdf_path):
    """
    Merges content (overlay) onto a background template.
    """
    if not os.path.exists(template_pdf_path):
        # If template doesn't exist, just copy the overlay
        import shutil
        shutil.copy(overlay_pdf_path, output_pdf_path)
        return
    
    template_pdf = PdfReader(template_pdf_path)
    overlay_pdf = PdfReader(overlay_pdf_path)
    output_pdf = PdfWriter()
    
    template_page = template_pdf.pages[0]
    
    for page in overlay_pdf.pages:
        new_page = template_page.create_blank_page(
            width=template_page.mediabox.width,
            height=template_page.mediabox.height
        )
        new_page.merge_page(template_page)
        new_page.merge_page(page)
        output_pdf.add_page(new_page)
    
    with open(output_pdf_path, "wb") as f:
        output_pdf.write(f)

# ==========================================
# MAIN REPORT GENERATOR
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
        self.temp_files = []
        
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
        include_disclaimers=True
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
            include_cover: Whether to include cover page
            include_disclaimers: Whether to include disclaimers page
        
        Returns:
            str: Path to generated PDF file
        """
        if output_filename is None:
            output_filename = f"Client_{self.client_id}_Portfolio_Summary_{self.report_date.strftime('%Y%m%d')}.pdf"
        
        all_elements = []
        all_temp_files = []
        
        # Cover page
        if include_cover:
            cover_elements = create_cover_page(self.client_id, self.report_date)
            all_elements.extend(cover_elements)
        
        # Portfolio Summary page
        summary_elements, temp_files = create_portfolio_summary_page(
            df_portfolio_summary,
            exposure_data,
            self.client_id,
            self.report_date
        )
        all_elements.extend(summary_elements)
        all_temp_files.extend(temp_files)
        
        # Growth Portfolios page
        if df_growth_portfolios is not None and not df_growth_portfolios.empty:
            growth_elements, temp_files = create_growth_portfolios_page(
                df_growth_portfolios,
                growth_exposure_data,
                self.client_id,
                self.report_date
            )
            all_elements.extend(growth_elements)
            all_temp_files.extend(temp_files)
        
        # Income & Preservation Portfolios page
        if df_income_portfolios is not None and not df_income_portfolios.empty:
            income_elements, temp_files = create_income_preservation_page(
                df_income_portfolios,
                income_exposure_data,
                self.client_id,
                self.report_date
            )
            all_elements.extend(income_elements)
            all_temp_files.extend(temp_files)
        
        # Disclaimers page
        if include_disclaimers:
            disclaimer_elements = create_disclaimers_page(self.client_id, self.report_date)
            all_elements.extend(disclaimer_elements)
        
        # Build PDF
        temp_pdf = "temp_report_content.pdf"
        doc = SimpleDocTemplate(
            temp_pdf,
            pagesize=landscape(A4),
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
        
        # Add page numbers
        numbered_pdf = "temp_numbered.pdf"
        add_page_numbers(temp_pdf, numbered_pdf)
        
        # Optionally merge with template
        template_path = self._get_template_path("Portfolio_Visualiser_Template_Landscape.pdf")
        if os.path.exists(template_path):
            merge_with_template(template_path, numbered_pdf, output_filename)
        else:
            import shutil
            shutil.copy(numbered_pdf, output_filename)
        
        # Cleanup temp files
        cleanup_files = [temp_pdf, numbered_pdf] + all_temp_files
        for f in cleanup_files:
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
    
    # Expected columns for summary table
    expected_cols = [
        'Portfolio ID', 'Portfolio Type', 'Classification', 
        'Risk rating (0-5)', 'Activated date', 'Value (SGD)', 
        'Allocation', 'Portfolio Returns', 'Returns (Q1)'
    ]
    
    # Rename columns if needed (map common variations)
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
    
    return df

def prepare_income_portfolio_df(portfolios_data):
    """
    Prepares the income portfolio DataFrame with yield columns.
    
    Args:
        portfolios_data: List of dicts or DataFrame
    
    Returns:
        DataFrame: Formatted for income portfolios table
    """
    if isinstance(portfolios_data, list):
        df = pd.DataFrame(portfolios_data)
    else:
        df = portfolios_data.copy()
    
    # Expected columns for income table
    expected_cols = [
        'Portfolio ID', 'Portfolio Type', 'Risk rating (0-5)', 
        'Value (SGD)', 'Est. Yield (%)*', 'Ann. Proj. Dividend (SGD)**',
        'Dividend Option', 'Maturity period', 'Allocation'
    ]
    
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
    
    return df

# ==========================================
# CONVENIENCE FUNCTION
# ==========================================

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
    
    Args:
        client_id: Client identifier
        portfolio_data: Dict with 'summary', 'growth', 'income' DataFrames
        exposure_data: Dict with 'summary', 'growth', 'income' exposure dicts
        report_date: Report date
        data_path: Path to data folder
        output_path: Output file path
    
    Returns:
        str: Path to generated PDF
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
    
    # Create sample data matching the PDF structure
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
            'Fixed Income': 0.35,
            'Equity': 0.25,
            'Cash': 0.25,
            'REITs': 0.15
        }),
        'country': pd.Series({
            'United States': 0.45,
            'Singapore': 0.30,
            'China': 0.10,
            'Japan': 0.08,
            'Others': 0.07
        }),
        'sector': pd.Series({
            'Financial Services': 0.20,
            'Technology': 0.18,
            'Real Estate': 0.15,
            'Healthcare': 0.12,
            'Consumer': 0.10,
            'Others': 0.25
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
            include_disclaimers=True
        )
        print(f"\nSUCCESS: Report generated at {output_file}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

