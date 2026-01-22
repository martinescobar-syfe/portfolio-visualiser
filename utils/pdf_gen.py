
import os
import io
import pandas as pd
import streamlit as st
import datetime
from datetime import date
import plotly.io as pio
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Image, Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter

# Import plotting functions to recreate charts
from utils.plotting import (
    plot_cumulative_returns, plot_monthly_heatmap, 
    plot_scatter_risk_return, plot_rolling_volatility, 
    plot_drawdowns, plot_correlation_heatmap
)
from utils.exposure import get_exposure_charts

# Import the new portfolio report generator
from utils.portfolio_report_generator import (
    PortfolioReportGenerator,
    prepare_holdings_dataframe,
    prepare_returns_data,
    prepare_cashflow_data,
    generate_client_report
)

# --- User Provided PDF Functions ---

def create_table(df, title):
    elements = []
    if df.empty:
        return elements
    else:
        # Convert all to string for PDF table stability
        df_str = df.astype(str)
        table_data = [df.columns.tolist()]  # Add header row
        for _, row in df_str.iterrows():
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

        # Highlight rows based on Portfolio ID logic (if applicable)
        if 'Portfolio ID' in df.columns:
             for i in range(len(df)):
                  if df.iloc[i]['Portfolio ID'] in ['Growth Portfolios', 'Income & Preservation Portfolios', "Core", "Specialised", "Income+", "REITs", "Cash / MMF"]:
                      table_style.add("BACKGROUND", (0, i + 1), (-1, i + 1), colors.HexColor("#879be4"))

        table.setStyle(table_style)

        custom_title_style = ParagraphStyle(
            name="CustomTitle",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=12,
            alignment=0,
        )

        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(f"{title}", custom_title_style))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(table)

    return elements

def add_page_numbers(input_pdf, output_pdf, page_size=landscape(A4)):
    page_width, page_height = page_size
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    total_pages = len(reader.pages)

    for page_num, page in enumerate(reader.pages, start=1):
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=page_size)
        can.setFont("Helvetica", 8)
        page_number_text = f"Page {page_num} of {total_pages}"
        text_width = can.stringWidth(page_number_text, "Helvetica", 8)
        x_position = page_width - 20 - text_width
        y_position = 20
        can.drawString(x_position, y_position, page_number_text)
        can.save()
        packet.seek(0)
        page_overlay = PdfReader(packet)
        page.merge_page(page_overlay.pages[0])
        writer.add_page(page)

    with open(output_pdf, "wb") as f:
        writer.write(f)

def create_pdf_with_table_and_charts(output_file, content_blocks, col_width_manual=None):
    """
    Modified to accept a list of content blocks (dicts) instead of single table/charts.
    block = {'type': 'table'|'chart', 'data': ...}
    """
    elements = []
    temp_images = []
    
    doc = SimpleDocTemplate(
        output_file,
        pagesize=landscape(A4),
        leftMargin=0.4 * inch,
        rightMargin=0.3 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.2 * inch
    )
    
    page_width = landscape(A4)[0]
    available_width = page_width - (doc.leftMargin + doc.rightMargin)

    for block in content_blocks:
        if block['type'] == 'header':
            custom_title_style = ParagraphStyle(
                name="SectionTitle",
                fontName="Helvetica-Bold",
                fontSize=14,
                leading=16,
                alignment=0,
                spaceBefore=12,
                spaceAfter=12
            )
            elements.append(Paragraph(block['title'], custom_title_style))
            elements.append(Spacer(1, 0.1 * inch))

        elif block['type'] == 'table':
            table_elems = create_table(block['df'], block.get('title', ''))
            if table_elems:
                # Last element is the table object
                table = table_elems[-1]
                if isinstance(table, Table) and not isinstance(table, Spacer): # Safety check
                    # Auto-width logic as per user code
                    try:
                        text_lengths = block['df'].astype(str).applymap(len).sum(axis=0)
                        total_text_length = sum(text_lengths)
                        if total_text_length > 0:
                            if col_width_manual is None:
                                col_widths = [(available_width * (col_size / total_text_length)) for col_size in text_lengths]
                                table._argW = col_widths
                            else:
                                table._argW = col_width_manual
                    except:
                        pass # Fallback to auto
                
                elements.extend(table_elems)
                elements.append(Spacer(1, 0.2 * inch))

        elif block['type'] == 'charts':
            charts = block['charts']
            if not charts: continue
            
            num_charts = len(charts)
            cols_per_row = 2 # Fixed for consistency
            chart_width = 4.5 * inch
            chart_height = 3.0 * inch

            chart_grid = []
            row = []
            
            for idx, chart in enumerate(charts):
                image_file = f"temp_chart_{len(temp_images)}_{idx}.png"
                try:
                    # kaleido needed
                    chart.write_image(image_file, format="png", width=1200, height=800, scale=2)
                    temp_images.append(image_file)
                    img = Image(image_file, width=chart_width, height=chart_height)
                    row.append(img)
                except Exception as e:
                    print(f"Chart gen failed: {e}")
                    continue

                if len(row) == cols_per_row or idx == len(charts) - 1:
                    chart_grid.append(row)
                    row = []

            for chart_row in chart_grid:
                col_widths = [available_width / len(chart_row)] * len(chart_row)
                chart_table = Table([chart_row], colWidths=col_widths)
                chart_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
                elements.append(chart_table)
                elements.append(Spacer(1, 0.1 * inch))

    try:
        doc.build(elements)
    except Exception as e:
        print(f"PDF Build Failed: {e}")
    
    # Cleanup
    for img in temp_images:
        if os.path.exists(img):
            os.remove(img)

# --- Integration Logic ---

# Column name mapping for clean display
COLUMN_NAME_MAPPING = {
    'nav_in_sgd': 'Net Asset Value (SGD)',
    'nav': 'Net Asset Value',
    'pnl_inception': 'P&L Since Inception',
    'pnl': 'Profit/Loss',
    'invested_amount': 'Invested Amount',
    'portfolio_name': 'Portfolio Name',
    'portfolio_id': 'Portfolio ID',
    'type': 'Portfolio Type',
    'name': 'Name',
    'code': 'Code',
    'allocation': 'Allocation (%)',
    'balance': 'Balance',
    'status': 'Status',
    'currency': 'Currency',
    'risk_rating': 'Risk Rating',
    'activated_date': 'Activated Date',
    'est_yield': 'Est. Yield (%)',
    'dividend_option': 'Dividend Option',
}

# Portfolio type to classification mapping
PORTFOLIO_CLASSIFICATION = {
    'Core Equity100': 'Core',
    'Core Growth': 'Core',
    'Core Balanced': 'Core',
    'Core Defensive': 'Core',
    'SRS Equity100': 'Core',
    'China Growth': 'Thematic',
    'Disruptive Technology': 'Thematic',
    'ESG & Clean Energy': 'Thematic',
    'Healthcare Innovation': 'Thematic',
    'Income Enhance': 'Income+',
    'Income Preserve': 'Income+',
    'SRS Income Enhance': 'Income+',
    'SRS Income Preserve': 'Income+',
    'REIT 100': 'REITs',
    'REIT Risk Managed': 'REITs',
    'Cash SGD flexi': 'Cash / MMF',
    'Cash USD flexi': 'Cash / MMF',
    'Cash SGD guaranteed': 'Cash / MMF',
    'SRS Cash SGD flexi': 'Cash / MMF',
    'Blackrock Private Credit Fund': 'Private Credit',
    'Monroe Income Plus Fund': 'Private Credit',
    'Silverdale Fund (Nov 2027)': 'Private Credit',
}

# Portfolio type to objective mapping
PORTFOLIO_OBJECTIVE = {
    'Core Equity100': 'Wealth Growth',
    'Core Growth': 'Wealth Growth',
    'Core Balanced': 'Wealth Growth',
    'SRS Equity100': 'Wealth Growth',
    'China Growth': 'Wealth Growth',
    'Disruptive Technology': 'Wealth Growth',
    'ESG & Clean Energy': 'Wealth Growth',
    'Healthcare Innovation': 'Wealth Growth',
    'Income Enhance': 'Income Generation',
    'Income Preserve': 'Income Generation',
    'SRS Income Enhance': 'Income Generation',
    'SRS Income Preserve': 'Income Generation',
    'REIT 100': 'Income Generation',
    'REIT Risk Managed': 'Income Generation',
    'Blackrock Private Credit Fund': 'Income Generation',
    'Monroe Income Plus Fund': 'Income Generation',
    'Silverdale Fund (Nov 2027)': 'Income Generation',
    'Core Defensive': 'Capital Preservation',
    'Cash SGD flexi': 'Capital Preservation',
    'Cash USD flexi': 'Capital Preservation',
    'Cash SGD guaranteed': 'Capital Preservation',
    'SRS Cash SGD flexi': 'Capital Preservation',
}

# Syfe chart colors
SYFE_CHART_COLORS = [
    '#0052CC',  # Primary Blue
    '#36B37E',  # Green
    '#FFAB00',  # Yellow
    '#FF5630',  # Red
    '#00B8D9',  # Teal
    '#6554C0',  # Purple
    '#FF8B00',  # Orange
    '#2684FF',  # Secondary Blue
    '#4C9AFF',  # Tertiary Blue
    '#172B4D',  # Dark Blue
]

def format_column_name(col_name):
    """Convert column names to clean display format."""
    if col_name in COLUMN_NAME_MAPPING:
        return COLUMN_NAME_MAPPING[col_name]
    # Convert snake_case to Title Case
    return col_name.replace('_', ' ').title()

def format_dataframe_for_pdf(df):
    """Format DataFrame columns and values for PDF display."""
    df = df.copy()
    
    # Rename columns
    new_columns = {col: format_column_name(col) for col in df.columns}
    df = df.rename(columns=new_columns)
    
    # Format numeric columns
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Check if it's a percentage column
            if 'Allocation' in col or '%' in col or 'Return' in col:
                df[col] = df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) and abs(x) < 10 else f"{x:.2f}%" if pd.notna(x) else '-')
            elif 'Value' in col or 'Amount' in col or 'SGD' in col or 'P&L' in col or 'Balance' in col or 'NAV' in col:
                df[col] = df[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else '-')
            elif 'Rating' in col:
                df[col] = df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else '-')
            else:
                df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else '-')
    
    return df

def aggregate_by_portfolio_type(df):
    """Aggregate portfolio data by type and calculate summary."""
    if df.empty:
        return pd.DataFrame()
    
    # Identify value column
    value_col = None
    for col in ['nav_in_sgd', 'nav', 'value', 'balance']:
        if col in df.columns:
            value_col = col
            break
    
    if value_col is None:
        return df
    
    # Identify type column
    type_col = None
    for col in ['type', 'portfolio_type', 'name']:
        if col in df.columns:
            type_col = col
            break
    
    if type_col is None:
        return df
    
    # Aggregate by type
    agg_df = df.groupby(type_col).agg({
        value_col: 'sum'
    }).reset_index()
    
    # Calculate allocation
    total = agg_df[value_col].sum()
    agg_df['allocation'] = agg_df[value_col] / total if total > 0 else 0
    
    # Add classification and objective
    agg_df['classification'] = agg_df[type_col].map(PORTFOLIO_CLASSIFICATION).fillna('Other')
    agg_df['objective'] = agg_df[type_col].map(PORTFOLIO_OBJECTIVE).fillna('Other')
    
    # Sort by value descending
    agg_df = agg_df.sort_values(value_col, ascending=False)
    
    return agg_df

def create_exposure_pie_chart(data, title, colors=None):
    """Create a pie chart with Syfe styling using Plotly."""
    
    if data is None or (isinstance(data, pd.Series) and data.empty):
        return None
    
    if isinstance(data, dict):
        data = pd.Series(data)
    
    # Filter small values and limit to top 10
    data = data[data > 0.001]
    if len(data) > 10:
        top_data = data.head(9)
        others = data.iloc[9:].sum()
        if others > 0:
            top_data['Others'] = others
        data = top_data
    
    if colors is None:
        colors = SYFE_CHART_COLORS
    
    fig = go.Figure(data=[go.Pie(
        labels=data.index,
        values=data.values,
        hole=0.4,
        marker=dict(colors=colors[:len(data)]),
        textinfo='percent+label',
        textposition='outside',
        textfont=dict(size=9),
    )])
    
    fig.update_layout(
        title=dict(text=f'<b>{title}</b>', font=dict(size=12, color='#172B4D'), x=0.5),
        showlegend=True,
        legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='right', x=1.3, font=dict(size=8)),
        margin=dict(l=20, r=100, t=50, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=500,
        height=350
    )
    
    return fig


def create_native_pie_chart_drawing(data, title="", width=220, height=180):
    """
    Creates a native ReportLab pie chart drawing (for PDF export without Plotly).
    
    Args:
        data: pd.Series or dict with labels as keys and values as data
        title: Chart title
        width, height: Drawing dimensions
    
    Returns:
        Drawing object or None
    """
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.legends import Legend
    from reportlab.lib import colors as rl_colors
    
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
    
    # Native colors
    PIE_COLORS_NATIVE = [
        rl_colors.HexColor("#0052CC"),
        rl_colors.HexColor("#36B37E"),
        rl_colors.HexColor("#FFAB00"),
        rl_colors.HexColor("#FF5630"),
        rl_colors.HexColor("#00B8D9"),
        rl_colors.HexColor("#6554C0"),
        rl_colors.HexColor("#FF8B00"),
        rl_colors.HexColor("#2684FF"),
        rl_colors.HexColor("#4C9AFF"),
        rl_colors.HexColor("#172B4D"),
    ]
    
    # Create drawing
    d = Drawing(width, height)
    
    # Create pie chart
    pie = Pie()
    pie.x = width // 2 - 50
    pie.y = 25
    pie.width = 100
    pie.height = 100
    pie.data = list(data.values)
    pie.labels = None
    
    # Set colors
    for i in range(len(data)):
        pie.slices[i].fillColor = PIE_COLORS_NATIVE[i % len(PIE_COLORS_NATIVE)]
        pie.slices[i].strokeColor = rl_colors.white
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
    legend.strokeColor = rl_colors.white
    legend.deltax = 5
    legend.deltay = 3
    legend.dxTextSpace = 4
    legend.alignment = 'right'
    
    # Legend items with percentages
    legend_items = []
    for i, (label, value) in enumerate(data.items()):
        color = PIE_COLORS_NATIVE[i % len(PIE_COLORS_NATIVE)]
        pct = value * 100
        short_label = str(label)[:12] + ('...' if len(str(label)) > 12 else '')
        legend_items.append((color, f"{short_label} ({pct:.1f}%)"))
    
    legend.colorNamePairs = legend_items
    
    # Add title
    title_str = String(width // 2, height - 10, title,
                       fontName='Helvetica-Bold', fontSize=9,
                       fillColor=rl_colors.HexColor("#172B4D"), textAnchor='middle')
    
    d.add(pie)
    d.add(legend)
    d.add(title_str)
    
    return d

def generate_report_content(modules):
    """Gather data and charts based on selected modules."""
    content_blocks = []
    
    # 1. Client Data Source
    if "1. Client Data Source" in modules:
        content_blocks.append({'type': 'header', 'title': '1. Client Data Source'})
        
        # Client Info
        if 'metabase_client_info' in st.session_state:
            info = st.session_state['metabase_client_info']
            # Clean up the info display
            clean_info = {}
            for k, v in info.items():
                clean_key = format_column_name(k)
                clean_info[clean_key] = v if v is not None else '-'
            df_info = pd.DataFrame([clean_info])
            content_blocks.append({'type': 'table', 'df': df_info, 'title': 'Client Profile'})
        
        # Portfolio Data - Aggregate by Type
        if 'metabase_client_df' in st.session_state:
            df_port = st.session_state['metabase_client_df'].copy()
            if not df_port.empty:
                # Aggregate by portfolio type
                agg_df = aggregate_by_portfolio_type(df_port)
                
                if not agg_df.empty:
                    # Select and rename columns for display
                    display_cols = []
                    for col in ['type', 'portfolio_type', 'name']:
                        if col in agg_df.columns:
                            display_cols.append(col)
                            break
                    
                    for col in ['nav_in_sgd', 'nav', 'value', 'balance']:
                        if col in agg_df.columns:
                            display_cols.append(col)
                            break
                    
                    if 'allocation' in agg_df.columns:
                        display_cols.append('allocation')
                    if 'classification' in agg_df.columns:
                        display_cols.append('classification')
                    if 'objective' in agg_df.columns:
                        display_cols.append('objective')
                    
                    df_display = format_dataframe_for_pdf(agg_df[display_cols])
                    
                    # Add total row
                    total_row = {col: '' for col in df_display.columns}
                    total_row[df_display.columns[0]] = 'Total'
                    
                    value_col = df_display.columns[1] if len(df_display.columns) > 1 else None
                    if value_col:
                        # Sum the original numeric values
                        orig_val_col = [c for c in agg_df.columns if 'nav' in c.lower() or 'value' in c.lower() or 'balance' in c.lower()]
                        if orig_val_col:
                            total_val = agg_df[orig_val_col[0]].sum()
                            total_row[value_col] = f"{total_val:,.2f}"
                    
                    if 'Allocation (%)' in total_row:
                        total_row['Allocation (%)'] = '100.00%'
                    
                    df_display = pd.concat([df_display, pd.DataFrame([total_row])], ignore_index=True)
                    
                    content_blocks.append({'type': 'table', 'df': df_display, 'title': 'Portfolio Summary by Type'})
                    
                    # --- Exposure Charts ---
                    charts = []
                    
                    # By Portfolio Type
                    type_col = [c for c in agg_df.columns if c in ['type', 'portfolio_type', 'name']][0] if any(c in agg_df.columns for c in ['type', 'portfolio_type', 'name']) else None
                    val_col = [c for c in agg_df.columns if 'nav' in c.lower() or 'value' in c.lower()][0] if any('nav' in c.lower() or 'value' in c.lower() for c in agg_df.columns) else None
                    
                    if type_col and val_col:
                        type_exposure = agg_df.set_index(type_col)[val_col]
                        type_exposure = type_exposure / type_exposure.sum()
                        fig_type = create_exposure_pie_chart(type_exposure, 'By Portfolio Type')
                        if fig_type:
                            charts.append(fig_type)
                    
                    # By Classification
                    if 'classification' in agg_df.columns and val_col:
                        class_exposure = agg_df.groupby('classification')[val_col].sum()
                        class_exposure = class_exposure / class_exposure.sum()
                        fig_class = create_exposure_pie_chart(class_exposure, 'By Asset Class')
                        if fig_class:
                            charts.append(fig_class)
                    
                    # By Objective
                    if 'objective' in agg_df.columns and val_col:
                        obj_exposure = agg_df.groupby('objective')[val_col].sum()
                        obj_exposure = obj_exposure / obj_exposure.sum()
                        fig_obj = create_exposure_pie_chart(obj_exposure, 'By Investment Objective')
                        if fig_obj:
                            charts.append(fig_obj)
                    
                    if charts:
                        content_blocks.append({'type': 'charts', 'charts': charts})

    # 4. Analysis Results
    if "4. Analysis Results" in modules:
         content_blocks.append({'type': 'header', 'title': '4. Analysis Results'})
         
         if 'optimization_results' in st.session_state:
             res_data = st.session_state['optimization_results']
             if res_data:
                 ret_all = res_data.get('ret_all')
                 ret_bm = res_data.get('ret_bm')
             else:
                 ret_all = None
                 ret_bm = None
             
             if ret_all is not None:
                 charts = []
                 cum_ret = (1 + ret_all).cumprod()
                 fig_cum = plot_cumulative_returns(cum_ret, benchmark_ret=ret_bm)
                 charts.append(fig_cum)
                 
                 fig_risk = plot_scatter_risk_return(ret_all, benchmark_ret=ret_bm)
                 charts.append(fig_risk)
                 
                 fig_dd = plot_drawdowns(ret_all)
                 charts.append(fig_dd)
                 
                 content_blocks.append({'type': 'charts', 'charts': charts})

    # 5. Exposure Analysis
    if "5. Exposure Analysis" in modules:
        content_blocks.append({'type': 'header', 'title': '5. Exposure Analysis'})
        
        charts = []
        
        # Check for exposure data from PDF parsing
        pdf_key = None
        for key in ['exist_pdf_parsed_data', 'client_pv_parsed_data', 'parsed_portfolio_data']:
            if key in st.session_state:
                pdf_key = key
                break
        
        if pdf_key and pdf_key in st.session_state:
            data = st.session_state[pdf_key]
            
            # Country Exposure
            country_exp = data.get('country_exp', {})
            if country_exp:
                country_series = pd.Series(country_exp)
                fig_country = create_exposure_pie_chart(country_series, 'Country Exposure')
                if fig_country:
                    charts.append(fig_country)
            
            # Sector Exposure
            sector_exp = data.get('sector_exp', {})
            if sector_exp:
                sector_series = pd.Series(sector_exp)
                fig_sector = create_exposure_pie_chart(sector_series, 'Sector Exposure')
                if fig_sector:
                    charts.append(fig_sector)
            
            # Top Holdings Table
            holdings_df = data.get('holdings_df', pd.DataFrame())
            if not holdings_df.empty:
                # Show top 10 holdings
                top_holdings = holdings_df.head(10).copy()
                top_holdings = format_dataframe_for_pdf(top_holdings)
                content_blocks.append({'type': 'table', 'df': top_holdings, 'title': 'Top 10 Holdings'})
        
        # Check for exposure from client upload
        if 'client_exposure_charts' in st.session_state:
            exp_charts = st.session_state['client_exposure_charts']
            
            if 'sector_pie' in exp_charts:
                charts.append(exp_charts['sector_pie'])
            if 'country_pie' in exp_charts:
                charts.append(exp_charts['country_pie'])
            if 'asset_pie' in exp_charts:
                charts.append(exp_charts['asset_pie'])
        
        if charts:
            content_blocks.append({'type': 'charts', 'charts': charts})

    return content_blocks

def render_export_ui(modules_selected):
    if st.button("Generate PDF Report", key="gen_pdf_btn"):
        if not modules_selected:
            st.error("Please select at least one module to export.")
            return

        with st.spinner("Generating PDF..."):
            # Check if we have client data from either CSV or Metabase
            df_port = None
            client_info = {}
            
            # Try CSV upload first
            if 'csv_client_df' in st.session_state and st.session_state['csv_client_df'] is not None:
                df_port = st.session_state['csv_client_df'].copy()
                client_info = st.session_state.get('metabase_client_info', {'client_id': 'CSV_Client'})
            # Then try Metabase
            elif 'metabase_client_df' in st.session_state and st.session_state['metabase_client_df'] is not None:
                if not st.session_state['metabase_client_df'].empty:
                    df_port = st.session_state['metabase_client_df'].copy()
                    client_info = st.session_state.get('metabase_client_info', {'client_id': 'Client'})
            
            has_client_data = df_port is not None and not df_port.empty
            
            if has_client_data and "1. Client Data Source" in modules_selected:
                try:
                    from utils.portfolio_report_generator import (
                        PortfolioReportGenerator, 
                        prepare_holdings_dataframe,
                        prepare_returns_data,
                        prepare_dividends_data,
                        prepare_cashflow_data,
                        calculate_portfolio_exposure
                    )
                    from utils.data_loader import load_static_data, get_data_path
                    from utils.exposure import calculate_exposure, group_tail
                    
                    # Extract client_id from CSV's client_id column
                    df_temp = df_port.copy()
                    df_temp.columns = df_temp.columns.str.lower().str.strip()
                    if 'client_id' in df_temp.columns:
                        # Get first non-null client_id from CSV
                        client_id_values = df_temp['client_id'].dropna().unique()
                        if len(client_id_values) > 0:
                            client_id = str(client_id_values[0])
                        else:
                            client_id = str(client_info.get('client_id', 'Client'))
                    else:
                        client_id = str(client_info.get('client_id', 'Client'))
                    
                    client_id = client_id.replace(' ', '_')
                    
                    # Load exposure matrices from Inv_Data_Hardcoded.xlsx
                    data_path = get_data_path()
                    df_sec_list, df_asset_class, df_sector, df_country, df_fi_metrics, df_fx, df_daily_ret, df_risk_rating, df_underlying = load_static_data(data_path)
                    
                    # Prepare holdings table
                    df_holdings = prepare_holdings_dataframe(df_port)
                    
                    # Prepare returns data for bar chart
                    returns_data = prepare_returns_data(df_port)
                    
                    # Prepare dividends data for bar chart
                    print(f"[DEBUG] CSV columns: {list(df_port.columns)}")
                    dividends_data = prepare_dividends_data(df_port)
                    print(f"[DEBUG] Dividends data length: {len(dividends_data) if dividends_data is not None else 0}")
                    
                    # ============================================
                    # CALCULATE EXPOSURES USING SUMPRODUCT LOGIC
                    # ============================================
                    
                    # Portfolio type to ticker mapping (matches ui_functions.py)
                    PORTFOLIO_TICKER_MAP = {
                        # Core Portfolios
                        'GLOBAL_EQUITY_100': 'Core Equity100',
                        'GLOBAL_EQUITY': 'Core Equity100',
                        'CORE_EQUITY100': 'Core Equity100',
                        'CORE_EQUITY_100': 'Core Equity100',
                        'EQUITY100': 'Core Equity100',
                        'CORE_GROWTH': 'Core Growth',
                        'CORE_BALANCED': 'Core Balanced',
                        'CORE_DEFENSIVE': 'Core Defensive',
                        'DOWNSIDE_PROTECTED': 'Downside protected',
                        # Thematic Portfolios
                        'CHINA_GROWTH': 'China Growth',
                        'DISRUPTIVE_TECHNOLOGY': 'Disruptive Technology',
                        'DISRUPTIVE_TECH': 'Disruptive Technology',
                        'ESG_AND_CLEAN_ENERGY': 'ESG & Clean Energy',
                        'ESG_CLEAN_ENERGY': 'ESG & Clean Energy',
                        'HEALTHCARE_INNOVATION': 'Healthcare Innovation',
                        # Income Portfolios
                        'INCOME_ENHANCE': 'Income Enhance',
                        'INCOME_ENHANCED': 'Income Enhance',
                        'INCOME_PRESERVE': 'Income Preserve',
                        # REIT Portfolios
                        'REIT': 'REIT 100',
                        'REIT_100': 'REIT 100',
                        'REIT_RISK_MANAGED': 'REIT Risk Managed',
                        'REIT_RM': 'REIT Risk Managed',
                        # Private Credit / AI Products
                        'BLACKROCK': 'Blackrock Private Credit Fund',
                        'BLACKROCK_PRIVATE_CREDIT': 'Blackrock Private Credit Fund',
                        'BLACKROCK_PCF': 'Blackrock Private Credit Fund',
                        'MONROE': 'Monroe Income Plus Fund',
                        'MONROE_INCOME_PLUS': 'Monroe Income Plus Fund',
                        'MONROE_CAPITAL': 'Monroe Income Plus Fund',
                        'SILVERDALE': 'Silverdale Fund (Nov 2027)',
                        # Cash Management
                        'CASH_PLUS_GUARANTEED_SGD': 'Cash SGD guaranteed',
                        'CASH_PLUS_GUARANTEED': 'Cash SGD guaranteed',
                        'CASH_GUARANTEED': 'Cash SGD guaranteed',
                        'CASH_PLUS_USD': 'Cash USD flexi',
                        'CASH_USD': 'Cash USD flexi',
                        'CASH_PLUS': 'Cash SGD flexi',
                        'CASH_SGD': 'Cash SGD flexi',
                        'CASH_PLUS_SGD': 'Cash SGD flexi',
                        # SRS Portfolios
                        'SRS_INCOME_ENHANCE': 'SRS Income Enhance',
                        'SRS_INCOME_ENHANCED': 'SRS Income Enhance',
                        'SRS_INCOME_PRESERVE': 'SRS Income Preserve',
                        'SRS_CASH_PLUS': 'SRS Cash SGD flexi',
                        'SRS_CASH': 'SRS Cash SGD flexi',
                        'SRS_GLOBAL_EQUITY_100': 'SRS Equity100',
                    }
                    
                    # Build weights from CSV data
                    client_weights = {}
                    total_nav = 0.0
                    
                    def safe_float(val):
                        """Convert value to float, handling comma-separated numbers."""
                        if pd.isna(val):
                            return 0.0
                        if isinstance(val, (int, float)):
                            return float(val)
                        # Remove commas from string representation
                        return float(str(val).replace(',', ''))
                    
                    if 'nav_in_sgd' in df_port.columns and 'type' in df_port.columns:
                        for _, row in df_port.iterrows():
                            port_type = str(row.get('type', '')).upper().strip()
                            nav = safe_float(row.get('nav_in_sgd', 0))
                            
                            if nav < 50:
                                continue
                            
                            # Map to internal ticker or use type directly
                            ticker = PORTFOLIO_TICKER_MAP.get(port_type, port_type)
                            
                            client_weights[ticker] = client_weights.get(ticker, 0) + nav
                            total_nav += nav
                    
                    # Normalize weights
                    if total_nav > 0:
                        for k in client_weights:
                            client_weights[k] /= total_nav
                    
                    # ============================================
                    # USE EXPOSURE DATA FROM SESSION STATE (same as app display)
                    # ============================================
                    exposure_data = {}
                    
                    # First, try to use the stored exposure data from the app display
                    if 'asset_class_exposure' in st.session_state and st.session_state['asset_class_exposure'] is not None:
                        exposure_data['asset_class'] = st.session_state['asset_class_exposure']
                    
                    if 'country_exposure' in st.session_state and st.session_state['country_exposure'] is not None:
                        exposure_data['country'] = st.session_state['country_exposure']
                    
                    if 'sector_exposure' in st.session_state and st.session_state['sector_exposure'] is not None:
                        exposure_data['sector'] = st.session_state['sector_exposure']
                    
                    # Fallback: Calculate exposures if not available in session_state
                    if not exposure_data:
                        weights_series = pd.Series({f"{k}_wgt": v for k, v in client_weights.items()})
                        
                        if df_asset_class is not None and not df_asset_class.empty:
                            asset_exp = calculate_exposure(weights_series, df_asset_class)
                            if not asset_exp.empty:
                                asset_exp = group_tail(asset_exp, top_n=8)
                                exposure_data['asset_class'] = asset_exp
                        
                        if df_country is not None and not df_country.empty:
                            country_exp = calculate_exposure(weights_series, df_country)
                            if not country_exp.empty:
                                country_exp = group_tail(country_exp, top_n=8)
                                exposure_data['country'] = country_exp
                        
                        if df_sector is not None and not df_sector.empty:
                            sector_exp = calculate_exposure(weights_series, df_sector)
                            if not sector_exp.empty:
                                sector_exp = group_tail(sector_exp, top_n=8)
                                exposure_data['sector'] = sector_exp
                        
                        # If still nothing, use portfolio type breakdown
                        if not exposure_data:
                            type_exp = pd.Series(client_weights)
                            exposure_data['asset_class'] = group_tail(type_exp, top_n=8)
                    
                    # Generate report
                    generator = PortfolioReportGenerator(
                        client_id=client_id,
                        report_date=date.today(),
                        data_path='Data/'
                    )
                    
                    final_filename = f"Client_{client_id}_Portfolio_Summary_{date.today().strftime('%Y%m%d')}.pdf"
                    
                    # Calculate cash flow data from CSV (using yields)
                    print(f"[DEBUG] Preparing cash flow data from CSV...")
                    cashflow_data = prepare_cashflow_data(df_port, df_fi_metrics)
                    cashflow_currency = 'SGD'
                    
                    if cashflow_data:
                        print(f"[DEBUG] Cash flow data generated with {len(cashflow_data)} portfolios")
                    else:
                        print(f"[DEBUG] No cash flow data generated")
                    
                    generated_file = generator.generate_report(
                        df_holdings=df_holdings,
                        exposure_data=exposure_data if exposure_data else None,
                        returns_data=returns_data,
                        dividends_data=dividends_data if dividends_data is not None and len(dividends_data) > 0 else None,
                        cashflow_data=cashflow_data,
                        cashflow_currency=cashflow_currency,
                        output_filename=final_filename,
                        include_cover=True,
                        include_appendix=True
                    )
                    
                    # Read for download
                    with open(generated_file, "rb") as f:
                        pdf_data = f.read()
                    
                    st.success(f"PDF generated successfully! Client: {client_id}")
                    
                    st.download_button(
                        label="📥 Download Portfolio Summary PDF",
                        data=pdf_data,
                        file_name=final_filename,
                        mime="application/pdf"
                    )
                    
                    # Cleanup
                    if os.path.exists(generated_file):
                        os.remove(generated_file)
                    
                    return
                        
                except Exception as e:
                    st.error(f"Error generating PDF: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
            
            # Fallback to standard export if no client data
            content_blocks = generate_report_content(modules_selected)
            
            if not content_blocks:
                st.warning("No data found for selected modules. Please load client data first (CSV or Metabase).")
                return

            # Create Temp PDF
            temp_filename = "temp_report.pdf"
            final_filename = f"Portfolio_Report_{datetime.date.today()}.pdf"
            
            create_pdf_with_table_and_charts(temp_filename, content_blocks)
            
            # Add Page Numbers
            add_page_numbers(temp_filename, final_filename)
            
            # Read back for download
            with open(final_filename, "rb") as f:
                pdf_data = f.read()
            
            # Show Download Button
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_data,
                file_name=final_filename,
                mime="application/pdf"
            )
            
            # Cleanup
            if os.path.exists(temp_filename): os.remove(temp_filename)
            if os.path.exists(final_filename): os.remove(final_filename)


def render_portfolio_summary_export_ui(key_prefix="portfolio_summary"):
    """
    Renders a Streamlit UI for generating Portfolio Summary PDFs similar to the 
    Syfe Portfolio Summary format with tables and exposure charts.
    
    This function should be called from the main app or a page where client
    portfolio data is available in session state.
    """
    st.markdown("### 📄 Generate Portfolio Summary PDF")
    
    with st.expander("PDF Report Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            client_id = st.text_input(
                "Client ID",
                value=st.session_state.get('client_id', 'Client'),
                key=f"{key_prefix}_client_id"
            )
            
            report_date = st.date_input(
                "Report Date",
                value=date.today(),
                key=f"{key_prefix}_report_date"
            )
        
        with col2:
            include_cover = st.checkbox("Include Cover Page", value=True, key=f"{key_prefix}_cover")
            include_disclaimers = st.checkbox("Include Disclaimers", value=True, key=f"{key_prefix}_disclaimers")
            include_growth = st.checkbox("Include Growth Portfolios Page", value=True, key=f"{key_prefix}_growth")
            include_income = st.checkbox("Include Income & Preservation Page", value=True, key=f"{key_prefix}_income")
    
    if st.button("🖨️ Generate Portfolio Summary PDF", key=f"{key_prefix}_gen_btn"):
        # Check for data in session state
        portfolio_data = None
        exposure_data = None
        
        # Try to get data from various session state keys
        if 'metabase_client_df' in st.session_state:
            portfolio_data = st.session_state['metabase_client_df']
        elif 'portfolio_summary_df' in st.session_state:
            portfolio_data = st.session_state['portfolio_summary_df']
        elif 'client_portfolios' in st.session_state:
            portfolio_data = st.session_state['client_portfolios']
        
        if portfolio_data is None or (isinstance(portfolio_data, pd.DataFrame) and portfolio_data.empty):
            st.error("No portfolio data found. Please load client data first.")
            return
        
        with st.spinner("Generating Portfolio Summary PDF..."):
            try:
                # Prepare the data
                df_summary = prepare_portfolio_summary_df(portfolio_data)
                
                # Get exposure data if available
                exposure_dict = {}
                if 'asset_class_exposure' in st.session_state:
                    exposure_dict['asset_class'] = st.session_state['asset_class_exposure']
                if 'country_exposure' in st.session_state:
                    exposure_dict['country'] = st.session_state['country_exposure']
                if 'sector_exposure' in st.session_state:
                    exposure_dict['sector'] = st.session_state['sector_exposure']
                
                # Prepare growth and income DataFrames if separate data exists
                df_growth = None
                df_income = None
                
                if include_growth and 'growth_portfolios_df' in st.session_state:
                    df_growth = st.session_state['growth_portfolios_df']
                
                if include_income and 'income_portfolios_df' in st.session_state:
                    df_income = prepare_income_portfolio_df(st.session_state['income_portfolios_df'])
                
                # Generate the report
                generator = PortfolioReportGenerator(
                    client_id=client_id,
                    report_date=report_date,
                    data_path='Data/'
                )
                
                output_filename = f"Client_{client_id}_Portfolio_Summary_{report_date.strftime('%Y%m%d')}.pdf"
                
                generated_file = generator.generate_report(
                    df_portfolio_summary=df_summary,
                    df_growth_portfolios=df_growth if include_growth else None,
                    df_income_portfolios=df_income if include_income else None,
                    exposure_data=exposure_dict if exposure_dict else None,
                    output_filename=output_filename,
                    include_cover=include_cover,
                    include_disclaimers=include_disclaimers
                )
                
                # Read the generated file for download
                with open(generated_file, "rb") as f:
                    pdf_data = f.read()
                
                st.success(f"PDF generated successfully!")
                
                # Download button
                st.download_button(
                    label="📥 Download Portfolio Summary PDF",
                    data=pdf_data,
                    file_name=output_filename,
                    mime="application/pdf",
                    key=f"{key_prefix}_download_btn"
                )
                
                # Cleanup
                if os.path.exists(generated_file):
                    os.remove(generated_file)
                    
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def generate_portfolio_summary_pdf_from_data(
    client_id,
    df_portfolios,
    asset_class_exposure=None,
    country_exposure=None,
    sector_exposure=None,
    report_date=None,
    data_path='Data/'
):
    """
    Programmatic function to generate a Portfolio Summary PDF.
    
    Args:
        client_id: Client identifier
        df_portfolios: DataFrame with portfolio data
        asset_class_exposure: Series with asset class weights
        country_exposure: Series with country weights
        sector_exposure: Series with sector weights
        report_date: Report date (defaults to today)
        data_path: Path to data folder with templates
    
    Returns:
        bytes: PDF file content
    """
    if report_date is None:
        report_date = date.today()
    
    # Prepare data
    df_summary = prepare_portfolio_summary_df(df_portfolios)
    
    # Build exposure dict
    exposure_data = {}
    if asset_class_exposure is not None:
        exposure_data['asset_class'] = asset_class_exposure
    if country_exposure is not None:
        exposure_data['country'] = country_exposure
    if sector_exposure is not None:
        exposure_data['sector'] = sector_exposure
    
    # Generate report
    generator = PortfolioReportGenerator(
        client_id=client_id,
        report_date=report_date,
        data_path=data_path
    )
    
    output_filename = f"temp_Client_{client_id}_{report_date.strftime('%Y%m%d')}.pdf"
    
    generated_file = generator.generate_report(
        df_portfolio_summary=df_summary,
        exposure_data=exposure_data if exposure_data else None,
        output_filename=output_filename
    )
    
    # Read and return bytes
    with open(generated_file, "rb") as f:
        pdf_data = f.read()
    
    # Cleanup
    if os.path.exists(generated_file):
        os.remove(generated_file)
    
    return pdf_data
