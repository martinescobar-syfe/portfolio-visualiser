
import os
import io
import pandas as pd
import streamlit as st
import datetime
import plotly.io as pio
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

def generate_report_content(modules):
    """Gather data and charts based on selected modules."""
    content_blocks = []
    
    # 1. Client Data Source
    if "1. Client Data Source" in modules:
        content_blocks.append({'type': 'header', 'title': '1. Client Data Source'})
        if 'metabase_client_info' in st.session_state:
             info = st.session_state['metabase_client_info']
             df_info = pd.DataFrame([info])
             content_blocks.append({'type': 'table', 'df': df_info, 'title': 'Client Profile (Metabase)'})
        
        if 'metabase_client_df' in st.session_state:
             df_port = st.session_state['metabase_client_df']
             if not df_port.empty:
                # Limit columns for PDF width
                disp_cols = ['portfolio_name', 'name', 'code', 'nav_in_sgd', 'allocation']
                existing_cols = [c for c in disp_cols if c in df_port.columns]
                content_blocks.append({'type': 'table', 'df': df_port[existing_cols].head(20), 'title': 'Current Holdings (Top 20)'})

    # 4. Analysis Results (Re-using logic from render_analysis)
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
                 # Charts
                 charts = []
                 # Cumulative
                 cum_ret = (1 + ret_all).cumprod()
                 fig_cum = plot_cumulative_returns(cum_ret, benchmark_ret=ret_bm)
                 charts.append(fig_cum)
                 
                 # Risk/Return
                 fig_risk = plot_scatter_risk_return(ret_all, benchmark_ret=ret_bm)
                 charts.append(fig_risk)
                 
                 # Drawdown
                 fig_dd = plot_drawdowns(ret_all)
                 charts.append(fig_dd)
                 
                 content_blocks.append({'type': 'charts', 'charts': charts})

    # 5. Exposure Analysis
    if "5. Exposure Analysis" in modules:
        content_blocks.append({'type': 'header', 'title': '5. Exposure Analysis'})
        # We need sector/country data to re-gen charts. Passing empty dicts if not readily available in simple session var
        # Assuming DataFrames are re-loaded or passed globally? 
        # For safety/speed in this context, we check if we can get aggregator weights
        pass # Complexity: Requires Access to massive dfs (df_sector, etc) which are passed deeply. 
             # SKIP for now unless requested to fetch them.

    return content_blocks

def render_export_ui(modules_selected):
    if st.button("Generate PDF Report", key="gen_pdf_btn"):
        if not modules_selected:
            st.error("Please select at least one module to export.")
            return

        with st.spinner("Generating PDF..."):
            # 1. Gather Content
            content_blocks = generate_report_content(modules_selected)
            
            if not content_blocks:
                st.warning("No data found for selected modules. Run analysis first.")
                return

            # 2. Create Temp PDF
            temp_filename = "temp_report.pdf"
            final_filename = f"Portfolio_Report_{datetime.date.today()}.pdf"
            
            create_pdf_with_table_and_charts(temp_filename, content_blocks)
            
            # 3. Add Page Numbers
            add_page_numbers(temp_filename, final_filename)
            
            # 4. Read back for download
            with open(final_filename, "rb") as f:
                pdf_data = f.read()
            
            # 5. Show Download Button
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_data,
                file_name=final_filename,
                mime="application/pdf"
            )
            
            # Cleanup
            if os.path.exists(temp_filename): os.remove(temp_filename)
            if os.path.exists(final_filename): os.remove(final_filename)
