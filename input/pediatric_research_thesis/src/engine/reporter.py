import csv
import io
import datetime
from fpdf import FPDF
from src.state.telemetry import TelemetryState

def generate_csv(state: TelemetryState) -> bytes:
    """Generates a CSV report of the current telemetry state."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(["Metric", "Value", "Timestamp"])
    timestamp = datetime.datetime.now().isoformat()
    
    writer.writerow(["Total Daily Adults", state.total_daily_adults, timestamp])
    writer.writerow(["Total Daily Children", state.total_daily_children, timestamp])
    writer.writerow(["Current Adults", state.current_adults, timestamp])
    writer.writerow(["Current Children", state.current_children, timestamp])
    writer.writerow(["Overcrowding Alert", str(state.overcrowding_alert), timestamp])
    
    return bytes(output.getvalue().encode('utf-8'))

def generate_pdf(state: TelemetryState) -> bytes:
    """Generates a clinical-grade PDF capacity report."""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("helvetica", "B", 24)
    pdf.set_text_color(30, 58, 138) # Indigo-900
    pdf.cell(0, 15, "Pediatric Clinical Command Center", ln=True, align="C")
    
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(71, 85, 105) # Slate-600
    pdf.cell(0, 10, "Daily Capacity & Telemetry Report", ln=True, align="C")
    pdf.ln(10)
    
    # Timestamp
    pdf.set_font("helvetica", "I", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 10, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    
    # Real-Time Stats
    pdf.set_font("helvetica", "B", 14)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "Real-Time Occupancy", ln=True)
    
    pdf.set_font("helvetica", "", 12)
    pdf.cell(100, 10, f"Current Pediatric Count: {state.current_children}")
    pdf.cell(100, 10, f"Current Adult Count: {state.current_adults}", ln=True)
    pdf.ln(5)
    
    # Daily Aggregates
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Daily Processed Volumes", ln=True)
    
    pdf.set_font("helvetica", "", 12)
    pdf.cell(100, 10, f"Total Pediatric Processed: {state.total_daily_children}")
    pdf.cell(100, 10, f"Total Adults Processed: {state.total_daily_adults}", ln=True)
    pdf.ln(10)
    
    # Overcrowding Status
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "System Alerts", ln=True)
    
    if state.overcrowding_alert:
        pdf.set_text_color(220, 38, 38) # Red-600
        pdf.cell(0, 10, "CRITICAL: Pediatric load exceeds standard waiting capacity.", ln=True)
    else:
        pdf.set_text_color(16, 185, 129) # Emerald-500
        pdf.cell(0, 10, "STATUS NOMINAL: Capacity is within standard limits.", ln=True)
        
    return bytes(pdf.output(dest='S'))
