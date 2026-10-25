"""
AeroLoss AI — Maintenance Report PDF Generation Service
Generates official, vector-crisp, publication-quality maintenance reports with selectable text,
KPI summary tables, aerodynamic degradation analyses, financial economics, and engineering governance disclaimers.
"""

import os
import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Color palette aligned with AeroLoss AI design system
COLOR_PRIMARY = colors.HexColor("#2563eb")     # Blue 600
COLOR_DARK = colors.HexColor("#0f172a")        # Slate 900
COLOR_MUTED = colors.HexColor("#64748b")       # Slate 500
COLOR_BG_LIGHT = colors.HexColor("#f8fafc")    # Slate 50
COLOR_BORDER = colors.HexColor("#e2e8f0")      # Slate 200
COLOR_ROSE_BG = colors.HexColor("#fef2f2")     # Rose 50
COLOR_ROSE_BORDER = colors.HexColor("#fecaca") # Rose 200
COLOR_ROSE_TEXT = colors.HexColor("#991b1b")   # Rose 800
COLOR_AMBER_BG = colors.HexColor("#fffbeb")    # Amber 50
COLOR_AMBER_BORDER = colors.HexColor("#fde68a")# Amber 200
COLOR_AMBER_TEXT = colors.HexColor("#92400e")  # Amber 800
COLOR_GREEN_BG = colors.HexColor("#f0fdf4")    # Emerald 50
COLOR_GREEN_BORDER = colors.HexColor("#bbf7d0")# Emerald 200
COLOR_GREEN_TEXT = colors.HexColor("#166534")  # Emerald 800


class ReportService:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        self.style_title = ParagraphStyle(
            'ReportTitle',
            parent=self.styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            textColor=COLOR_DARK,
            spaceAfter=2
        )
        self.style_subtitle = ParagraphStyle(
            'ReportSubtitle',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=8.5,
            leading=11,
            textColor=COLOR_MUTED,
            spaceAfter=6
        )
        self.style_section_heading = ParagraphStyle(
            'ReportSectionHeading',
            parent=self.styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=13,
            textColor=COLOR_DARK,
            spaceBefore=8,
            spaceAfter=4,
            textTransform='uppercase'
        )
        self.style_cell_label = ParagraphStyle(
            'ReportCellLabel',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=7.5,
            leading=9,
            textColor=COLOR_MUTED
        )
        self.style_cell_value = ParagraphStyle(
            'ReportCellValue',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=8.5,
            leading=11,
            textColor=COLOR_DARK
        )
        self.style_cell_bold = ParagraphStyle(
            'ReportCellBold',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8.5,
            leading=11,
            textColor=COLOR_DARK
        )
        self.style_disclaimer = ParagraphStyle(
            'ReportDisclaimer',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=6.8,
            leading=9.2,
            textColor=COLOR_MUTED
        )
        self.style_banner_title = ParagraphStyle(
            'ReportBannerTitle',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9.5,
            leading=12,
            textColor=COLOR_DARK
        )
        self.style_banner_body = ParagraphStyle(
            'ReportBannerBody',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=7.8,
            leading=10.5,
            textColor=COLOR_DARK
        )

    def generate_pdf(self, analysis_data: dict, turbine_id: str = None) -> bytes:
        """
        Builds an official PDF maintenance report buffer from analysis data.
        Returns bytes of the compiled PDF.
        """
        t_id = turbine_id or analysis_data.get('turbine_id', 'WTG-04B')
        insp_id = analysis_data.get('inspection_id', f"INSP-{t_id}-{int(datetime.now().timestamp())}")
        
        det = analysis_data.get('visual_detection', {})
        char = analysis_data.get('characterization', {})
        phys = analysis_data.get('physics_aerodynamics', {})
        fin = analysis_data.get('financial_impact', {})
        dec = analysis_data.get('maintenance_decision', {})

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=32,
            bottomMargin=32
        )
        story = []

        # 1. Header with title, subtitle, and report metadata
        header_table_data = [
            [
                Paragraph("<b>AEROLOSS AI — OFFICIAL MAINTENANCE REPORT</b>", self.style_title),
                Paragraph(f"<b>REPORT ID:</b> {insp_id}<br/><b>DATE:</b> {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}", self.style_cell_label)
            ],
            [
                Paragraph("Physics-Informed Defect & Financial Decision Engine • Offshore Cluster A", self.style_subtitle),
                Paragraph("<b>STATUS:</b> OFFICIAL DECISION SUPPORT", self.style_cell_label)
            ]
        ]
        header_table = Table(header_table_data, colWidths=[5.3 * inch, 2.2 * inch])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(header_table)
        story.append(HRFlowable(width="100%", thickness=2, color=COLOR_PRIMARY, spaceBefore=4, spaceAfter=8))

        # 2. Executive Metadata Box (Turbine & Inspection summary)
        det_class = str(det.get('detected_class', char.get('primary_defect', 'UNKNOWN'))).replace('_', ' ').upper()
        conf_pct = det.get('confidence_pct', f"{det.get('confidence', 0.90)*100:.1f}%")
        r_R_val = char.get('spanwise_position_r_R', 0.50)
        region_str = char.get('blade_region', 'Mid-Span')
        sev_proxy = char.get('severity_proxy', 0)
        area_proxy = char.get('blade_damage_area_proxy_pct', char.get('damage_area_pct', 0.0))
        frame_cov = char.get('union_frame_coverage_pct', char.get('image_frame_coverage_pct', 0.0))
        distinct_regions = char.get('distinct_damage_regions', 1)
        candidate_boxes = char.get('candidate_detections_count', det.get('defect_count', 1))

        meta_data = [
            [
                Paragraph("<b>TURBINE UNIT ID:</b>", self.style_cell_label),
                Paragraph(f"<b>{t_id}</b>", self.style_cell_bold),
                Paragraph("<b>INSPECTION ID:</b>", self.style_cell_label),
                Paragraph(f"{insp_id}", self.style_cell_value)
            ],
            [
                Paragraph("<b>PRIMARY DEFECT:</b>", self.style_cell_label),
                Paragraph(f"<b>{det_class}</b>", self.style_cell_bold),
                Paragraph("<b>MODEL CONFIDENCE:</b>", self.style_cell_label),
                Paragraph(f"{conf_pct}", self.style_cell_value)
            ],
            [
                Paragraph("<b>RADIAL SPANWISE LOCATION:</b>", self.style_cell_label),
                Paragraph(f"r/R = {r_R_val:.2f} ({region_str})", self.style_cell_value),
                Paragraph("<b>SEVERITY RATING:</b>", self.style_cell_label),
                Paragraph(f"Level {sev_proxy} / 5", self.style_cell_bold)
            ],
            [
                Paragraph("<b>2D FRAME COVERAGE:</b>", self.style_cell_label),
                Paragraph(f"{frame_cov:.2f}% Frame", self.style_cell_value),
                Paragraph("<b>BLADE SURFACE PROXY:</b>", self.style_cell_label),
                Paragraph(f"{area_proxy:.2f}% Area Proxy (Capped)", self.style_cell_value)
            ],
            [
                Paragraph("<b>DAMAGE REGIONS:</b>", self.style_cell_label),
                Paragraph(f"{distinct_regions} Distinct Region(s) ({candidate_boxes} Proposals)", self.style_cell_value),
                Paragraph("<b>REFERENCE AIRFOIL:</b>", self.style_cell_label),
                Paragraph("FFA-W3-241 (Re = 1e7)", self.style_cell_value)
            ]
        ]
        meta_table = Table(meta_data, colWidths=[1.7 * inch, 2.05 * inch, 1.7 * inch, 2.05 * inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_LIGHT),
            ('BOX', (0, 0), (-1, -1), 0.75, COLOR_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 6))

        # 3. Section: Aerodynamic & Energy Impact
        story.append(Paragraph("1. Aerodynamic Degradation & Energy Yield Impact", self.style_section_heading))
        
        base_kw = phys.get('power_baseline_kw', phys.get('baseline_power_kw', 0.0))
        dam_kw = phys.get('power_damaged_kw', phys.get('damaged_power_kw', 0.0))
        loss_kw = phys.get('power_loss_kw', phys.get('estimated_power_loss_kw', 0.0))
        aep_pct = phys.get('aep_loss_pct', 0.0)
        daily_kwh = phys.get('daily_energy_loss_kwh')
        annual_kwh = phys.get('annual_energy_loss_kwh', round(daily_kwh * 365, 1) if daily_kwh is not None else None)

        daily_kwh_str = f"{daily_kwh:,.1f} kWh / day" if daily_kwh is not None else "N/A (Unlocalized / Pending NDT)"
        annual_kwh_str = f"{annual_kwh:,.1f} kWh / year" if annual_kwh is not None else "N/A (Unlocalized / Pending NDT)"
        loss_kw_str = f"-{loss_kw:,.1f} kW" if loss_kw is not None else "N/A"
        aep_pct_str = f"-{aep_pct:.2f}% AEP" if aep_pct is not None else "N/A"

        aero_data = [
            [
                Paragraph("<b>Operational Baseline Power:</b>", self.style_cell_label),
                Paragraph(f"{base_kw:,.1f} kW", self.style_cell_bold),
                Paragraph("<b>Damaged Power Output:</b>", self.style_cell_label),
                Paragraph(f"{dam_kw:,.1f} kW", self.style_cell_bold)
            ],
            [
                Paragraph("<b>Instantaneous Deficit:</b>", self.style_cell_label),
                Paragraph(loss_kw_str, self.style_cell_bold),
                Paragraph("<b>Annual Energy Production (AEP) Loss:</b>", self.style_cell_label),
                Paragraph(f"<b>{aep_pct_str}</b>", self.style_cell_bold)
            ],
            [
                Paragraph("<b>Daily Equivalent Energy Loss:</b>", self.style_cell_label),
                Paragraph(f"<b>{daily_kwh_str}</b>", self.style_cell_bold),
                Paragraph("<b>Annualized Equivalent Energy Loss:</b>", self.style_cell_label),
                Paragraph(annual_kwh_str, self.style_cell_value)
            ]
        ]
        aero_table = Table(aero_data, colWidths=[1.8 * inch, 1.95 * inch, 1.8 * inch, 1.95 * inch])
        aero_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_LIGHT),
            ('BOX', (0, 0), (-1, -1), 0.75, COLOR_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(aero_table)
        story.append(Spacer(1, 6))

        # 4. Section: Financial Revenue Impact & Repair Economics
        story.append(Paragraph("2. Financial Revenue Impact & Repair Economics", self.style_section_heading))
        
        tariff = fin.get('tariff_inr_kwh', 4.50)
        daily_inr = fin.get('daily_financial_loss_inr', fin.get('daily_loss_inr'))
        monthly_inr = fin.get('monthly_financial_loss_inr', fin.get('monthly_loss_inr', round(daily_inr * 30, 2) if daily_inr is not None else None))
        annual_inr = fin.get('annual_financial_loss_inr', fin.get('annual_loss_inr', round(daily_inr * 365, 2) if daily_inr is not None else None))
        repair_cost = fin.get('repair_cost_inr', 40000.0)
        payback_days = fin.get('payback_days', None)
        payback_str = f"{payback_days:.1f} Days" if payback_days is not None else "N/A (Baseline Zero Loss / Pending NDT)"
        daily_inr_str = f"INR {daily_inr:,.2f} / day" if daily_inr is not None else "N/A (Pending NDT)"
        monthly_inr_str = f"INR {monthly_inr:,.2f} / month" if monthly_inr is not None else "N/A"
        annual_inr_str = f"INR {annual_inr:,.2f} / year" if annual_inr is not None else "N/A"

        fin_data = [
            [
                Paragraph("<b>Electricity Tariff:</b>", self.style_cell_label),
                Paragraph(f"INR {tariff:.2f} / kWh", self.style_cell_value),
                Paragraph("<b>Repair Mobilization Cost:</b>", self.style_cell_label),
                Paragraph(f"INR {repair_cost:,.2f}", self.style_cell_value)
            ],
            [
                Paragraph("<b>Daily Revenue Deficit:</b>", self.style_cell_label),
                Paragraph(f"<b>{daily_inr_str}</b>", self.style_cell_bold),
                Paragraph("<b>Monthly Revenue Deficit:</b>", self.style_cell_label),
                Paragraph(monthly_inr_str, self.style_cell_value)
            ],
            [
                Paragraph("<b>Annualized Revenue Loss:</b>", self.style_cell_label),
                Paragraph(annual_inr_str, self.style_cell_value),
                Paragraph("<b>Simple Repair Payback:</b>", self.style_cell_label),
                Paragraph(f"<b>{payback_str}</b>", self.style_cell_bold)
            ]
        ]
        fin_table = Table(fin_data, colWidths=[1.8 * inch, 1.95 * inch, 1.8 * inch, 1.95 * inch])
        fin_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), COLOR_BG_LIGHT),
            ('BOX', (0, 0), (-1, -1), 0.75, COLOR_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(fin_table)
        story.append(Spacer(1, 6))

        # 5. Section: Maintenance Decision Support
        story.append(Paragraph("3. Maintenance Decision Support & Action Recommendation", self.style_section_heading))
        
        action = dec.get('recommended_action', 'MONITOR')
        urgency = dec.get('urgency', 'LOW')
        safety_override = dec.get('safety_override_active', False)
        reasoning_list = dec.get('reasoning', dec.get('reasons', []))
        if isinstance(reasoning_list, list):
            reasoning_text = "<br/>• ".join(reasoning_list)
            if reasoning_text:
                reasoning_text = "• " + reasoning_text
        else:
            reasoning_text = str(reasoning_list)

        if action in ['ENGINEERING_ASSESSMENT', 'REJECTED']:
            bg_color = COLOR_ROSE_BG
            border_color = COLOR_ROSE_BORDER
            text_color = COLOR_ROSE_TEXT
        elif action == 'REPAIR':
            bg_color = COLOR_AMBER_BG
            border_color = COLOR_AMBER_BORDER
            text_color = COLOR_AMBER_TEXT
        else: # MONITOR
            bg_color = COLOR_GREEN_BG
            border_color = COLOR_GREEN_BORDER
            text_color = COLOR_GREEN_TEXT

        override_badge = " [MANDATORY STRUCTURAL SAFETY OVERRIDE ACTIVE]" if safety_override else ""
        banner_title_text = f"<b>RECOMMENDED ACTION: {action} (URGENCY: {urgency}){override_badge}</b>"
        
        banner_data = [
            [Paragraph(banner_title_text, ParagraphStyle('BTitle', parent=self.style_banner_title, textColor=text_color))],
            [Paragraph(reasoning_text or "No defect anomalies detected. Continue routine inspection schedule.", self.style_banner_body)]
        ]
        banner_table = Table(banner_data, colWidths=[7.5 * inch])
        banner_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('BOX', (0, 0), (-1, -1), 1.0, border_color),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(banner_table)
        story.append(Spacer(1, 8))

        # 6. Governance & Limitations Disclaimer
        disclaimer_text = (
            "<b>Engineering Governance & Scientific Limitations:</b><br/>"
            "1. <b>Model Estimates:</b> AeroLoss AI provides engineering decision support. All aerodynamic power deficits, "
            "AEP loss percentages, and monetary losses are physics-informed model estimates derived from uncalibrated 2D optical proxies "
            "and standard FFA-W3-241 polar degradation curves.<br/>"
            "2. <b>Geometric Calibration:</b> 2D photographic bounding-box coverage does not directly equal true 3D damaged blade area. "
            "Defect areas are capped at 5.0% for aerodynamic solver stability until photogrammetric laser scale calibration is available.<br/>"
            "3. <b>SCADA Attribution:</b> Observed empirical power deficits include fleet operational factors (wake turbulence, yaw misalignment, "
            "control margins) and cannot be solely attributed to blade surface damage without paired time-synchronized inspection telemetry.<br/>"
            "4. <b>Safety First:</b> Structural safety overrides strictly supersede financial payback economics. Any potential structural defect "
            "mandates independent qualified rope-access or engineering NDT inspection prior to maintenance deferral."
        )
        story.append(Paragraph(disclaimer_text, self.style_disclaimer))

        # Build PDF
        doc.build(story)
        return buf.getvalue()

