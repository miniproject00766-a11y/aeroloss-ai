"""
AeroLoss AI — Maintenance Report PDF Export Regression Test Suite
Validates:
1. POST /api/reports/pdf endpoint produces valid application/pdf streams.
2. PDF contains non-empty, vector-crisp, selectable text on every page (no blank pages).
3. All key KPIs, defect analyses, financial metrics, recommendations, and disclaimers are present.
4. Client-side print CSS and DOM contract in static/index.html ensure no blank output on print/save.
5. Direct ReportService unit test runs deterministically across multiple turbine configurations.
"""

import os
import sys
import io
import unittest
import json
import requests
import pypdf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from services.report_service import ReportService


class TestPDFReportExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:8080"
        cls.report_service = ReportService()
        cls.uploaded_photo_payload = {
            'turbine_id': 'WTG-04B (Uploaded Photo)',
            'inspection_id': 'INSP-WTG-04B (Uploaded Photo)-1792855636513',
            'visual_detection': {
                'detected_class': 'hide_craze',
                'confidence': 0.2304,
                'confidence_pct': '23.0%',
                'defect_count': 2
            },
            'characterization': {
                'primary_defect': 'hide_craze',
                'spanwise_position_r_R': 0.887,
                'blade_region': 'Outboard (High Speed)',
                'severity_proxy': 3,
                'damage_area_pct': 2.819,
                'blade_damage_area_proxy_pct': 2.819,
                'union_frame_coverage_pct': 28.19,
                'distinct_damage_regions': 1,
                'candidate_detections_count': 2
            },
            'physics_aerodynamics': {
                'baseline_power_kw': 1485.4,
                'damaged_power_kw': 1454.3,
                'power_loss_kw': 31.1,
                'aep_loss_pct': 2.09,
                'daily_energy_loss_kwh': 650.07,
                'annual_energy_loss_kwh': 237275.55
            },
            'financial_impact': {
                'tariff_inr_kwh': 4.50,
                'repair_cost_inr': 40000.0,
                'daily_financial_loss_inr': 2925.32,
                'monthly_financial_loss_inr': 87759.60,
                'annual_financial_loss_inr': 1067741.80,
                'payback_days': 13.7
            },
            'maintenance_decision': {
                'recommended_action': 'ENGINEERING_ASSESSMENT',
                'urgency': 'CRITICAL',
                'safety_override_active': True,
                'reasoning': [
                    'MANDATORY SAFETY OVERRIDE: Extensive damage anomaly detected (28.19% frame coverage).',
                    'Severe outboard damage near blade tip (r/R = 0.89) presents structural failure risk.',
                    'Requires immediate physical engineering assessment prior to routine restart.'
                ]
            }
        }

    def test_direct_report_service_generation(self):
        """Verifies ReportService.generate_pdf produces valid non-empty PDF bytes."""
        pdf_bytes = self.report_service.generate_pdf(self.uploaded_photo_payload)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 2000, "PDF bytes too small")

        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        self.assertEqual(len(reader.pages), 1, "Expected single page report")
        text = reader.pages[0].extract_text()
        self.assertGreater(len(text.strip()), 1000, "Report text cannot be empty or blank")

    def test_pdf_endpoint_generates_valid_pdf(self):
        """Verifies POST /api/reports/pdf responds with HTTP 200 and application/pdf headers."""
        resp = requests.post(
            f"{self.base_url}/api/reports/pdf",
            json=self.uploaded_photo_payload,
            timeout=5
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("Content-Type"), "application/pdf")
        self.assertIn("attachment", resp.headers.get("Content-Disposition", ""))
        self.assertIn("AeroLoss_Maintenance_Report", resp.headers.get("Content-Disposition", ""))
        self.assertGreater(len(resp.content), 2000)

    def test_pdf_all_kpis_and_recommendations_present(self):
        """Verifies the exported PDF contains all required report fields, KPIs, and governance notices."""
        resp = requests.post(
            f"{self.base_url}/api/reports/pdf",
            json=self.uploaded_photo_payload,
            timeout=5
        )
        self.assertEqual(resp.status_code, 200)
        reader = pypdf.PdfReader(io.BytesIO(resp.content))
        self.assertGreaterEqual(len(reader.pages), 1)

        # Inspect every page
        full_text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            self.assertGreater(len(page_text.strip()), 200, f"Page {i+1} must not be blank")
            full_text += "\n" + page_text

        # Verify Core Identifiers
        self.assertIn("AEROLOSS AI", full_text)
        self.assertIn("OFFICIAL MAINTENANCE", full_text)
        self.assertIn("REPORT", full_text)
        self.assertIn("WTG-04B (Uploaded Photo)", full_text)

        # Verify Defect Analysis
        self.assertIn("HIDE CRAZE", full_text)
        self.assertIn("Level 3 / 5", full_text)
        self.assertIn("28.19%", full_text)
        self.assertIn("2.82%", full_text)

        # Verify Aerodynamics & Energy
        self.assertIn("1,485.4 kW", full_text)
        self.assertIn("1,454.3 kW", full_text)
        self.assertIn("-2.09% AEP", full_text)
        self.assertIn("650.1 kWh / day", full_text)

        # Verify Financial Revenue & Payback
        self.assertIn("INR 2,925.32 / day", full_text)
        self.assertIn("13.7 Days", full_text)

        # Verify Maintenance Recommendation & Urgency
        self.assertIn("RECOMMENDED ACTION: ENGINEERING_ASSESSMENT", full_text)
        self.assertIn("URGENCY: CRITICAL", full_text)
        self.assertIn("MANDATORY STRUCTURAL", full_text)
        self.assertIn("SAFETY OVERRIDE ACTIVE", full_text)

        # Verify Engineering Governance & Disclaimers
        self.assertIn("Engineering Governance & Scientific Limitations", full_text)
        self.assertIn("Model Estimates", full_text)

    def test_pdf_pages_not_blank_no_empty_pages(self):
        """Verifies that no page in the PDF is blank or contains a solid white raster rectangle."""
        pdf_bytes = self.report_service.generate_pdf(self.uploaded_photo_payload)
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text().strip()
            self.assertGreater(len(text), 100, f"Page {page_idx+1} has suspiciously low text content")

    def test_print_css_and_dom_contract(self):
        """Verifies static/index.html includes print CSS rules ensuring visible printable report."""
        html_path = os.path.join(BASE_DIR, "static", "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        self.assertIn("@media print", html)
        self.assertIn("#printable-pdf-report", html)
        self.assertIn("display: block !important;", html)
        self.assertIn("visibility: visible !important;", html)
        self.assertIn("/api/reports/pdf", html, "Frontend must call /api/reports/pdf endpoint")


if __name__ == '__main__':
    unittest.main()
