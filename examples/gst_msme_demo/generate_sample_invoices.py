from __future__ import annotations

from pathlib import Path
import fitz  # PyMuPDF


def create_sample_gst_invoices(target_dir: Path) -> list[Path]:
    """Generates 10 realistic Indian GST/MSME PDF documents for demo and testing."""
    target_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    invoices = [
        {
            "filename": "gst_inv_01_tech_consulting.pdf",
            "content": """
TAX INVOICE — ORIGINAL FOR RECIPIENT
Seller: TechSutra Solutions Pvt Ltd
GSTIN: 36AABCT1234F1Z5 | State: Telangana (36)
Buyer: Acme Enterprises Ltd | GSTIN: 36AAACA9876E1Z2
Invoice No: TS/2026/042 | Date: 15-Oct-2026

Description of Services: Software Architecture & Cloud Migration
SAC Code: 998311 | Hours: 120 | Rate: ₹1,500/hr
Taxable Value: ₹1,80,000.00
CGST @ 9%: ₹16,200.00
SGST @ 9%: ₹16,200.00
Total Invoice Value: ₹2,12,400.00 (Two Lakh Twelve Thousand Four Hundred Only)
Bank: HDFC Bank | A/C: 50200012345678 | IFSC: HDFC0000123
Payment Terms: Net 30 Days. MSME Reg: UDYAM-TS-02-0045123
""",
        },
        {
            "filename": "gst_inv_02_cloud_infrastructure.pdf",
            "content": """
TAX INVOICE — INTER-STATE SUPPLY
Supplier: CloudForge Hosting India Pvt Ltd
GSTIN: 27AABCC5544D1Z8 | State: Maharashtra (27)
Place of Supply: Hyderabad, Telangana (36)
Recipient: DataFlow Labs LLP | GSTIN: 36AAGFD3322C1Z4
Invoice No: CF-2026-9921 | Date: 18-Oct-2026

Particulars: GPU Dedicated Instance (RTX 6000 Ada - 30 Days)
SAC Code: 998315
Taxable Amount: ₹85,000.00
IGST @ 18%: ₹15,300.00
Total Amount Payable: ₹1,00,300.00
Reverse Charge: No | Supply covered under GST e-Invoice Portal
""",
        },
        {
            "filename": "gst_inv_03_hardware_retail.pdf",
            "content": """
RETAIL TAX INVOICE
Store: Sri Balaji Electronics & Hardware
GSTIN: 36AABFS8877M1Z1 | Panjagutta, Hyderabad
Customer: Walk-in Retail Customer
Invoice Ref: SB/OCT/1029 | Date: 20-Oct-2026

Item 1: USB-C Multiport Adapter 100W (HSN: 8471) | Qty: 2 | ₹3,000.00 | GST 18% (₹540.00)
Item 2: NVMe SSD 2TB Gen4 (HSN: 8471) | Qty: 1 | ₹12,000.00 | GST 18% (₹2,160.00)
Total Taxable: ₹15,000.00
CGST 9%: ₹1,350.00 | SGST 9%: ₹1,350.00
Grand Total: ₹17,700.00
Payment Mode: UPI Ref # 629811204912
""",
        },
        {
            "filename": "gst_inv_04_freight_logistics.pdf",
            "content": """
CONSIGNMENT NOTE & GST FREIGHT MEMO
Carrier: Deccan Swift Logistics
GSTIN: 36AABPD4433A1ZX | Secunderabad
Consignee: Vayu Robotics Pvt Ltd | GSTIN: 36AAHCV9911K1Z9
Lorry Receipt (LR) No: DSL-40912 | Date: 21-Oct-2026

Route: Hyderabad to Bengaluru | Weight: 450 Kg
Freight Charges: ₹18,000.00
Loading / Unloading: ₹2,000.00
Total Taxable: ₹20,000.00
GST Payable under Reverse Charge (RCM): Yes (GTA Services @ 5% = ₹1,000.00)
Amount Paid by Consignor: ₹20,000.00
""",
        },
        {
            "filename": "gst_inv_05_scanned_receipt.pdf",
            "is_scanned": True,
            "content": "",  # Sparse text to test vision OCR fallback!
            "simulated_text": "CASH MEMO # 891 — Cafe Chai Shai — GSTIN: 36AAUPC1010A1Z2 — Total: ₹450.00 (Food @ 5% GST)",
        },
        {
            "filename": "gst_inv_06_msme_manufacturing.pdf",
            "content": """
COMMERCIAL TAX INVOICE (MSME ENTERPRISE)
Manufacturer: Telan Precision Castings Pvt Ltd
Udyam Reg: UDYAM-TS-09-0099881 | GSTIN: 36AAECT6677F1Z3
Customer: Bharat Heavy Components Ltd | GSTIN: 36AABCB1122D1Z0
Invoice No: TPC/26-27/088 | Date: 22-Oct-2026

Description: CNC Milled Aluminum Brackets (Grade 6061-T6)
HSN: 76169990 | Qty: 500 Pcs | Rate: ₹320/pc
Taxable Value: ₹1,60,000.00
CGST 9%: ₹14,400.00 | SGST 9%: ₹14,400.00
Total Value: ₹1,88,800.00
Declaration: We certify that this invoice shows the actual price of the goods described.
""",
        },
        {
            "filename": "gst_inv_07_subcontract_services.pdf",
            "content": """
SUBCONTRACTOR MILESTONE BILL
Agency: PixelWave Media & Design
GSTIN: 36AALCP2211E1Z8 | Madhapur, Hyderabad
Client: Global Outreach Brand Management | GSTIN: 36AAACG5566K1Z7
Invoice No: PWM-MS3-2026 | Date: 23-Oct-2026

Service: Milestone 3 UI/UX Design & Front-End Design System
SAC: 998314 | Taxable: ₹75,000.00
CGST @ 9%: ₹6,750.00 | SGST @ 9%: ₹6,750.00
Invoice Total: ₹88,500.00
Less TDS under Section 194J (2%): ₹1,500.00
Net Receivable: ₹87,000.00
""",
        },
        {
            "filename": "gst_inv_08_restaurant_hospitality.pdf",
            "content": """
BILL OF SUPPLY (COMPOSITION TAXABLE PERSON)
Establishment: Telangana Ruchulu Traditional Dining
GSTIN: 36AAKFT9922M1Z0 | Composition Scheme Dealer
Bill No: TR-5512 | Date: 23-Oct-2026

Dine-in Order # 42 | Tables 4 & 5
Food & Beverages: ₹2,450.00
Packaged Water: ₹150.00
Total Amount: ₹2,600.00
Note: Composition taxable person, not eligible to collect tax on supplies.
""",
        },
        {
            "filename": "gst_inv_09_export_sez.pdf",
            "content": """
INVOICE FOR SUPPLY TO SEZ UNIT UNDER LETTER OF UNDERTAKING (LUT)
Exporter/Supplier: CyberMatrix Global Tech Ltd
GSTIN: 36AABCC7788P1Z6 | Hitech City, Hyderabad
Recipient SEZ Unit: Virtusa SEZ Facility | GSTIN: 36AAACV1234F1Z8
LUT Ref: AD360324009121E | Invoice No: EXP-SEZ-092 | Date: 24-Oct-2026

Description: Embedded Firmware Verification Suite
SAC: 998313 | Foreign Currency: USD 2,500 @ ₹84.50 = ₹2,11,250.00
IGST @ 0% (Supply to SEZ under LUT without payment of IGST)
Total Invoice Value: ₹2,11,250.00
Declaration: Supply meant for SEZ unit for authorized operations under LUT.
""",
        },
        {
            "filename": "gst_inv_10_annual_audit_summary.pdf",
            "content": """
ANNUAL GST RECONCILIATION SUMMARY (PAGE 1 OF 2)
Auditee: KSN Technologies LLP | GSTIN: 36AAOFK4433H1Z2
Financial Year: 2025-2026 | Assessment Quarter: Q2
Generated: 24-Oct-2026

Table 1: Outward Taxable Supplies Summary
Total B2B Taxable Turnover: ₹48,50,000.00
Total CGST Paid: ₹4,36,500.00
Total SGST Paid: ₹4,36,500.00
Total IGST Paid: ₹1,20,000.00
Total Tax Liability: ₹9,93,000.00
Eligible Input Tax Credit (ITC) Available in GSTR-2B: ₹6,12,000.00
Net Cash Paid via Challan (PMT-06): ₹3,81,000.00
""",
        },
    ]

    for item in invoices:
        p = target_dir / item["filename"]
        doc = fitz.open()
        page = doc.new_page()

        if item.get("is_scanned"):
            # Insert minimal text to force vision fallback
            page.insert_text((50, 72), "SCAN_001", fontsize=8)
        else:
            page.insert_text((50, 72), item["content"].strip(), fontsize=10)

        doc.save(str(p))
        doc.close()
        generated.append(p)

    return generated


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "data"
    files = create_sample_gst_invoices(out_dir)
    print(f"Generated {len(files)} sample GST/MSME PDF documents in {out_dir}")
