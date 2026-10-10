"""
Unit & Integration Tests for CCTV Lead Generator & Outreach Engine
Verifies:
- India geographic bounding box & entity validation
- Indian phone number cleaning & WhatsApp classification
- Multi-indexed LeadDatabase deduplication & caching
- Dotenv parsing
- Fallback pitch generation
- Dashboard & CSV export integrity
"""

import os
import unittest
import cctv_leadgen
from cctv_leadgen import (
    is_in_india_bbox,
    is_indian_entity,
    clean_indian_phone,
    extract_clean_domain,
    extract_pincode,
    LeadDatabase,
    load_dotenv,
    get_rule_based_fallback,
    MultiLLMRotator,
    COLS,
    DEMO_URL,
    DEMO_ADMIN_URL
)


class TestCCTVLeadGen(unittest.TestCase):

    def test_india_bounding_box(self):
        # Known Indian locations (lat: 6-38, lon: 68-98)
        self.assertTrue(is_in_india_bbox(12.9716, 77.5946))   # Bengaluru
        self.assertTrue(is_in_india_bbox(28.7041, 77.1025))   # Delhi
        self.assertTrue(is_in_india_bbox(19.0760, 72.8777))   # Mumbai
        self.assertTrue(is_in_india_bbox(13.0827, 80.2707))   # Chennai

        # Foreign / US / UK locations
        self.assertFalse(is_in_india_bbox(41.8781, -87.6298)) # Chicago, IL (Negative Lon)
        self.assertFalse(is_in_india_bbox(25.7617, -80.1918)) # Miami, FL
        self.assertFalse(is_in_india_bbox(51.5074, -0.1278))  # London, UK
        self.assertFalse(is_in_india_bbox(37.7749, -122.4194))# San Francisco, CA

    def test_indian_entity_filtering(self):
        # Indian addresses
        self.assertTrue(is_indian_entity("SP Road, Bengaluru, Karnataka 560002, India", "09876543210"))
        self.assertTrue(is_indian_entity("Nehru Place, New Delhi 110019", "9811122233"))

        # Explicit non-Indian indicators
        self.assertFalse(is_indian_entity("7050 S Cicero Ave, Bedford Park, IL 60638, USA", "708-499-1234"))
        self.assertFalse(is_indian_entity("Miami, FL 33166, United States", "305-555-0199"))
        self.assertFalse(is_indian_entity("Cicero, Illinois", "+1 708-555-1234"))

        # Reject via bounding box
        self.assertFalse(is_indian_entity("Some CCTV Shop", "9876543210", lat=41.8, lon=-87.6))

    def test_clean_indian_phone(self):
        # 10-digit mobile starting with 6-9
        wa, ptype, orig = clean_indian_phone("9876543210")
        self.assertEqual(wa, "919876543210")
        self.assertEqual(ptype, "Mobile")

        # 11-digit leading 0
        wa, ptype, orig = clean_indian_phone("09876543210")
        self.assertEqual(wa, "919876543210")
        self.assertEqual(ptype, "Mobile")

        # 12-digit leading 91
        wa, ptype, orig = clean_indian_phone("919876543210")
        self.assertEqual(wa, "919876543210")
        self.assertEqual(ptype, "Mobile")

        # Formatted with spaces / dashes
        wa, ptype, orig = clean_indian_phone("+91 98765-43210")
        self.assertEqual(wa, "919876543210")
        self.assertEqual(ptype, "Mobile")

        # Landlines (STD codes e.g. 080)
        wa, ptype, orig = clean_indian_phone("080-22234567")
        self.assertEqual(wa, "")
        self.assertEqual(ptype, "Landline")

        # Foreign phone
        wa, ptype, orig = clean_indian_phone("+1 708-499-1234", address="Bedford Park, IL, USA")
        self.assertEqual(wa, "")
        self.assertEqual(ptype, "International")

    def test_pincode_extraction(self):
        self.assertEqual(extract_pincode("SP Road, Bengaluru, Karnataka 560002"), "560002")
        self.assertEqual(extract_pincode("Nehru Place, New Delhi 110019, India"), "110019")
        self.assertEqual(extract_pincode("No pincode here"), "")

    def test_clean_domain_extraction(self):
        self.assertEqual(extract_clean_domain("https://www.cctvpro.in/contact/"), "cctvpro.in")
        self.assertEqual(extract_clean_domain("http://hikvision-dealer.com"), "hikvision-dealer.com")

    def test_lead_database_multi_key_deduplication(self):
        db = LeadDatabase()
        lead1 = {
            "place_id": "ChIJ_lead_1",
            "name": "Bharat Security & CCTV Systems",
            "city": "Bengaluru",
            "phone": "9876543210",
            "website": "https://bharatcctv.com",
            "site_status": "healthy",
            "need_score": 65
        }
        db.upsert(lead1)
        self.assertEqual(len(db), 1)

        # 1. Lookup by place_id
        res = db.find(place_id="ChIJ_lead_1")
        self.assertIsNotNone(res)
        self.assertEqual(res["name"], "Bharat Security & CCTV Systems")

        # 2. Lookup by 10-digit phone
        res_phone = db.find(phone="9876543210")
        self.assertEqual(res_phone["place_id"], "ChIJ_lead_1")

        # 3. Lookup by 12-digit phone (+91)
        res_phone91 = db.find(phone="919876543210")
        self.assertEqual(res_phone91["place_id"], "ChIJ_lead_1")

        # 4. Lookup by website domain
        res_domain = db.find(website="http://www.bharatcctv.com/about")
        self.assertEqual(res_domain["place_id"], "ChIJ_lead_1")

        # 5. Lookup by normalized name + city
        res_nc = db.find(name="Bharat Security CCTV Systems", city="Bengaluru")
        self.assertEqual(res_nc["place_id"], "ChIJ_lead_1")

        # 6. Re-inserting the same business with different place_id but matching phone merges into canonical record
        lead2 = {
            "place_id": "osm_9999",
            "name": "Bharat Security CCTV",
            "city": "Bengaluru",
            "phone": "9876543210",
            "additional_phones": "080-22233344",
            "email": "contact@bharatcctv.com"
        }
        db.upsert(lead2)
        self.assertEqual(len(db), 1) # Airtight: still 1 unique company
        merged = db.find(place_id="ChIJ_lead_1")
        self.assertEqual(merged["email"], "contact@bharatcctv.com")
        self.assertEqual(merged["site_status"], "healthy") # Preserves cached audit status
        self.assertIs(db.find(place_id="osm_9999"), merged) # New alias also resolves

    def test_fallback_pitch_generation(self):
        lead = {
            "name": "Apex CCTV Solutions",
            "city": "Hyderabad",
            "segment": "no_website",
            "issues": "no website found",
            "rating": "4.8",
            "reviews": "50"
        }
        pitch = get_rule_based_fallback(lead)
        self.assertIn("whatsapp", pitch)
        self.assertIn("email_subject", pitch)
        self.assertIn("email_body", pitch)
        self.assertIn(DEMO_URL, pitch["whatsapp"])
        self.assertIn(DEMO_ADMIN_URL, pitch["whatsapp"])
        self.assertIn("Reply STOP", pitch["email_body"])

    def test_pitch_sets_outreach_status(self):
        lead = {
            "name": "Apex CCTV Solutions",
            "city": "Hyderabad",
            "segment": "no_website",
            "issues": "no website found",
            "outreach_status": "NEW",
            "whatsapp_number": "919876543210"
        }
        original_call = cctv_leadgen.llm_rotator.call_with_failover
        try:
            cctv_leadgen.llm_rotator.call_with_failover = lambda *_: {
                "whatsapp": "Hi Apex, quick demo?",
                "email_subject": "Quick CCTV Demo",
                "email_body": "Hello. Reply STOP and I won't contact you again."
            }
            pitched = cctv_leadgen.pitch(lead)
        finally:
            cctv_leadgen.llm_rotator.call_with_failover = original_call

        self.assertEqual(pitched["outreach_status"], "PITCH_DRAFTED")
        self.assertTrue(pitched["whatsapp_click_link"].startswith("https://wa.me/"))

    def test_cols_specification(self):
        self.assertIn("place_id", COLS)
        self.assertIn("outreach_status", COLS)
        self.assertIn("whatsapp_click_link", COLS)
        self.assertIn("need_score", COLS)

    def test_load_dotenv(self):
        test_env_file = "test_sample.env"
        try:
            with open(test_env_file, "w", encoding="utf-8") as f:
                f.write("# Sample test env\n")
                f.write("TEST_CCTV_KEY='test_value_123'\n")
                f.write('TEST_CCTV_CITY="Pune"\n')
            load_dotenv(test_env_file)
            self.assertEqual(os.environ.get("TEST_CCTV_KEY"), "test_value_123")
            self.assertEqual(os.environ.get("TEST_CCTV_CITY"), "Pune")
        finally:
            if os.path.exists(test_env_file):
                os.remove(test_env_file)

    def test_rotator_bounded_failover(self):
        rotator = MultiLLMRotator()
        rotator.pool = []  # Empty pool
        result = rotator.call_with_failover("sys", "user")
        self.assertIsNone(result)

    def test_clean_company_name(self):
        from cctv_leadgen import clean_company_name
        raw = "Mahadev Enterprises - Cctv installation, Camera installation, security camera repair"
        cleaned = clean_company_name(raw)
        self.assertEqual(cleaned, "Mahadev Enterprises")

    def test_classify_business_profile(self):
        from cctv_leadgen import classify_business_profile
        self.assertEqual(classify_business_profile("MY CHOICE IT HUB"), "it_hardware_hub")
        self.assertEqual(classify_business_profile("CP Plus World Store"), "brand_distributor")
        self.assertEqual(classify_business_profile("Matrix Biometrics & Access"), "access_control_security")
        self.assertEqual(classify_business_profile("Indo Spy Camera World"), "spy_surveillance")
        self.assertEqual(classify_business_profile("Bharat Security Surveillance"), "cctv_installer")

    def test_rule_based_pitch_with_sender_phone(self):
        from cctv_leadgen import get_rule_based_fallback
        lead = {
            "name": "MY CHOICE IT HUB - Computer & CCTV dealer",
            "city": "Bengaluru",
            "segment": "no_website",
            "category": "Computer store",
            "issues": "no website found",
        }
        res = get_rule_based_fallback(lead, sender_phone="+919988776655")
        self.assertIn("MY CHOICE IT HUB", res["whatsapp"])
        self.assertIn("Atrya Solutions", res["whatsapp"])
        self.assertIn("+919988776655", res["email_body"])



if __name__ == "__main__":
    unittest.main()

