#!/usr/bin/env python3
"""
Automated Outreach Delivery Engine for CCTV & Security Leads
Delivers tailored WhatsApp messages and Cold Emails directly to verified Indian prospects.
Supports:
1. Zero-Click Local Desktop Auto-Send (Auto-presses Enter via PyAutoGUI, closes tab)
2. Direct REST API Gateways (Green-API, UltraMsg, Twilio, Webhook) for 100% headless / GitHub Actions dispatch
3. SMTP Background Email Delivery

Usage:
  python dispatch_outreach.py --channel whatsapp --limit 10
  python dispatch_outreach.py --channel email --limit 10
  python dispatch_outreach.py --channel both --limit 10
  python dispatch_outreach.py --dry-run
"""

import argparse
import csv
import json
import logging
import os
import smtplib
import sys
import time
import urllib.parse
import webbrowser
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests

try:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OutreachDispatch")

# Load environment variables from .env
ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(ENV_FILE):
    try:
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v
    except Exception as ex:
        logger.warning(f"Error reading .env: {ex}")

MY_PHONE_NUMBER = os.environ.get("MY_PHONE_NUMBER", "+919353489736").strip()
SENDER_NAME = os.environ.get("SENDER_NAME", "Atrya Solutions").strip()

# WhatsApp Direct REST Gateway Credentials (optional for headless / cloud dispatch)
WHATSAPP_GATEWAY_URL = os.environ.get("WHATSAPP_GATEWAY_URL", "").strip()
GREEN_API_INSTANCE_ID = os.environ.get("GREEN_API_INSTANCE_ID", "").strip()
GREEN_API_TOKEN = os.environ.get("GREEN_API_TOKEN", "").strip()
ULTRAMSG_INSTANCE_ID = os.environ.get("ULTRAMSG_INSTANCE_ID", "").strip()
ULTRAMSG_TOKEN = os.environ.get("ULTRAMSG_TOKEN", "").strip()
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_WHATSAPP_FROM = os.environ.get("TWILIO_WHATSAPP_FROM", "").strip()

# SMTP Credentials
SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "").strip()


def clean_pitch_text(text):
    """Decodes URL-encoded pitches and unescapes legacy literal backslashes into clean newlines."""
    if not text:
        return ""
    s = str(text)
    if "%20" in s or "%0A" in s:
        try:
            s = urllib.parse.unquote(s)
        except Exception:
            pass
    s = s.replace(" \\n ", "\n").replace("\\n", "\n")
    return s.strip()


def load_leads(csv_file="leads.csv"):
    if not os.path.exists(csv_file):
        logger.error(f"{csv_file} not found. Run cctv_leadgen.py first.")
        sys.exit(1)
    leads = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            leads.append(dict(r))
    return leads


def save_leads(leads, csv_file="leads.csv"):
    if not leads:
        return
    fieldnames = list(leads[0].keys())
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(leads)


def send_email_smtp(to_email, subject, body):
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        return False, "SMTP credentials missing (SMTP_HOST, SMTP_USER, SMTP_PASSWORD)"

    try:
        msg = MIMEMultipart()
        msg["From"] = f"{SENDER_NAME} <{SMTP_USER}>"
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        if SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15)
            server.starttls()

        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True, "Email sent via SMTP"
    except Exception as ex:
        return False, f"SMTP Error: {str(ex)}"


def send_whatsapp_via_api(clean_phone, message):
    """
    Sends WhatsApp message via configured HTTP REST API Gateway.
    Zero browser interaction required — 100% automated & headless.
    """
    # 1. Green-API
    if GREEN_API_INSTANCE_ID and GREEN_API_TOKEN:
        url = f"https://api.green-api.com/waInstance{GREEN_API_INSTANCE_ID}/sendMessage/{GREEN_API_TOKEN}"
        chat_id = f"{clean_phone}@c.us"
        try:
            r = requests.post(url, json={"chatId": chat_id, "message": message}, timeout=15)
            if r.status_code == 200:
                return True, "Delivered via Green-API"
            return False, f"Green-API HTTP {r.status_code}: {r.text[:80]}"
        except Exception as ex:
            return False, f"Green-API Error: {ex}"

    # 2. UltraMsg
    if ULTRAMSG_INSTANCE_ID and ULTRAMSG_TOKEN:
        url = f"https://api.ultramsg.com/{ULTRAMSG_INSTANCE_ID}/messages/chat"
        payload = {"token": ULTRAMSG_TOKEN, "to": clean_phone, "body": message}
        try:
            r = requests.post(url, data=payload, timeout=15)
            if r.status_code == 200:
                return True, "Delivered via UltraMsg"
            return False, f"UltraMsg HTTP {r.status_code}: {r.text[:80]}"
        except Exception as ex:
            return False, f"UltraMsg Error: {ex}"

    # 3. Custom Gateway / Webhook
    if WHATSAPP_GATEWAY_URL:
        payload = {"phone": clean_phone, "to": clean_phone, "message": message}
        try:
            r = requests.post(WHATSAPP_GATEWAY_URL, json=payload, timeout=15)
            if r.status_code in (200, 201, 202):
                return True, "Delivered via Custom Gateway"
            return False, f"Gateway HTTP {r.status_code}: {r.text[:80]}"
        except Exception as ex:
            return False, f"Gateway Error: {ex}"

    # 4. Twilio WhatsApp
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_WHATSAPP_FROM:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
        data = {
            "From": TWILIO_WHATSAPP_FROM,
            "To": f"whatsapp:+{clean_phone}",
            "Body": message
        }
        try:
            r = requests.post(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=15)
            if r.status_code in (200, 201):
                return True, "Delivered via Twilio"
            return False, f"Twilio HTTP {r.status_code}: {r.text[:80]}"
        except Exception as ex:
            return False, f"Twilio Error: {ex}"

    return None, "No WhatsApp API Gateway configured"


def dispatch_whatsapp(lead, auto_send=True, wait_time=12, dry_run=False):
    """
    Delivers pitch message to prospect via WhatsApp automatically without requiring user click.
    """
    raw_phone = lead.get("whatsapp_number") or lead.get("phone", "")
    clean_p = "".join(filter(str.isdigit, raw_phone))
    if len(clean_p) == 10:
        clean_p = "91" + clean_p
    elif len(clean_p) == 12 and clean_p.startswith("91"):
        pass
    else:
        return False, "Not an eligible Indian mobile number"

    msg = clean_pitch_text(lead.get("whatsapp_pitch", ""))
    if not msg:
        return False, "No pitch message drafted"

    name = lead.get("name", "Prospect")

    if dry_run:
        print(f"    [DRY-RUN WA] -> {name} (+{clean_p})")
        print(f"      Message Preview:\n        {msg.replace(chr(10), chr(10) + '        ')}")
        return True, "Dry-run previewed"

    # Step A: Check if a direct REST API gateway is configured (100% headless, no GUI needed)
    api_success, api_reason = send_whatsapp_via_api(clean_p, msg)
    if api_success is True:
        print(f"    [SENT WA API] -> {name} (+{clean_p}) [{api_reason}]")
        return True, api_reason
    elif api_success is False and "Gateway Error" not in api_reason:
        # Gateway was configured but returned an error
        print(f"    [!] Gateway error for {name}: {api_reason}")

    # Step B: Zero-Click Local Desktop Auto-Send (Auto-press Enter via PyAutoGUI)
    if auto_send and HAS_PYAUTOGUI:
        print(f"    [AUTO-SEND WA] -> {name} (+{clean_p}). Opening chat & sending automatically...")
        wa_url = f"https://web.whatsapp.com/send?phone={clean_p}&text={urllib.parse.quote(msg)}"
        webbrowser.open(wa_url)
        # Wait for WhatsApp Web to load chat and pre-fill message box
        time.sleep(wait_time)
        try:
            # Simulate pressing ENTER to send message automatically
            pyautogui.press("enter")
            time.sleep(2)
            # Close browser tab automatically
            pyautogui.hotkey("ctrl", "w")
            return True, "Message sent automatically via PyAutoGUI"
        except Exception as ex:
            return False, f"PyAutoGUI error: {ex}"

    # Step C: Fallback to opening wa.me URL
    wa_url = f"https://wa.me/{clean_p}?text={urllib.parse.quote(msg)}"
    webbrowser.open(wa_url)
    return True, "Chat opened in browser (Press Enter to send)"


def dispatch_email(lead, dry_run=False):
    raw_email = lead.get("email", "")
    if not raw_email:
        return False, "No email address found"

    first_email = raw_email.split(";")[0].strip()
    if not first_email or "@" not in first_email:
        return False, f"Invalid email format ({raw_email})"

    subject = clean_pitch_text(lead.get("email_subject") or "CCTV Growth & Demo Walkthrough")
    body = clean_pitch_text(lead.get("email_body") or "")
    if not body:
        return False, "No email body pitch drafted"

    name = lead.get("name", "Prospect")

    if dry_run:
        print(f"    [DRY-RUN MAIL] -> {name} <{first_email}>")
        print(f"      Subject: {subject}")
        return True, "Dry-run previewed"

    if SMTP_HOST and SMTP_USER and SMTP_PASSWORD:
        success, msg = send_email_smtp(first_email, subject, body)
        return success, msg
    else:
        mailto_url = f"mailto:{urllib.parse.quote(first_email)}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
        webbrowser.open(mailto_url)
        return True, "Opened in default email client (configure SMTP in .env for background delivery)"


def main():
    parser = argparse.ArgumentParser(description="Automated Outreach Delivery Engine for CCTV Leads")
    parser.add_argument("--channel", choices=["whatsapp", "email", "both"], default="whatsapp", help="Delivery channel (default: whatsapp)")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of leads to contact in this run (default: 10)")
    parser.add_argument("--wait", type=int, default=12, help="Wait time in seconds for WhatsApp Web to load before auto-pressing Enter (default: 12)")
    parser.add_argument("--no-auto-send", action="store_true", help="Disable automatic Enter keypress in desktop mode")
    parser.add_argument("--dry-run", action="store_true", help="Preview outreach queue without sending")
    parser.add_argument("--my-number", type=str, default="", help="Your WhatsApp sender phone number")
    parser.add_argument("--input", type=str, default="leads.csv", help="Input CSV file (default: leads.csv)")
    args = parser.parse_args()

    sender_phone = args.my_number or MY_PHONE_NUMBER
    auto_send = not args.no_auto_send

    print("=" * 80)
    print(" CCTV LEAD OUTREACH DISPATCH ENGINE (ZERO-CLICK AUTOMATION)")
    print(f" Channel:        {args.channel.upper()}")
    print(f" Limit:          {args.limit} prospects")
    print(f" Sender Phone:   {sender_phone or 'Default (Atrya Solutions)'}")
    print(f" Auto-Send:      {'YES (Auto-press Enter & close tab)' if auto_send else 'MANUAL CLICK'}")
    print(f" Mode:           {'DRY-RUN (Preview Only)' if args.dry_run else 'LIVE DISPATCH'}")
    print("=" * 80)

    leads = load_leads(args.input)

    # Filter eligible prospects
    if args.channel == "whatsapp":
        eligible = [l for l in leads if l.get("whatsapp_pitch") and (l.get("whatsapp_number") or l.get("phone_type") == "Mobile")]
    elif args.channel == "email":
        eligible = [l for l in leads if l.get("email") and l.get("email_body")]
    else:
        eligible = [l for l in leads if (l.get("whatsapp_pitch") and (l.get("whatsapp_number") or l.get("phone_type") == "Mobile")) or (l.get("email") and l.get("email_body"))]

    print(f"[*] Found {len(eligible)} eligible prospects. Processing top {min(len(eligible), args.limit)}...\n")

    dispatched_count = 0
    to_process = eligible[:args.limit]

    for idx, lead in enumerate(to_process, 1):
        name = lead.get("name", f"Lead #{idx}")
        print(f"[{idx}/{len(to_process)}] Processing: {name}")

        wa_ok = False
        email_ok = False

        if args.channel in ("whatsapp", "both") and (lead.get("whatsapp_number") or lead.get("phone_type") == "Mobile"):
            wa_ok, wa_msg = dispatch_whatsapp(lead, auto_send=auto_send, wait_time=args.wait, dry_run=args.dry_run)
            print(f"      WhatsApp: {wa_msg}")

        if args.channel in ("email", "both") and lead.get("email"):
            email_ok, email_msg = dispatch_email(lead, dry_run=args.dry_run)
            print(f"      Email:    {email_msg}")

        if not args.dry_run and (wa_ok or email_ok):
            if wa_ok and email_ok:
                lead["outreach_status"] = "CONTACTED_BOTH"
            elif wa_ok:
                lead["outreach_status"] = "CONTACTED_WHATSAPP"
            elif email_ok:
                lead["outreach_status"] = "CONTACTED_EMAIL"
            dispatched_count += 1

    if not args.dry_run:
        save_leads(leads, args.input)
        print(f"\n[+] Successfully dispatched to {dispatched_count} prospects. Updated {args.input}.")
    else:
        print(f"\n[+] Dry-run complete. {len(to_process)} prospects previewed.")

    print("=" * 80)


if __name__ == "__main__":
    main()
