#!/usr/bin/env python3
"""
Automated Outreach Delivery Engine for CCTV & Security Leads
Delivers tailored WhatsApp messages and Cold Emails directly to verified Indian prospects.

Usage:
  python dispatch_outreach.py --channel whatsapp --limit 10
  python dispatch_outreach.py --channel email --limit 10
  python dispatch_outreach.py --channel both --limit 5
  python dispatch_outreach.py --dry-run
"""

import argparse
import csv
import json
import os
import smtplib
import sys
import time
import urllib.parse
import webbrowser
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Load environment variables if .env exists
ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(ENV_FILE):
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

MY_PHONE_NUMBER = os.environ.get("MY_PHONE_NUMBER", "").strip()
SENDER_NAME = os.environ.get("SENDER_NAME", "Atrya Solutions").strip()
SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "").strip()


def load_leads(csv_file="leads.csv"):
    if not os.path.exists(csv_file):
        print(f"[!] Error: {csv_file} not found. Run cctv_leadgen.py first.")
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
        return False, "SMTP credentials missing in .env (SMTP_HOST, SMTP_USER, SMTP_PASSWORD)"

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
        return True, "Email sent successfully"
    except Exception as ex:
        return False, f"SMTP Error: {str(ex)}"


def dispatch_whatsapp(lead, delay=3.5, dry_run=False):
    wa_link = lead.get("whatsapp_click_link", "")
    phone = lead.get("whatsapp_number") or lead.get("phone", "")
    name = lead.get("name", "Prospect")

    if not wa_link and phone:
        msg = lead.get("whatsapp_pitch", "")
        if msg:
            clean_p = "".join(filter(str.isdigit, phone))
            if len(clean_p) == 10:
                clean_p = "91" + clean_p
            wa_link = f"https://wa.me/{clean_p}?text={urllib.parse.quote(msg)}"

    if not wa_link:
        return False, "No valid WhatsApp link or mobile number"

    if dry_run:
        print(f"    [DRY-RUN WA] -> {name} ({phone})")
        print(f"      Link: {wa_link[:75]}...")
        return True, "Dry-run previewed"

    print(f"    [SENDING WA] -> {name} ({phone})...")
    webbrowser.open(wa_link)
    time.sleep(delay)
    return True, "WhatsApp chat opened in browser"


def dispatch_email(lead, dry_run=False):
    raw_email = lead.get("email", "")
    if not raw_email:
        return False, "No email address found"

    first_email = raw_email.split(";")[0].strip()
    if not first_email or "@" not in first_email:
        return False, f"Invalid email format ({raw_email})"

    subject = lead.get("email_subject") or "CCTV Growth & Demo Walkthrough"
    body = lead.get("email_body") or ""
    if not body:
        return False, "No email body pitch drafted"

    if dry_run:
        print(f"    [DRY-RUN MAIL] -> {first_email}")
        print(f"      Subject: {subject}")
        return True, "Dry-run previewed"

    if SMTP_HOST and SMTP_USER and SMTP_PASSWORD:
        success, msg = send_email_smtp(first_email, subject, body)
        return success, msg
    else:
        # Fallback to mailto link
        mailto_url = f"mailto:{urllib.parse.quote(first_email)}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
        webbrowser.open(mailto_url)
        return True, "Opened in default email client (configure SMTP in .env for background delivery)"


def main():
    parser = argparse.ArgumentParser(description="Automated Outreach Delivery Engine for CCTV Leads")
    parser.add_argument("--channel", choices=["whatsapp", "email", "both"], default="whatsapp", help="Delivery channel (default: whatsapp)")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of leads to contact in this run (default: 10)")
    parser.add_argument("--delay", type=float, default=3.5, help="Delay in seconds between browser tabs (default: 3.5)")
    parser.add_argument("--dry-run", action="store_true", help="Preview outreach queue without sending")
    parser.add_argument("--my-number", type=str, default="", help="Your WhatsApp sender phone number")
    parser.add_argument("--input", type=str, default="leads.csv", help="Input CSV file (default: leads.csv)")
    args = parser.parse_args()

    sender_phone = args.my_number or MY_PHONE_NUMBER
    print("=" * 80)
    print(" CCTV LEAD OUTREACH DISPATCH ENGINE")
    print(f" Channel:        {args.channel.upper()}")
    print(f" Limit:          {args.limit} prospects")
    print(f" Sender Phone:   {sender_phone or 'Default (Atrya Solutions)'}")
    print(f" Mode:           {'DRY-RUN (Preview Only)' if args.dry_run else 'LIVE DELIVERY'}")
    print("=" * 80)

    leads = load_leads(args.input)
    dispatched_count = 0

    # Filter candidates
    candidates = []
    for l in leads:
        has_wa = bool(l.get("whatsapp_click_link") or l.get("whatsapp_number"))
        has_em = bool(l.get("email") and "@" in l.get("email", ""))
        if args.channel == "whatsapp" and has_wa:
            candidates.append(l)
        elif args.channel == "email" and has_em:
            candidates.append(l)
        elif args.channel == "both" and (has_wa or has_em):
            candidates.append(l)

    target_leads = candidates[:args.limit]
    print(f"[*] Found {len(candidates)} eligible prospects. Processing top {len(target_leads)}...\n")

    for idx, l in enumerate(target_leads, 1):
        name = l.get("name", "Unknown")
        print(f"[{idx}/{len(target_leads)}] Processing: {name}")

        wa_ok = False
        if args.channel in ("whatsapp", "both"):
            wa_ok, wa_msg = dispatch_whatsapp(l, delay=args.delay, dry_run=args.dry_run)
            print(f"      WhatsApp: {wa_msg}")

        em_ok = False
        if args.channel in ("email", "both"):
            em_ok, em_msg = dispatch_email(l, dry_run=args.dry_run)
            print(f"      Email:    {em_msg}")

        if not args.dry_run and (wa_ok or em_ok):
            l["outreach_status"] = "OUTREACH_SENT"
            dispatched_count += 1

    if not args.dry_run and dispatched_count > 0:
        save_leads(leads, args.input)
        print(f"\n[+] Successfully dispatched and updated {dispatched_count} leads in {args.input}!")
    elif args.dry_run:
        print(f"\n[+] Dry-run complete. {len(target_leads)} prospects previewed.")

    print("=" * 80)


if __name__ == "__main__":
    main()
