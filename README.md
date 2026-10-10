# CCTV Lead Generation & Outreach Engine

Automated, production-grade lead generation engine for CCTV and electronic security companies across India.

---

##  Running in the Cloud (Automated with Laptop Off)

This repository includes a preconfigured **GitHub Actions Workflow** that runs entirely on GitHub's cloud servers on a daily schedule or on-demand.

### 1. Push to a Private GitHub Repository
Run these commands in your project folder:

```bash
git init
git add .
git commit -m "Initial commit: CCTV Lead Generator"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```



---

### 2. Add Secrets to GitHub (Recommended)
Go to your GitHub repo in your browser:
1. Navigate to: **Settings** -> **Secrets and variables** -> **Actions**
2. Click **New repository secret** and add:
   - `GOOGLE_API_KEY` (Your Google Places API key)
   - `GEMINI_KEYS` (Your Gemini API key)
   - `GROQ_KEYS` (Your Groq API key)
   - `OPENAI_KEYS` (Your OpenAI API key - optional)

---

### 3. Run the Bot from Anywhere (PC, Phone, Tablet)
1. Go to the **Actions** tab on your GitHub repository.
2. Click on **CCTV Lead Generation & Outreach Engine** on the left.
3. Click **Run workflow**:
   - Set **Target number of leads** (e.g., `50` or `100`).
   - Set **Pitch count** (e.g., `25`).
   - (Optional) Enter specific cities (e.g., `Bengaluru,Mumbai`) or leave blank for all.
4. Click **Run workflow** and close your browser/laptop!

---

### 4. Downloading the Leads
- As soon as the run finishes, GitHub automatically commits the updated `leads.csv`, `leads.xlsx`, `leads_dashboard.html`, and `company_details.json` directly into your repository.
- You can also download them as a ZIP file under **Artifacts** on the workflow run page.

---

### 5. Custom Automation Schedule
The default GitHub Actions schedule runs daily at **9:30 AM IST**:

```yaml
schedule:
  - cron: "0 4 * * *"
```

For hourly runs, change it to:

```yaml
schedule:
  - cron: "0 * * * *"
```

For every 6 hours:

```yaml
schedule:
  - cron: "0 */6 * * *"
```

---

## 🚀 Automated Outreach Delivery (WhatsApp & Email)

The pipeline generates **dedicated, hardcoded, high-converting pitches** (no external AI required) that automatically slot in the clean company name and match the business profile:
- **CCTV Installers**: High-margin commercial lead capture + 48h website & AMC panel.
- **IT & Hardware Hubs**: B2B quote catalog for computers & CCTV without directory commissions.
- **Brand Showrooms & Wholesalers**: Digital dealer product catalog with direct WhatsApp quote buttons.
- **Biometric & Access Control**: Corporate lead forms for attendance & security systems.
- **Spy & Surveillance**: Dedicated gadget & covert camera quote landing page.

### 1. Delivery via Interactive Dashboard (`leads_dashboard.html`)
- **1-Click WhatsApp**: Click `💬 WhatsApp` on any row to open WhatsApp Web/App pre-filled with the tailored message.
- **1-Click Email**: Click `✉️ Mail` to launch your default mail client with pre-filled subject and body.
- **Auto-Dispatch Queue**: Click **⚡ Auto-Dispatch Queue** in the top header to queue and send messages consecutively with an automatic delay.

### 2. Delivery via Command Line (`dispatch_outreach.py`)
Run the automated delivery script directly from your terminal:

```bash
# Preview outreach without sending (Dry Run)
python dispatch_outreach.py --channel whatsapp --limit 10 --dry-run

# Sequentially open WhatsApp tabs for top 25 prospects
python dispatch_outreach.py --channel whatsapp --limit 25 --delay 5

# Send cold emails via SMTP (requires SMTP_USER & SMTP_PASSWORD in .env)
python dispatch_outreach.py --channel email --limit 20
```

### 3. Configuring Your Number
In your `.env` file, add:
```ini
MY_PHONE_NUMBER=+91XXXXXXXXXX
SENDER_NAME="Atrya Solutions"
```
Or pass it directly:
```bash
python cctv_leadgen.py --target 50 --pitch-top 25 --my-number "+919876543210" --force-pitch
```

