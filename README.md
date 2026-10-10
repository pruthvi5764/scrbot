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
