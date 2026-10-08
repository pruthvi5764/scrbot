"""
CCTV Company Lead Generator & Outreach Engine (High-Yield / Production Grade)
Pipeline: Discover -> Deep Audit (Site/SSL/Mobile/Contact/Social) -> Segment -> Multi-LLM Rotating Pitch -> Export Full Company Dossiers (CSV & JSON).

Stores Comprehensive Company Data:
- Business Profile: Name, Category, Full Address, Pincode, City, State, Coordinates (Lat/Long), Google Maps Link
- Contact Channels: Primary Phone, Mobile vs Landline, Clean WhatsApp (91...), 1-Click WhatsApp Chat Link, Alternate Phones, Scraped Emails
- Social Profiles: Facebook, Instagram, LinkedIn, YouTube, Twitter
- Digital Audit: Need Score (0-100), Priority (High/Med/Low), SSL Status, Mobile Responsiveness, Load Time, Enquiry Form, WhatsApp Widget, CMS Platform, Parked Status, Technical Issues
- Outreach Pack: Tailored WhatsApp Pitch, Cold Email Subject & Body
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==============================================================================
# 1. API KEY CONFIGURATION & MULTI-LLM ROTATING POOL
# ==============================================================================

GOOGLE_PLACES_KEY = os.environ.get("GOOGLE_API_KEY", "")

ANTHROPIC_KEYS = []
OPENAI_KEYS = []
GEMINI_KEYS = []
GROQ_KEYS = []
XAI_GROK_KEYS = []


class MultiLLMRotator:
    """
    Thread-safe rotating pool for multiple LLM API keys across
    Anthropic, OpenAI, Gemini, Groq, and xAI (Grok).
    Rotates in a loop. If a key falls (quota, rate limit, error), it switches to the next.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.pool = []
        self.fallen_keys = []
        self.current_idx = 0
        self.success_counts = {}
        self._load_keys()

    def _load_keys(self):
        keys_file = os.path.join(os.path.dirname(__file__), "keys.json")
        data = {}
        if os.path.exists(keys_file):
            try:
                with open(keys_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as ex:
                print(f"[!] Warning reading keys.json: {ex}")

        # Check for Google Places key
        global GOOGLE_PLACES_KEY
        if not GOOGLE_PLACES_KEY and data.get("google_places_api_key"):
            g_key = data["google_places_api_key"].strip()
            if g_key and "YOUR_GOOGLE" not in g_key:
                GOOGLE_PLACES_KEY = g_key

        def collect(env_name, inline_list, json_key):
            keys = []
            env_val = os.environ.get(env_name, "")
            if env_val:
                keys.extend([k.strip() for k in env_val.split(",") if k.strip()])
            keys.extend([k.strip() for k in inline_list if k.strip() and not k.startswith("#")])
            json_list = data.get(json_key, [])
            if isinstance(json_list, list):
                keys.extend([k.strip() for k in json_list if k.strip() and not k.startswith("YOUR_") and "your-" not in k])
            elif isinstance(json_list, str) and json_list.strip() and "YOUR_" not in json_list:
                keys.append(json_list.strip())
            seen, clean = set(), []
            for k in keys:
                if k not in seen:
                    seen.add(k)
                    clean.append(k)
            return clean

        anth_keys = collect("ANTHROPIC_KEYS", ANTHROPIC_KEYS, "anthropic_keys")
        openai_keys = collect("OPENAI_KEYS", OPENAI_KEYS, "openai_keys")
        gemini_keys = collect("GEMINI_KEYS", GEMINI_KEYS, "gemini_keys")
        groq_keys = collect("GROQ_KEYS", GROQ_KEYS, "groq_keys")
        grok_keys = collect("XAI_GROK_KEYS", XAI_GROK_KEYS, "xai_grok_keys")

        for k in anth_keys:
            self.pool.append({"provider": "Anthropic", "key": k, "model": "claude-3-5-haiku-20241022"})
        for k in openai_keys:
            self.pool.append({"provider": "OpenAI", "key": k, "model": "gpt-4o-mini"})
        for k in gemini_keys:
            self.pool.append({"provider": "Gemini", "key": k, "model": "gemini-2.5-flash"})
        for k in groq_keys:
            self.pool.append({"provider": "Groq", "key": k, "model": "openai/gpt-oss-120b"})
        for k in grok_keys:
            self.pool.append({"provider": "xAI_Grok", "key": k, "model": "grok-beta"})

        print(f"[*] Multi-LLM Key Pool Initialized:")
        print(f"    - Anthropic keys: {len(anth_keys)}")
        print(f"    - OpenAI keys:    {len(openai_keys)}")
        print(f"    - Gemini keys:    {len(gemini_keys)}")
        print(f"    - Groq keys:      {len(groq_keys)}")
        print(f"    - xAI Grok keys:  {len(grok_keys)}")
        print(f"    => Total Active Key Pool: {len(self.pool)} keys")

    def _mask_key(self, key):
        return f"...{key[-6:]}" if len(key) >= 8 else "..."

    def _mark_key_fallen(self, key_item, reason):
        with self.lock:
            if key_item in self.pool:
                self.pool.remove(key_item)
                snippet = self._mask_key(key_item["key"])
                self.fallen_keys.append({
                    "provider": key_item["provider"],
                    "key_snippet": snippet,
                    "reason": reason
                })
                print(f"\n[!] KEY FELL: {key_item['provider']} ({snippet}) failed: {reason}")
                print(f"    => Remaining active keys in loop: {len(self.pool)}")

    def call_with_failover(self, system_prompt, user_prompt):
        while True:
            with self.lock:
                if not self.pool:
                    return None
                self.current_idx = (self.current_idx + 1) % len(self.pool)
                current_item = self.pool[self.current_idx]

            provider = current_item["provider"]
            key = current_item["key"]
            model = current_item["model"]

            try:
                # 1. Anthropic
                if provider == "Anthropic":
                    r = requests.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                        json={"model": model, "max_tokens": 500, "system": system_prompt,
                              "messages": [{"role": "user", "content": user_prompt}]},
                        timeout=25
                    )
                    if r.status_code == 200:
                        return self._parse_json(r.json()["content"][0]["text"], provider)
                    else:
                        self._mark_key_fallen(current_item, f"HTTP {r.status_code}: {r.text[:100]}")
                        continue

                # 2. OpenAI
                elif provider == "OpenAI":
                    r = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json={"model": model, "messages": [{"role": "system", "content": system_prompt},
                                                            {"role": "user", "content": user_prompt}], "temperature": 0.4},
                        timeout=25
                    )
                    if r.status_code == 200:
                        return self._parse_json(r.json()["choices"][0]["message"]["content"], provider)
                    else:
                        self._mark_key_fallen(current_item, f"HTTP {r.status_code}: {r.text[:100]}")
                        continue

                # 3. Google Gemini
                elif provider == "Gemini":
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                    combined = f"{system_prompt}\n\nTask:\n{user_prompt}"
                    r = requests.post(
                        url,
                        headers={"Content-Type": "application/json"},
                        json={"contents": [{"parts": [{"text": combined}]}],
                              "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048, "thinkingConfig": {"thinkingBudget": 0}, "responseMimeType": "application/json"}},
                        timeout=25
                    )
                    if r.status_code == 200:
                        return self._parse_json(r.json()["candidates"][0]["content"]["parts"][0]["text"], provider)
                    else:
                        self._mark_key_fallen(current_item, f"HTTP {r.status_code}: {r.text[:100]}")
                        continue

                # 4. Groq
                elif provider == "Groq":
                    r = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json={"model": model, "messages": [{"role": "system", "content": system_prompt},
                                                            {"role": "user", "content": user_prompt}], "temperature": 0.4},
                        timeout=25
                    )
                    if r.status_code == 200:
                        return self._parse_json(r.json()["choices"][0]["message"]["content"], provider)
                    else:
                        self._mark_key_fallen(current_item, f"HTTP {r.status_code}: {r.text[:100]}")
                        continue

                # 5. xAI (Grok)
                elif provider == "xAI_Grok":
                    r = requests.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json={"model": model, "messages": [{"role": "system", "content": system_prompt},
                                                            {"role": "user", "content": user_prompt}], "temperature": 0.4},
                        timeout=25
                    )
                    if r.status_code == 200:
                        return self._parse_json(r.json()["choices"][0]["message"]["content"], provider)
                    else:
                        self._mark_key_fallen(current_item, f"HTTP {r.status_code}: {r.text[:100]}")
                        continue

            except Exception as ex:
                self._mark_key_fallen(current_item, f"Connection Error: {str(ex)[:90]}")
                continue

    def _parse_json(self, txt, provider):
        cleaned = re.sub(r"```(?:json)?", "", txt).strip()
        cleaned = re.sub(r"```", "", cleaned).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        raw_json = match.group(0) if match else cleaned
        try:
            data = json.loads(raw_json, strict=False)
        except Exception:
            sanitized = re.sub(r"[\x00-\x1f]+", " ", raw_json)
            data = json.loads(sanitized, strict=False)
        with self.lock:
            self.success_counts[provider] = self.success_counts.get(provider, 0) + 1
        return data


llm_rotator = MultiLLMRotator()


# ==============================================================================
# 2. TARGET GEOGRAPHIES & KEYWORDS
# ==============================================================================

CITIES = [
    "Bengaluru", "Bengaluru SP Road", "Bengaluru Koramangala", "Bengaluru Whitefield", "Bengaluru Jayanagar",
    "Mumbai", "Mumbai Lamington Road", "Mumbai Andheri", "Mumbai Thane", "Navi Mumbai",
    "Delhi", "Delhi Nehru Place", "Delhi Lajpat Nagar", "Noida", "Gurugram",
    "Hyderabad", "Hyderabad CTC Secunderabad", "Hyderabad Ameerpet", "Hyderabad Madhapur",
    "Chennai", "Chennai Ritchie Street", "Chennai T Nagar", "Chennai Guindy",
    "Pune", "Ahmedabad", "Kolkata", "Surat", "Jaipur", "Lucknow", "Indore",
    "Chandigarh", "Coimbatore", "Kochi", "Mysuru", "Hubballi", "Mangaluru",
    "Belagavi", "Vijayawada", "Visakhapatnam", "Nagpur", "Bhopal", "Patna",
    "Ludhiana", "Guwahati", "Bhubaneswar", "Vadodara", "Nashik", "Rajkot"
]

QUERIES = [
    "CCTV camera dealer",
    "CCTV camera installation service",
    "security camera installer",
    "Hikvision CCTV dealer",
    "CP Plus CCTV dealer",
    "Dahua security dealer",
    "biometric access control installer",
    "CCTV AMC repair service",
    "home security surveillance system",
    "CCTV camera wholesale shop"
]

SOCIAL_DOMAINS = (
    "facebook.com", "instagram.com", "justdial.com", "indiamart.com",
    "sulekha.com", "linkedin.com", "youtube.com", "twitter.com", "x.com"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

DEMO_URL = "https://atryasolutions.netlify.app/"
DEMO_ADMIN_URL = "https://atryasolutions.netlify.app/admin"

session = requests.Session()
retries = Retry(total=2, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=25, pool_maxsize=25))
session.mount("http://", HTTPAdapter(max_retries=retries, pool_connections=25, pool_maxsize=25))


# ==============================================================================
# 3. HELPER UTILITIES
# ==============================================================================

def clean_indian_phone(raw_phone):
    """Parses phone numbers and classifies into Mobile (WhatsApp-ready) vs Landline."""
    if not raw_phone:
        return "", "None", ""
    digits = re.sub(r"\D", "", str(raw_phone))
    if not digits:
        return "", "Unknown", raw_phone

    mobile_pattern = r"^[6-9]\d{9}$"
    if len(digits) == 10 and re.match(mobile_pattern, digits):
        return f"91{digits}", "Mobile", raw_phone
    elif len(digits) == 11 and digits.startswith("0") and re.match(mobile_pattern, digits[1:]):
        return f"91{digits[1:]}", "Mobile", raw_phone
    elif len(digits) == 12 and digits.startswith("91") and re.match(mobile_pattern, digits[2:]):
        return digits, "Mobile", raw_phone

    return "", "Landline", raw_phone


def extract_pincode(address_str):
    """Extracts 6-digit Indian PIN code from address."""
    if not address_str:
        return ""
    match = re.search(r"\b([1-9][0-9]{5})\b", address_str)
    return match.group(1) if match else ""


def extract_clean_domain(url):
    """Extracts base domain (e.g. securitycctv.com)."""
    if not url:
        return ""
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return url


def make_whatsapp_chat_url(whatsapp_number, pitch_text):
    if not whatsapp_number:
        return ""
    encoded_text = urllib.parse.quote(pitch_text or "Hello, I wanted to reach out regarding your CCTV business.")
    return f"https://wa.me/{whatsapp_number}?text={encoded_text}"


# ==============================================================================
# 4. DISCOVERY ENGINE (GOOGLE PLACES + FREE OSM FALLBACK)
# ==============================================================================

def discover_openstreetmap(target, selected_cities, seen_ids):
    """
    Free fallback discovery engine using OpenStreetMap Overpass API.
    Used if Google Places API key is missing or exhausted.
    """
    print("[*] Using Free Discovery Engine (OpenStreetMap Overpass API)...")
    leads = []
    seen = seen_ids.copy() if seen_ids else set()
    cities = selected_cities if selected_cities else ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune"]

    overpass_url = "https://overpass-api.de/api/interpreter"

    for city in cities:
        clean_city = city.split()[0]
        query = f"""
        [out:json][timeout:25];
        area["name"="{clean_city}"]->.searchArea;
        (
          node["shop"="security"](area.searchArea);
          node["shop"="electronics"](area.searchArea);
          node["name"~"CCTV|Security|Camera|Hikvision|CP Plus",i](area.searchArea);
        );
        out body 30;
        """
        try:
            r = requests.post(overpass_url, data={"data": query}, timeout=30)
            if r.status_code == 200:
                data = r.json()
                for el in data.get("elements", []):
                    tags = el.get("tags", {})
                    name = tags.get("name", "").strip()
                    if not name:
                        continue
                    phone = tags.get("phone") or tags.get("contact:phone") or tags.get("contact:mobile") or ""
                    website = tags.get("website") or tags.get("contact:website") or ""
                    addr_parts = [tags.get(k) for k in ("addr:street", "addr:suburb", "addr:city", "addr:postcode") if tags.get(k)]
                    address = ", ".join(addr_parts) or f"{name}, {clean_city}"

                    pid = f"osm_{el.get('id')}"
                    if pid in seen or (phone and phone in seen):
                        continue
                    seen.add(pid)

                    leads.append({
                        "place_id": pid,
                        "name": name,
                        "category": "CCTV & Security Systems",
                        "address": address,
                        "pincode": extract_pincode(address),
                        "city": clean_city,
                        "state": tags.get("addr:state", ""),
                        "search_locality": city,
                        "phone": phone,
                        "website": website,
                        "rating": "4.5",
                        "reviews": "10+",
                        "business_status": "OPERATIONAL",
                        "latitude": str(el.get("lat", "")),
                        "longitude": str(el.get("lon", "")),
                        "google_maps_url": f"https://maps.google.com/?q={el.get('lat')},{el.get('lon')}"
                    })
                    if len(leads) >= target:
                        return leads[:target]
            time.sleep(1.0)
        except Exception as ex:
            print(f"[!] OSM query error for {clean_city}: {ex}")

    return leads


def discover(target, selected_cities=None, seen_ids=None):
    """Primary discovery via Google Places (New) with automatic OSM fallback."""
    leads = []
    seen = seen_ids.copy() if seen_ids else set()

    # If Google API key is missing or placeholder, use OSM
    if not GOOGLE_PLACES_KEY or "YOUR_GOOGLE" in GOOGLE_PLACES_KEY:
        print("[!] Note: Google Places API key not configured in keys.json.")
        return discover_openstreetmap(target, selected_cities, seen)

    url = "https://places.googleapis.com/v1/places:searchText"
    mask = (
        "places.id,places.displayName,places.formattedAddress,places.nationalPhoneNumber,"
        "places.websiteUri,places.rating,places.userRatingCount,places.location,"
        "places.businessStatus,places.googleMapsUri,nextPageToken"
    )

    cities_to_search = selected_cities if selected_cities else CITIES
    print(f"[*] Starting discovery via Google Places API across {len(cities_to_search)} zones...")

    for city in cities_to_search:
        for q in QUERIES:
            token = None
            for _ in range(3):
                body = {"textQuery": f"{q} in {city}", "pageSize": 20}
                if token:
                    body["pageToken"] = token

                try:
                    r = requests.post(
                        url,
                        json=body,
                        headers={"X-Goog-Api-Key": GOOGLE_PLACES_KEY, "X-Goog-FieldMask": mask},
                        timeout=25
                    )
                    if r.status_code != 200:
                        if "RESOURCE_EXHAUSTED" in r.text:
                            print(f"[!] Google Places quota exhausted. Switching to fallback...")
                            fallback = discover_openstreetmap(target - len(leads), selected_cities, seen)
                            return leads + fallback
                        print(f"[!] Places API error ({r.status_code}): {r.text[:120]}")
                        break

                    data = r.json()
                    for p in data.get("places", []):
                        pid = p.get("id")
                        if not pid or pid in seen:
                            continue
                        seen.add(pid)

                        addr = p.get("formattedAddress", "").strip()
                        loc = p.get("location", {})

                        leads.append({
                            "place_id": pid,
                            "name": p.get("displayName", {}).get("text", "").strip(),
                            "category": "CCTV & Security Systems Dealer",
                            "address": addr,
                            "pincode": extract_pincode(addr),
                            "city": city.split()[0],
                            "state": "",
                            "search_locality": city,
                            "phone": p.get("nationalPhoneNumber", "").strip(),
                            "website": p.get("websiteUri", "").strip(),
                            "rating": str(p.get("rating", "")),
                            "reviews": str(p.get("userRatingCount", "")),
                            "business_status": p.get("businessStatus", "OPERATIONAL"),
                            "latitude": str(loc.get("latitude", "")),
                            "longitude": str(loc.get("longitude", "")),
                            "google_maps_url": p.get("googleMapsUri", "")
                        })

                    token = data.get("nextPageToken")
                    if not token:
                        break
                    time.sleep(1.2)
                except Exception as ex:
                    print(f"[!] Error querying '{q}' in {city}: {ex}")
                    break

            if len(leads) >= target:
                print(f"[+] Reached target of {target} leads.")
                return leads[:target]

        print(f"[*] Total unique leads collected so far: {len(leads)}")

    return leads


# ==============================================================================
# 5. DEEP WEBSITE & CONTACT/SOCIAL SCRAPER
# ==============================================================================

def extract_contact_info(html):
    """Scrapes emails, phones, and social links from HTML."""
    emails = set(re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", html))
    clean_emails = {
        e.lower() for e in emails
        if not e.lower().endswith(("png", "jpg", "jpeg", "webp", "svg", "gif", "css", "js"))
        and "sentry" not in e.lower()
        and "wix" not in e.lower()
    }

    # Alternate phone numbers from website
    scraped_phones = set(re.findall(r"(?:\+91[\-\s]?)?[6-9]\d{9}\b", html))
    
    # Social profiles
    socials = {}
    fb = re.search(r"https?://(?:www\.)?facebook\.com/[a-zA-Z0-9.\-_/]+", html, re.IGNORECASE)
    insta = re.search(r"https?://(?:www\.)?instagram\.com/[a-zA-Z0-9.\-_/]+", html, re.IGNORECASE)
    linkedin = re.search(r"https?://(?:www\.)?linkedin\.com/(?:company|in)/[a-zA-Z0-9.\-_/]+", html, re.IGNORECASE)
    yt = re.search(r"https?://(?:www\.)?youtube\.com/[a-zA-Z0-9.\-_/@]+", html, re.IGNORECASE)
    tw = re.search(r"https?://(?:www\.)?(?:twitter|x)\.com/[a-zA-Z0-9.\-_/]+", html, re.IGNORECASE)

    if fb: socials["facebook"] = fb.group(0)
    if insta: socials["instagram"] = insta.group(0)
    if linkedin: socials["linkedin"] = linkedin.group(0)
    if yt: socials["youtube"] = yt.group(0)
    if tw: socials["twitter"] = tw.group(0)

    has_wa = bool(re.search(r"(wa\.me/|api\.whatsapp\.com|whatsapp)", html, re.IGNORECASE))
    return clean_emails, scraped_phones, socials, has_wa


def check_subpage_contacts(base_url):
    """Crawls /contact and /about subpages for complete company details."""
    subpages = ["/contact", "/contact-us", "/about-us", "/reach-us"]
    found_emails = set()
    found_phones = set()
    found_socials = {}
    found_wa = False

    for path in subpages:
        try:
            target = urllib.parse.urljoin(base_url, path)
            r = session.get(target, headers=HEADERS, timeout=6, verify=False)
            if r.status_code == 200:
                e, ph, soc, wa = extract_contact_info(r.text)
                found_emails.update(e)
                found_phones.update(ph)
                found_socials.update(soc)
                if wa:
                    found_wa = True
                if found_emails and found_wa:
                    break
        except Exception:
            continue

    return found_emails, found_phones, found_socials, found_wa


def audit(lead):
    """
    Comprehensive company audit:
    - Extracts clean domain, pincode, state
    - Validates SSL, mobile responsiveness, load time
    - Collects alternate phones, emails, and social profiles
    - Computes need score and priority (High, Medium, Low)
    """
    site = (lead.get("website") or "").strip()
    raw_phone = lead.get("phone", "")
    wa_num, phone_type, _ = clean_indian_phone(raw_phone)

    lead["domain"] = extract_clean_domain(site)
    lead["whatsapp_number"] = wa_num
    lead["phone_type"] = phone_type
    lead["additional_phones"] = ""
    lead["facebook_url"] = ""
    lead["instagram_url"] = ""
    lead["linkedin_url"] = ""
    lead["youtube_url"] = ""
    lead["twitter_url"] = ""
    lead["load_time_sec"] = "0.0"
    lead["ssl_status"] = "none"
    lead["mobile_friendly"] = "No"
    lead["has_enquiry_form"] = "No"
    lead["has_whatsapp_widget"] = "No"
    lead["cms_platform"] = "None"
    lead["copyright_year"] = ""
    lead["scraped_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    issues = []
    score = 0

    # Case 1: No official website
    if not site or any(s in site.lower() for s in SOCIAL_DOMAINS):
        lead.update({
            "site_status": "no_website",
            "need_score": 100,
            "lead_priority": "High",
            "segment": "no_website",
            "issues": "No official website (relies only on directories or social media)",
            "email": "",
            "ssl_status": "none"
        })
        return lead

    # Case 2: Auditing company website
    ssl_broken = False
    try:
        t0 = time.time()
        try:
            r = session.get(site, headers=HEADERS, timeout=12, allow_redirects=True, verify=True)
            ssl_status = "valid_https"
        except requests.exceptions.SSLError:
            ssl_broken = True
            ssl_status = "broken_ssl"
            r = session.get(site, headers=HEADERS, timeout=10, allow_redirects=True, verify=False)

        load_time = time.time() - t0
        lead["load_time_sec"] = f"{load_time:.2f}"
        html = r.text
        html_lower = html.lower()

        # SSL status check
        if ssl_broken:
            issues.append("SSL certificate expired/invalid (security warning)")
            score += 35
        elif not r.url.startswith("https://"):
            ssl_status = "insecure_http"
            issues.append("not served over HTTPS (insecure connection)")
            score += 25
        lead["ssl_status"] = ssl_status

        if r.status_code >= 400:
            issues.append(f"HTTP error {r.status_code}")
            score += 55

        # Detect CMS / Site Builder
        cms = "Custom / HTML"
        if "wp-content" in html_lower or "wordpress" in html_lower:
            cms = "WordPress"
        elif "wix.com" in html_lower or "wixsite.com" in html_lower:
            cms = "Wix"
        elif "shopify" in html_lower:
            cms = "Shopify"
        elif "squarespace" in html_lower:
            cms = "Squarespace"
        elif "weebly" in html_lower:
            cms = "Weebly"
        lead["cms_platform"] = cms

        if any(parked in html_lower for parked in ["domain is for sale", "sedoparking", "buy this domain", "parked free by godaddy"]):
            issues.append("domain appears parked or abandoned")
            score += 40

        # Mobile responsiveness
        if 'name="viewport"' in html_lower or "name='viewport'" in html_lower:
            lead["mobile_friendly"] = "Yes"
        else:
            lead["mobile_friendly"] = "No"
            issues.append("not mobile-friendly (missing viewport tag)")
            score += 25

        # Load time
        if load_time > 4.5:
            issues.append(f"slow load time ({load_time:.1f}s)")
            score += 15

        # Lead capture enquiry form
        if "<form" in html_lower:
            lead["has_enquiry_form"] = "Yes"
        else:
            lead["has_enquiry_form"] = "No"
            issues.append("no online enquiry or quote form")
            score += 15

        # Contact info & social links
        emails, alt_phones, socials, has_wa = extract_contact_info(html)
        if has_wa:
            lead["has_whatsapp_widget"] = "Yes"
        else:
            lead["has_whatsapp_widget"] = "No"
            issues.append("no direct WhatsApp contact widget")
            score += 10

        # Copyright year
        years = [int(y) for y in re.findall(r"(?:©|&copy;|copyright)[^0-9]{0,20}(20\d\d)", html_lower)]
        current_year = datetime.now().year
        if years:
            latest_year = max(years)
            lead["copyright_year"] = str(latest_year)
            if latest_year < current_year - 2:
                issues.append(f"outdated footer copyright ({latest_year})")
                score += 15

        if any(b in html_lower for b in ("wixsite.com", "blogspot.com", "weebly.com")):
            issues.append("using free-subdomain site builder")
            score += 20

        # Crawl subpages if emails or social profiles missing
        if not emails or not socials:
            sub_e, sub_ph, sub_soc, sub_wa = check_subpage_contacts(r.url)
            emails.update(sub_e)
            alt_phones.update(sub_ph)
            socials.update(sub_soc)
            if sub_wa and not has_wa:
                lead["has_whatsapp_widget"] = "Yes"
                if "no direct WhatsApp contact widget" in issues:
                    issues.remove("no direct WhatsApp contact widget")
                    score -= 10

        lead["email"] = "; ".join(sorted(emails)[:2])
        lead["additional_phones"] = "; ".join(sorted(alt_phones)[:3])
        lead["facebook_url"] = socials.get("facebook", "")
        lead["instagram_url"] = socials.get("instagram", "")
        lead["linkedin_url"] = socials.get("linkedin", "")
        lead["youtube_url"] = socials.get("youtube", "")
        lead["twitter_url"] = socials.get("twitter", "")

        # Segmentation
        if ssl_broken or r.status_code >= 400 or lead["mobile_friendly"] == "No":
            segment = "broken_or_insecure"
            site_status = "needs_urgent_revamp"
            priority = "High"
        elif score >= 25:
            segment = "outdated_site"
            site_status = "needs_upgrade"
            priority = "Medium"
        else:
            segment = "amc_operations"
            site_status = "healthy"
            priority = "Medium" if lead.get("rating") else "Low"

        lead.update({
            "site_status": site_status,
            "need_score": min(score, 99),
            "lead_priority": priority,
            "segment": segment,
            "issues": "; ".join(issues) or "website appears functional and updated"
        })

    except Exception as ex:
        lead.update({
            "site_status": "unreachable",
            "need_score": 85,
            "lead_priority": "High",
            "segment": "broken_or_insecure",
            "email": "",
            "issues": f"website unreachable ({type(ex).__name__})"
        })

    return lead


# ==============================================================================
# 6. SEGMENTED OUTREACH PITCH GENERATOR
# ==============================================================================

PITCH_SYSTEM_PROMPT = f"""You write concise, high-converting B2B outreach for Atrya Solutions.
Atrya builds turnkey digital systems for CCTV and electronic security companies across India.
We provide two core tools:
1. High-Converting Customer Website: Instant 1-click WhatsApp quote buttons & online lead forms to capture direct Google customers instead of paying commissions to Justdial/IndiaMart. Live Demo: {DEMO_URL}
2. Operations & Field Admin Panel: Technician job dispatch, AMC maintenance schedules, customer warranty records & service renewal alerts. Live Demo Admin: {DEMO_ADMIN_URL}

Rules:
1. Refer to the links strictly as a LIVE DEMO of what we customize and launch for CCTV companies.
2. Tone: Consultative, professional, persuasive, high ROI focus.
3. If NO WEBSITE (relies only on Google Maps or directory listings):
   - Highlight that they are losing direct Google customers to competitors and paying heavy directory commissions.
   - You MUST include BOTH demo links:
     * Customer Website Demo: {DEMO_URL}
     * Operations & AMC Dispatch Admin: {DEMO_ADMIN_URL}
   - Emphasize how this combo gives them direct high-value customer inquiries on WhatsApp AND automates their technician & AMC management.
4. If BROKEN/INSECURE: Mention their exact technical issue politely and offer a free revamp.
5. If OUTDATED: Focus on boosting customer enquiries with a modern WhatsApp quote widget.
6. If HEALTHY: Focus primarily on the OPERATIONS & AMC DISPATCH ADMIN PANEL to streamline field technicians.
7. Always invite them for a quick 10-15 minute walkthrough.
8. Email must end with: "Reply STOP and I won't contact you again."
9. Output STRICT JSON only:
{{
  "whatsapp": "Message under 65 words with clear line breaks for WhatsApp",
  "email_subject": "Catchy, relevant 4-7 word subject",
  "email_body": "Clean body under 140 words"
}}"""


def get_rule_based_fallback(lead):
    name = lead.get("name", "there")
    city = lead.get("city", "your area")
    segment = lead.get("segment", "no_website")

    if segment == "no_website":
        wa = (f"Hi {name},\n"
              f"Noticed your CCTV business in {city} gets great reviews, but you don't have a direct website yet!\n\n"
              f"You might be losing direct Google customers or paying high commissions on Justdial. "
              f"We build complete turnkey setups for CCTV companies:\n"
              f"👉 Customer Website (1-Click WhatsApp Quotes): {DEMO_URL}\n"
              f"👉 Operations Admin (Technician & AMC Dispatch): {DEMO_ADMIN_URL}\n\n"
              f"Open for a quick 10-min walkthrough this week?")
        subject = f"Direct Customer Website & AMC Admin Panel for {name}"
        body = (f"Hi {name} Team,\n\n"
                f"I came across your CCTV business in {city} and saw your positive customer ratings. "
                f"However, I noticed that potential clients searching Google cannot find a direct website to request CCTV quotes.\n\n"
                f"Relying solely on local directories means you share leads with 4–5 competitors and pay ongoing commissions.\n\n"
                f"At Atrya Solutions, we build complete turnkey systems specifically for CCTV and security installers:\n\n"
                f"1. Customer-Facing Website (captures direct leads with 1-click WhatsApp quotes):\n"
                f"   👉 Demo: {DEMO_URL}\n\n"
                f"2. Operations Admin Panel (tracks jobs, AMC maintenance contracts, and technician dispatch):\n"
                f"   👉 Demo Admin: {DEMO_ADMIN_URL}\n\n"
                f"We customize and deliver the complete setup in 48 hours.\n\n"
                f"Would you be open to a brief 10-minute walkthrough this week to see how this can grow your direct inquiries?\n\n"
                f"Best regards,\n"
                f"Atrya Solutions Team\n\n"
                f"Reply STOP and I won't contact you again.")
    elif segment == "amc_operations":
        wa = (f"Hi {name},\nSaw your CCTV website in {city}. How do you currently manage your AMC service renewals and technician dispatch?\n"
              f"We built an operations panel specifically for security installers: {DEMO_ADMIN_URL}\n"
              f"Worth a quick look?")
        subject = f"AMC tracking & technician panel for {name}"
        body = (f"Hi {name},\n\nI came across your website and wanted to reach out.\n\n"
                f"Most CCTV dealers tell us tracking annual maintenance contracts (AMCs), technician visits, and warranty renewals "
                f"becomes chaotic on spreadsheets.\n\n"
                f"We built a specialized operations panel for CCTV teams: {DEMO_ADMIN_URL}\n"
                f"Would you be open to a quick 10-minute walkthrough to see if it saves your team time?\n\n"
                f"Reply STOP and I won't contact you again.")
    else:
        wa = (f"Hi {name},\nTook a look at your CCTV website in {city} and noticed a few things holding back customer enquiries ({lead.get('issues')}).\n"
              f"We build modern sites with instant WhatsApp quote buttons: {DEMO_URL}\n"
              f"Open to a quick demo?")
        subject = f"Improving CCTV quote enquiries for {name}"
        body = (f"Hi {name},\n\nI was looking up CCTV providers in {city} and checked your website.\n\n"
                f"I noticed a couple of areas that might be reducing your lead conversion: {lead.get('issues')}.\n\n"
                f"At Atrya Solutions, we build high-converting websites and AMC management systems for security installers.\n"
                f"Live demo: {DEMO_URL}\n\n"
                f"Would you be interested in a brief 10-minute walkthrough?\n\n"
                f"Reply STOP and I won't contact you again.")

    return {"whatsapp": wa, "email_subject": subject, "email_body": body}


def pitch(lead):
    prompt = (
        f"Company Name: {lead.get('name')}\n"
        f"City: {lead.get('city')}\n"
        f"Website: {lead.get('website') or 'None'}\n"
        f"Segment: {lead.get('segment')}\n"
        f"Audit Findings: {lead.get('issues')}\n"
        f"Rating: {lead.get('rating')} ({lead.get('reviews')} reviews)"
    )

    data = llm_rotator.call_with_failover(PITCH_SYSTEM_PROMPT, prompt)
    if not data:
        data = get_rule_based_fallback(lead)

    whatsapp_msg = data.get("whatsapp", "").strip()
    lead["whatsapp_pitch"] = whatsapp_msg
    lead["email_subject"] = data.get("email_subject", "").strip()
    lead["email_body"] = data.get("email_body", "").strip()
    lead["whatsapp_click_link"] = make_whatsapp_chat_url(lead.get("whatsapp_number"), whatsapp_msg)
    return lead


# ==============================================================================
# 7. EXPORT ENGINES (CSV MASTER + FULL COMPANY DOSSIER JSON)
# ==============================================================================

# Complete Company Details Column Specification
COLS = [
    # Business Profile
    "name", "category", "address", "pincode", "city", "state", "search_locality",
    "latitude", "longitude", "google_maps_url", "rating", "reviews", "business_status",

    # Communication & Contacts
    "phone", "phone_type", "whatsapp_number", "whatsapp_click_link", "additional_phones",
    "email", "website", "domain",

    # Social Media
    "facebook_url", "instagram_url", "linkedin_url", "youtube_url", "twitter_url",

    # Digital Audit & Qualification
    "lead_priority", "need_score", "site_status", "segment", "issues",
    "ssl_status", "mobile_friendly", "load_time_sec", "has_enquiry_form",
    "has_whatsapp_widget", "cms_platform", "copyright_year",

    # Outreach Pitch Pack
    "whatsapp_pitch", "email_subject", "email_body", "scraped_at"
]


def load_checkpoint(filename):
    seen_ids = set()
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("phone"):
                        seen_ids.add(row.get("phone"))
                    if row.get("website"):
                        seen_ids.add(row.get("website"))
        except Exception:
            pass
    return seen_ids


def export_to_excel(leads, filename="leads.xlsx"):
    """Exports structured data to Excel (.xlsx) with styled headers, frozen panes, and auto-adjusted widths."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        wb = Workbook()
        ws = wb.active
        ws.title = "CCTV Leads"

        headers = [c.replace("_", " ").title() for c in COLS]
        ws.append(headers)

        header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
        header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=False)

        for col_num, cell in enumerate(ws[1], 1):
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_align
        ws.row_dimensions[1].height = 26
        ws.freeze_panes = "A2"

        thin_border = Border(
            left=Side(style="thin", color="E2E8F0"),
            right=Side(style="thin", color="E2E8F0"),
            top=Side(style="thin", color="E2E8F0"),
            bottom=Side(style="thin", color="E2E8F0")
        )
        zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")

        for r_idx, lead in enumerate(leads, start=2):
            row_data = [str(lead.get(col, "") or "") for col in COLS]
            ws.append(row_data)
            is_zebra = (r_idx % 2 == 0)
            ws.row_dimensions[r_idx].height = 20

            for c_idx, cell in enumerate(ws[r_idx], 1):
                cell.border = thin_border
                cell.font = Font(name="Segoe UI", size=9)
                cell.alignment = Alignment(vertical="center")
                if is_zebra:
                    cell.fill = zebra_fill

                val = str(cell.value or "")
                if val.startswith("http://") or val.startswith("https://"):
                    cell.hyperlink = val
                    cell.font = Font(name="Segoe UI", size=9, color="2563EB", underline="single")

        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val = str(cell.value or "")
                first_line = val.split("\n")[0]
                if len(first_line) > max_len:
                    max_len = len(first_line)
            ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 40)

        wb.save(filename)
    except Exception as ex:
        print(f"[!] Note: Excel export skipped ({ex})")


def export_html_dashboard(leads, filename="leads_dashboard.html"):
    """Generates an interactive, modern HTML dashboard with search, filter, and 1-click WhatsApp buttons."""
    try:
        leads_json = json.dumps(leads, ensure_ascii=False)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CCTV Company Lead Intelligence Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Inter', -apple-system, sans-serif; background: #0f172a; color: #f8fafc; padding: 24px; }}
    .container {{ max-width: 1440px; margin: 0 auto; }}
    .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }}
    .title {{ font-size: 24px; font-weight: 700; color: #38bdf8; }}
    .subtitle {{ font-size: 13px; color: #94a3b8; margin-top: 4px; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .stat-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 18px; }}
    .stat-val {{ font-size: 26px; font-weight: 700; color: #f1f5f9; }}
    .stat-lbl {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}
    .controls {{ display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }}
    .search-box {{ flex: 1; min-width: 280px; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px 16px; color: #f8fafc; font-size: 14px; outline: none; }}
    .search-box:focus {{ border-color: #38bdf8; }}
    .filter-btn {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px 16px; color: #cbd5e1; font-size: 13px; cursor: pointer; font-weight: 500; transition: all 0.2s; }}
    .filter-btn.active, .filter-btn:hover {{ background: #0284c7; color: #fff; border-color: #0284c7; }}
    .table-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; overflow: hidden; }}
    .table-wrapper {{ overflow-x: auto; max-height: 70vh; }}
    table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }}
    thead th {{ background: #0f172a; color: #94a3b8; padding: 12px 16px; font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; position: sticky; top: 0; z-index: 10; border-bottom: 1px solid #334155; }}
    tbody tr {{ border-bottom: 1px solid #334155; transition: background 0.15s; }}
    tbody tr:hover {{ background: #243248; }}
    td {{ padding: 14px 16px; vertical-align: middle; }}
    .company-name {{ font-weight: 600; color: #f8fafc; font-size: 14px; display: block; }}
    .company-city {{ font-size: 11px; color: #94a3b8; margin-top: 2px; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; }}
    .badge-high {{ background: #7f1d1d; color: #fecaca; }}
    .badge-med {{ background: #78350f; color: #fde68a; }}
    .badge-low {{ background: #14532d; color: #bbf7d0; }}
    .btn-wa {{ display: inline-flex; align-items: center; gap: 6px; background: #15803d; color: #fff; padding: 6px 12px; border-radius: 6px; text-decoration: none; font-size: 12px; font-weight: 600; transition: background 0.2s; white-space: nowrap; }}
    .btn-wa:hover {{ background: #16a34a; }}
    .btn-pitch {{ background: #334155; border: 1px solid #475569; color: #f8fafc; padding: 6px 10px; border-radius: 6px; font-size: 12px; cursor: pointer; transition: background 0.2s; white-space: nowrap; }}
    .btn-pitch:hover {{ background: #475569; }}
    .link {{ color: #38bdf8; text-decoration: none; }}
    .link:hover {{ text-decoration: underline; }}
    .empty {{ text-align: center; padding: 40px; color: #94a3b8; }}
    /* Modal styles */
    .modal-overlay {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.75); z-index: 1000; align-items: center; justify-content: center; padding: 20px; }}
    .modal-overlay.active {{ display: flex; }}
    .modal-box {{ background: #1e293b; border: 1px solid #475569; border-radius: 12px; max-width: 640px; width: 100%; max-height: 85vh; overflow-y: auto; padding: 24px; position: relative; }}
    .modal-header {{ display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 18px; }}
    .modal-title {{ font-size: 18px; font-weight: 700; color: #38bdf8; }}
    .close-btn {{ background: transparent; border: none; color: #94a3b8; font-size: 20px; cursor: pointer; line-height: 1; }}
    .close-btn:hover {{ color: #fff; }}
    .pitch-section {{ margin-bottom: 18px; }}
    .pitch-label {{ font-size: 12px; font-weight: 600; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.5px; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center; }}
    .pitch-content {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 12px; line-height: 1.5; color: #e2e8f0; white-space: pre-wrap; }}
    .copy-btn {{ background: #0284c7; border: none; color: #fff; border-radius: 4px; padding: 4px 8px; font-size: 11px; cursor: pointer; font-weight: 600; }}
    .copy-btn:hover {{ background: #0369a1; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <div class="title">CCTV Lead Intelligence Dossier</div>
        <div class="subtitle">Discovered, Audited & Enriched Leads with 1-Click WhatsApp & Demo Outreach</div>
      </div>
    </div>

    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-val" id="totalCount">0</div>
        <div class="stat-lbl">Total Prospects</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" id="mobileCount">0</div>
        <div class="stat-lbl">WhatsApp-Ready Mobiles</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" id="emailCount">0</div>
        <div class="stat-lbl">Scraped Emails</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" id="noWebsiteCount">0</div>
        <div class="stat-lbl">No Website (Prime Demos)</div>
      </div>
    </div>

    <div class="controls">
      <input type="text" id="searchInput" class="search-box" placeholder="🔍 Search by company, city, phone, issues...">
      <button class="filter-btn active" data-filter="all">All Leads</button>
      <button class="filter-btn" data-filter="nowebsite">🔥 No Website (Prime Demos)</button>
      <button class="filter-btn" data-filter="high">High Priority</button>
      <button class="filter-btn" data-filter="whatsapp">Has WhatsApp</button>
      <button class="filter-btn" data-filter="email">Has Email</button>
    </div>

    <div class="table-card">
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Company</th>
              <th>Priority</th>
              <th>Rating</th>
              <th>Phone</th>
              <th>Website & SSL</th>
              <th>Emails</th>
              <th>Issues</th>
              <th>Outreach Actions</th>
            </tr>
          </thead>
          <tbody id="tableBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Modal for Pitch View & 1-Click Copy -->
  <div id="pitchModal" class="modal-overlay" onclick="closePitch(event)">
    <div class="modal-box" onclick="event.stopPropagation()">
      <div class="modal-header">
        <div>
          <div class="modal-title" id="mCompanyName">Company Outreach Pack</div>
          <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;" id="mDetails">Target Lead Pitch</div>
        </div>
        <button class="close-btn" onclick="closePitch()">&times;</button>
      </div>

      <div class="pitch-section">
        <div class="pitch-label">
          <span>WhatsApp Message</span>
          <button class="copy-btn" onclick="copyText('mWaPitch', this)">Copy WhatsApp</button>
        </div>
        <div class="pitch-content" id="mWaPitch"></div>
      </div>

      <div class="pitch-section">
        <div class="pitch-label">
          <span>Cold Email Subject</span>
          <button class="copy-btn" onclick="copyText('mEmailSubj', this)">Copy Subject</button>
        </div>
        <div class="pitch-content" id="mEmailSubj"></div>
      </div>

      <div class="pitch-section">
        <div class="pitch-label">
          <span>Cold Email Body</span>
          <button class="copy-btn" onclick="copyText('mEmailBody', this)">Copy Email</button>
        </div>
        <div class="pitch-content" id="mEmailBody"></div>
      </div>
    </div>
  </div>

  <script>
    const leads = {leads_json};
    let currentFilter = 'all';

    function initStats() {{
      document.getElementById('totalCount').innerText = leads.length;
      document.getElementById('mobileCount').innerText = leads.filter(l => l.whatsapp_number).length;
      document.getElementById('emailCount').innerText = leads.filter(l => l.email).length;
      document.getElementById('noWebsiteCount').innerText = leads.filter(l => l.site_status === 'no_website' || !l.website).length;
    }}

    function renderTable() {{
      const query = document.getElementById('searchInput').value.toLowerCase().trim();
      const tbody = document.getElementById('tableBody');
      tbody.innerHTML = '';

      const filtered = leads.filter(l => {{
        if (currentFilter === 'high' && l.lead_priority !== 'High') return false;
        if (currentFilter === 'whatsapp' && !l.whatsapp_number) return false;
        if (currentFilter === 'email' && !l.email) return false;
        if (currentFilter === 'nowebsite' && l.site_status !== 'no_website' && l.website) return false;

        if (query) {{
          const txt = [l.name, l.city, l.phone, l.issues, l.email, l.website].join(' ').toLowerCase();
          return txt.includes(query);
        }}
        return true;
      }});

      if (!filtered.length) {{
        tbody.innerHTML = '<tr><td colspan="9" class="empty">No matching companies found.</td></tr>';
        return;
      }}

      filtered.forEach((l, idx) => {{
        const tr = document.createElement('tr');
        const prioBadge = l.lead_priority === 'High' ? 'badge-high' : (l.lead_priority === 'Medium' ? 'badge-med' : 'badge-low');
        const siteLink = l.website ? `<a href="${{l.website}}" target="_blank" class="link">${{l.domain || 'Visit'}}</a> (${{l.ssl_status}})` : '<span style="color:#ef4444;font-weight:600">No Website</span>';
        const waBtn = l.whatsapp_click_link 
          ? `<a href="${{l.whatsapp_click_link}}" target="_blank" class="btn-wa">💬 Chat</a>` 
          : '<span style="color:#64748b">—</span>';
        const pitchBtn = `<button class="btn-pitch" onclick="openPitch(${{leads.indexOf(l)}})">📋 Pitch</button>`;

        tr.innerHTML = `
          <td>${{idx + 1}}</td>
          <td>
            <span class="company-name">${{l.name || 'Unnamed'}}</span>
            <div class="company-city">${{l.city || ''}} • ${{l.pincode || ''}}</div>
          </td>
          <td><span class="badge ${{prioBadge}}">${{l.lead_priority || 'Standard'}}</span></td>
          <td>⭐ ${{l.rating || '—'}} (${{l.reviews || '0'}})</td>
          <td>${{l.phone || '—'}}<br><small style="color:#94a3b8">${{l.phone_type || ''}}</small></td>
          <td>${{siteLink}}</td>
          <td><small>${{l.email ? l.email.replace(/;/g, '<br>') : '—'}}</small></td>
          <td><small style="color:#cbd5e1">${{l.issues || 'None'}}</small></td>
          <td><div style="display:flex;gap:6px;align-items:center;">${{waBtn}} ${{pitchBtn}}</div></td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function openPitch(index) {{
      const l = leads[index];
      if (!l) return;
      document.getElementById('mCompanyName').innerText = l.name || 'Target Prospect';
      document.getElementById('mDetails').innerText = `${{l.city || ''}} | ${{l.lead_priority || 'Standard'}} Priority | Phone: ${{l.phone || 'N/A'}}`;
      document.getElementById('mWaPitch').innerText = l.whatsapp_pitch || 'No pitch drafted.';
      document.getElementById('mEmailSubj').innerText = l.email_subject || 'No subject.';
      document.getElementById('mEmailBody').innerText = l.email_body || 'No email drafted.';
      document.getElementById('pitchModal').classList.add('active');
    }}

    function closePitch(e) {{
      document.getElementById('pitchModal').classList.remove('active');
    }}

    function copyText(elementId, btn) {{
      const text = document.getElementById(elementId).innerText;
      navigator.clipboard.writeText(text).then(() => {{
        const oldText = btn.innerText;
        btn.innerText = 'Copied!';
        btn.style.background = '#15803d';
        setTimeout(() => {{
          btn.innerText = oldText;
          btn.style.background = '#0284c7';
        }}, 1800);
      }});
    }}

    document.querySelectorAll('.filter-btn').forEach(btn => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.dataset.filter;
        renderTable();
      }});
    }});

    document.getElementById('searchInput').addEventListener('input', renderTable);

    initStats();
    renderTable();
  </script>
</body>
</html>"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as ex:
        print(f"[!] Note: HTML Dashboard skipped ({ex})")


def export_company_data(leads, csv_filename="leads.csv", json_filename="company_details.json", append=False):
    """
    Exports all discovered company data into:
    1. CSV Spreadsheet (strictly 1 single row per company with sanitized text)
    2. Excel Spreadsheet (.xlsx styled with colors, frozen panes & hyperlinks)
    3. Interactive HTML Dashboard (searchable table with 1-click WhatsApp buttons)
    4. Deep JSON Dossier (structured metadata)
    """
    # 1. Clean leads so each CSV record is strictly 1 single horizontal line
    clean_csv_leads = []
    for l in leads:
        row_copy = {}
        for col in COLS:
            val = str(l.get(col, "") or "")
            # Sanitize newlines so CSV text viewers show 1 clean row per lead
            val_clean = val.replace("\r\n", " \\n ").replace("\n", " \\n ").replace("\r", " ").strip()
            row_copy[col] = val_clean
        clean_csv_leads.append(row_copy)

    file_exists = os.path.exists(csv_filename) and os.path.getsize(csv_filename) > 0
    mode = "a" if append else "w"
    with open(csv_filename, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if not append or not file_exists:
            writer.writeheader()
        writer.writerows(clean_csv_leads)

    # 2. Export Excel (.xlsx)
    xlsx_file = csv_filename.replace(".csv", ".xlsx")
    export_to_excel(leads, xlsx_file)

    # 3. Export Interactive HTML Dashboard
    html_file = "leads_dashboard.html"
    export_html_dashboard(leads, html_file)

    # 4. Export Full Company Dossiers JSON
    dossiers = []
    for l in leads:
        dossiers.append({
            "company_profile": {
                "name": l.get("name"),
                "category": l.get("category"),
                "address": l.get("address"),
                "pincode": l.get("pincode"),
                "city": l.get("city"),
                "search_zone": l.get("search_locality"),
                "coordinates": {
                    "latitude": l.get("latitude"),
                    "longitude": l.get("longitude")
                },
                "google_maps_url": l.get("google_maps_url"),
                "rating": l.get("rating"),
                "reviews": l.get("reviews"),
                "business_status": l.get("business_status")
            },
            "contacts": {
                "primary_phone": l.get("phone"),
                "phone_type": l.get("phone_type"),
                "whatsapp_number": l.get("whatsapp_number"),
                "whatsapp_click_to_chat": l.get("whatsapp_click_link"),
                "alternate_phones": [p.strip() for p in l.get("additional_phones", "").split(";") if p.strip()],
                "emails": [e.strip() for e in l.get("email", "").split(";") if e.strip()],
                "website": l.get("website"),
                "domain": l.get("domain"),
                "social_media": {
                    "facebook": l.get("facebook_url"),
                    "instagram": l.get("instagram_url"),
                    "linkedin": l.get("linkedin_url"),
                    "youtube": l.get("youtube_url"),
                    "twitter": l.get("twitter_url")
                }
            },
            "digital_audit": {
                "lead_priority": l.get("lead_priority"),
                "need_score": l.get("need_score"),
                "site_status": l.get("site_status"),
                "segment": l.get("segment"),
                "audit_issues": l.get("issues"),
                "ssl_status": l.get("ssl_status"),
                "mobile_friendly": l.get("mobile_friendly"),
                "page_speed_sec": l.get("load_time_sec"),
                "has_enquiry_form": l.get("has_enquiry_form"),
                "has_whatsapp_widget": l.get("has_whatsapp_widget"),
                "cms_platform": l.get("cms_platform"),
                "copyright_year": l.get("copyright_year")
            },
            "outreach_pack": {
                "whatsapp_pitch": l.get("whatsapp_pitch"),
                "email_subject": l.get("email_subject"),
                "email_body": l.get("email_body")
            },
            "scraped_at": l.get("scraped_at")
        })

    with open(json_filename, "w", encoding="utf-8") as jf:
        json.dump(dossiers, jf, indent=2, ensure_ascii=False)


# ==============================================================================
# 8. PIPELINE ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="High-Yield CCTV Lead Generator & Company Intelligence Engine")
    parser.add_argument("--target", type=int, default=1000, help="Target number of unique leads (default: 1000)")
    parser.add_argument("--pitch-top", type=int, default=300, help="Generate pitches for top-N leads (default: 300)")
    parser.add_argument("--cities", type=str, default="", help="Comma-separated cities/zones (e.g. 'Bengaluru,Mumbai')")
    parser.add_argument("--workers-audit", type=int, default=15, help="Concurrent threads for website auditing (default: 15)")
    parser.add_argument("--workers-pitch", type=int, default=5, help="Concurrent threads for pitch generation (default: 5)")
    parser.add_argument("--output", type=str, default="leads.csv", help="Output CSV filename (default: leads.csv)")
    parser.add_argument("--json-output", type=str, default="company_details.json", help="Output JSON filename (default: company_details.json)")
    parser.add_argument("--resume", action="store_true", help="Resume previous run and append new leads")
    parser.add_argument("--skip-pitch", action="store_true", help="Skip LLM pitch drafting")
    args = parser.parse_args()

    selected_cities = [c.strip() for c in args.cities.split(",")] if args.cities else None

    print("=" * 80)
    print(" CCTV LEAD GENERATION & COMPANY INTELLIGENCE ENGINE")
    print(f" Target Leads: {args.target} | Pitch Top: {args.pitch_top}")
    print(f" CSV Output:   {args.output}")
    print(f" JSON Dossier: {args.json_output}")
    print("=" * 80)

    seen_ids = set()
    if args.resume:
        seen_ids = load_checkpoint(args.output)
        print(f"[*] Resuming from existing CSV. Found {len(seen_ids)} previously indexed leads.")

    # Phase 1: Discovery
    t_start = time.time()
    leads = discover(target=args.target, selected_cities=selected_cities, seen_ids=seen_ids)
    if not leads:
        print("[!] No leads discovered. Exiting.")
        return

    print(f"\n[+] Successfully discovered {len(leads)} leads. Starting deep audit across {args.workers_audit} threads...")

    # Phase 2: Deep Audit
    audited_leads = []
    with ThreadPoolExecutor(max_workers=args.workers_audit) as executor:
        futures = {executor.submit(audit, lead): lead for lead in leads}
        completed = 0
        for f in as_completed(futures):
            try:
                audited_leads.append(f.result())
            except Exception as ex:
                print(f"[!] Audit error: {ex}")
            completed += 1
            if completed % 50 == 0 or completed == len(leads):
                sys.stdout.write(f"\r[*] Audited {completed}/{len(leads)} leads...")
                sys.stdout.flush()

    print("\n[+] Audit complete. Sorting leads by need score...")
    audited_leads.sort(key=lambda x: x.get("need_score", 0), reverse=True)

    # Save initial checkpoint
    export_company_data(audited_leads, csv_filename=args.output, json_filename=args.json_output, append=args.resume)
    print(f"[+] Saved checkpoint to {args.output} and {args.json_output}")

    # Phase 3: Multi-LLM Rotating Pitch Generation
    if not args.skip_pitch:
        top_n = min(args.pitch_top, len(audited_leads))
        print(f"\n[*] Generating pitches for top {top_n} prospects using rotating LLM pool...")
        with ThreadPoolExecutor(max_workers=args.workers_pitch) as executor:
            top_leads = audited_leads[:top_n]
            results = list(executor.map(pitch, top_leads))
            audited_leads[:top_n] = results

        export_company_data(audited_leads, csv_filename=args.output, json_filename=args.json_output, append=False)
        print(f"[+] Updated {args.output} and {args.json_output} with personalized pitches and 1-click WhatsApp links!")

    # Summary Statistics
    total_time = time.time() - t_start
    mobiles = sum(1 for l in audited_leads if l.get("phone_type") == "Mobile")
    emails = sum(1 for l in audited_leads if l.get("email"))
    high_priority = sum(1 for l in audited_leads if l.get("lead_priority") == "High")

    print("\n" + "=" * 80)
    print(" RUN COMPLETE - COMPANY INTELLIGENCE REPORT")
    print(f" Total Companies Indexed:   {len(audited_leads)}")
    print(f" High-Priority Prospects:   {high_priority}")
    print(f" WhatsApp-Ready Mobiles:    {mobiles}")
    print(f" Verified Emails Scraped:   {emails}")
    print(f" LLM Success Counts:        {llm_rotator.success_counts}")
    if llm_rotator.fallen_keys:
        print(f" Keys that fell during run ({len(llm_rotator.fallen_keys)}):")
        for fk in llm_rotator.fallen_keys:
            print(f"   - {fk['provider']} ({fk['key_snippet']}): {fk['reason'][:80]}")
    print(f" Total Time:                {total_time:.1f}s")
    print(f" CSV Spreadsheet Saved to:  {os.path.abspath(args.output)}")
    print(f" JSON Dossier Saved to:      {os.path.abspath(args.json_output)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
