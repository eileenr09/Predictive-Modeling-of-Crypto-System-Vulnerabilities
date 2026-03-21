"""
data_ingestion.py
=================
Loads and harmonises all five real datasets uploaded for this project:

  Source A – Balloon / Information is Beautiful (IIB)
             Balloon_Race_Data_Breaches_-_LATEST_-_breaches.csv
             ~440 high-profile global breaches 2004-2022, includes sector & method.

  Source B – HHS / OCR Healthcare Breaches
             Cyber_Security_Breaches.csv
             ~1 285 US healthcare breaches 2009-present, fine-grained breach type.

  Source C – Wikipedia/Statista breach list (DataBreachN)
             Data_BreachesN_new.csv
             ~352 international breaches, country, org-type, method.

  Source D – European breach registry (DataBreachEN)
             Data_Breaches_EN_V2_2004_2017_20180220.csv
             ~277 breaches, European focus 2004-2017, sector & method.

  Source E – Wikipedia extended list (df_1)
             df_1.csv
             ~352 breaches with entity name, year, records, method.

  Source F – CSIS Significant Cyber Incidents PDF
             260306_Cyber_Events.pdf
             State-sponsored / major incidents 2006-present (text-mined).

Each source is loaded into a canonical schema:
  date, year, entity, sector, method_raw, method_category,
  records_affected, is_crypto, source_id
"""

import re
import logging
import os
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

# ── Crypto-related keywords ─────────────────────────────────────────────────
CRYPTO_KEYWORDS = [
    "crypto", "bitcoin", "btc", "ethereum", "eth", "blockchain", "defi",
    "nft", "exchange", "wallet", "binance", "coinbase", "bybit", "kraken",
    "protocol", "token", "web3", "dex", "cex", "uniswap", "wazirx",
    "ftx", "bitmart", "bitmex", "bitfinex", "mt.gox", "mtgox",
    "digital asset", "altcoin",
]

# ── Method normalisation map ────────────────────────────────────────────────
METHOD_MAP = {
    "hacked"               : "hacking",
    "hack"                 : "hacking",
    "hacking"              : "hacking",
    "ransomware"           : "malware",
    "malware"              : "malware",
    "virus"                : "malware",
    "phish"                : "phishing",
    "social engineering"   : "phishing",
    "spear"                : "phishing",
    "inside job"           : "insider",
    "insider"              : "insider",
    "employee"             : "insider",
    "poor security"        : "poor_security",
    "accidentally published": "poor_security",
    "misconfigur"          : "poor_security",
    "oops"                 : "poor_security",
    "lost device"          : "lost_device",
    "lost / stolen"        : "lost_device",
    "theft"                : "lost_device",
    "stolen"               : "lost_device",
    "unauthorized"         : "unauthorized_access",
    "smart contract"       : "smart_contract_exploit",
    "protocol exploit"     : "smart_contract_exploit",
    "flash loan"           : "smart_contract_exploit",
    "ddos"                 : "ddos",
    "denial of service"    : "ddos",
    "supply chain"         : "supply_chain",
    "third.party"          : "supply_chain",
}

SECTOR_MAP = {
    "web"         : "tech_web",
    "tech"        : "tech_web",
    "app"         : "tech_web",
    "healthcare"  : "healthcare",
    "health"      : "healthcare",
    "medical"     : "healthcare",
    "financial"   : "financial",
    "finance"     : "financial",
    "banking"     : "financial",
    "government"  : "government",
    "gov"         : "government",
    "military"    : "government",
    "retail"      : "retail",
    "shopping"    : "retail",
    "social"      : "social_media",
    "gaming"      : "gaming",
    "energy"      : "energy",
    "transport"   : "transport",
    "telecom"     : "telecom",
    "academic"    : "academic",
    "media"       : "media",
    "legal"       : "legal",
}


def categorise_method(raw: str) -> str:
    if not isinstance(raw, str):
        return "unknown"
    r = raw.lower()
    for kw, cat in METHOD_MAP.items():
        if kw in r:
            return cat
    return "other"


def categorise_sector(raw: str) -> str:
    if not isinstance(raw, str):
        return "unknown"
    r = raw.lower()
    for kw, cat in SECTOR_MAP.items():
        if kw in r:
            return cat
    return "other"


def flag_crypto(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()
    return any(k in t for k in CRYPTO_KEYWORDS)


def _clean_records(val) -> float:
    """Parse messy record-count strings like '15,000,000', '1,37e+09', '3m'."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    # Remove footnote references like [5][6]
    s = re.sub(r"\[.*?\]", "", s).strip()

    # Detect European format: digits.digits.digits (e.g. 15.000.000)
    if re.match(r"^\d{1,3}(\.\d{3})+$", s):
        # European thousands separator — remove dots
        s = s.replace(".", "")
    else:
        # Assume comma is either thousands separator or decimal
        # If it matches 1,37e+09 style (European decimal comma), swap
        if re.match(r"^[\d]+,\d{1,2}(e[+\-]\d+)?$", s):
            s = s.replace(",", ".")
        else:
            # Normal comma thousands separator
            s = s.replace(",", "")

    s = s.replace(" ", "")
    try:
        return float(s)
    except ValueError:
        # handle strings like '3m' -> 3_000_000
        m = re.match(r"([\d.]+)\s*([kmb]?)", s)
        if m:
            try:
                n      = float(m.group(1))
                suffix = m.group(2)
                mult   = {"k": 1e3, "m": 1e6, "b": 1e9}.get(suffix, 1)
                return n * mult
            except ValueError:
                pass
    return np.nan


CANONICAL_COLS = [
    "year", "entity", "sector_raw", "sector", "method_raw",
    "method_category", "records_affected", "is_crypto", "source_id",
]


# ── Source A: IIB / Balloon ──────────────────────────────────────────────────
def load_iib(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", header=0, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # Skip the two header/legend rows (rows 0 and sometimes 1)
    df = df[df["organisation"].notna()].copy()
    df = df[~df["organisation"].str.startswith("visualisation", na=True)]

    records = []
    for _, row in df.iterrows():
        entity    = str(row.get("organisation", "")).strip()
        year_raw  = row.get("year", row.get("year   ", np.nan))
        try:
            year = int(float(str(year_raw).strip()))
        except (ValueError, TypeError):
            continue
        if year < 2000 or year > 2026:
            continue

        sector_r  = str(row.get("sector", ""))
        method_r  = str(row.get("method", ""))
        story     = str(row.get("story", ""))
        rec_raw   = row.get("records_lost", np.nan)

        records.append({
            "year"             : year,
            "entity"           : entity,
            "sector_raw"       : sector_r,
            "sector"           : categorise_sector(sector_r),
            "method_raw"       : method_r,
            "method_category"  : categorise_method(method_r),
            "records_affected" : _clean_records(rec_raw),
            "is_crypto"        : flag_crypto(entity + " " + story),
            "source_id"        : "IIB",
        })

    log.info(f"  IIB: {len(records)} rows loaded from {path}")
    return pd.DataFrame(records, columns=CANONICAL_COLS)


# ── Source B: HHS / Cyber Security Breaches ─────────────────────────────────
def load_hhs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    HHS_METHOD_MAP = {
        "theft"           : "lost_device",
        "loss"            : "lost_device",
        "hacking"         : "hacking",
        "it incident"     : "hacking",
        "unauthorized access": "unauthorized_access",
        "improper disposal": "poor_security",
        "other"           : "other",
    }

    records = []
    for _, row in df.iterrows():
        date_raw = row.get("date_of_breach", "")
        try:
            dt   = pd.to_datetime(date_raw, errors="coerce")
            year = int(dt.year) if pd.notna(dt) else None
        except Exception:
            year = None
        if year is None or year < 2000:
            continue

        entity   = str(row.get("name_of_covered_entity", ""))
        method_r = str(row.get("type_of_breach", ""))
        mc = "other"
        for kw, cat in HHS_METHOD_MAP.items():
            if kw.lower() in method_r.lower():
                mc = cat
                break

        summary = str(row.get("summary", ""))
        records.append({
            "year"             : year,
            "entity"           : entity,
            "sector_raw"       : "healthcare",
            "sector"           : "healthcare",
            "method_raw"       : method_r,
            "method_category"  : mc,
            "records_affected" : _clean_records(row.get("individuals_affected")),
            "is_crypto"        : flag_crypto(entity + " " + summary),
            "source_id"        : "HHS",
        })

    log.info(f"  HHS: {len(records)} rows loaded from {path}")
    return pd.DataFrame(records, columns=CANONICAL_COLS)


# ── Source C: DataBreachN (country + org-type) ───────────────────────────────
def load_dbn(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    records = []
    for _, row in df.iterrows():
        try:
            year = int(float(str(row.get("year", "")).strip()))
        except (ValueError, TypeError):
            continue
        if year < 2000 or year > 2026:
            continue

        entity   = str(row.get("country", ""))
        sector_r = str(row.get("organization_type", ""))
        method_r = str(row.get("method", ""))

        records.append({
            "year"             : year,
            "entity"           : entity,
            "sector_raw"       : sector_r,
            "sector"           : categorise_sector(sector_r),
            "method_raw"       : method_r,
            "method_category"  : categorise_method(method_r),
            "records_affected" : _clean_records(row.get("records")),
            "is_crypto"        : flag_crypto(entity + " " + sector_r + " " + method_r),
            "source_id"        : "DBN",
        })

    log.info(f"  DBN: {len(records)} rows loaded from {path}")
    return pd.DataFrame(records, columns=CANONICAL_COLS)


# ── Source D: DataBreachEN (European registry) ───────────────────────────────
def load_dben(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="latin-1", low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]

    records = []
    for _, row in df.iterrows():
        try:
            year = int(float(str(row.get("year", "")).strip()))
        except (ValueError, TypeError):
            continue
        if year < 2000 or year > 2026:
            continue

        entity   = str(row.get("entity", ""))
        sector_r = str(row.get("sector", ""))
        method_r = str(row.get("method_of_leak", ""))
        story    = str(row.get("story", ""))

        records.append({
            "year"             : year,
            "entity"           : entity,
            "sector_raw"       : sector_r,
            "sector"           : categorise_sector(sector_r),
            "method_raw"       : method_r,
            "method_category"  : categorise_method(method_r),
            "records_affected" : _clean_records(row.get("records_lost")),
            "is_crypto"        : flag_crypto(entity + " " + story),
            "source_id"        : "DBEN",
        })

    log.info(f"  DBEN: {len(records)} rows loaded from {path}")
    return pd.DataFrame(records, columns=CANONICAL_COLS)


# ── Source E: df_1 (Wikipedia extended) ─────────────────────────────────────
def load_df1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    records = []
    for _, row in df.iterrows():
        try:
            year = int(float(str(row.get("year", "")).strip()))
        except (ValueError, TypeError):
            continue
        if year < 2000 or year > 2026:
            continue

        entity   = str(row.get("entity", ""))
        sector_r = str(row.get("organization_type", ""))
        method_r = str(row.get("method", ""))

        records.append({
            "year"             : year,
            "entity"           : entity,
            "sector_raw"       : sector_r,
            "sector"           : categorise_sector(sector_r),
            "method_raw"       : method_r,
            "method_category"  : categorise_method(method_r),
            "records_affected" : _clean_records(row.get("records")),
            "is_crypto"        : flag_crypto(entity + " " + sector_r + " " + method_r),
            "source_id"        : "DF1",
        })

    log.info(f"  DF1: {len(records)} rows loaded from {path}")
    return pd.DataFrame(records, columns=CANONICAL_COLS)


# ── Source F: CSIS PDF ───────────────────────────────────────────────────────
def _extract_pdf_text(path: str) -> str:
    """
    Extract text from a PDF using multiple methods in order of preference:
      1. pypdf  (pure Python, pip install pypdf)
      2. pdfplumber (pure Python, pip install pdfplumber)
      3. pdftotext / poppler (system tool, Windows needs explicit encoding)
    Always returns a str — never None.
    """
    # Method 1: pypdf
    try:
        import pypdf as _pypdf
        text_parts = []
        with _pypdf.PdfReader(path) as reader:
            for page in reader.pages:
                extracted = page.extract_text()
                text_parts.append(extracted if isinstance(extracted, str) else "")
        text = "\n".join(text_parts)
        if text.strip():
            log.info("  CSIS PDF: extracted via pypdf")
            return text
        log.warning("  pypdf returned empty text — trying next method")
    except ImportError:
        log.warning("  pypdf not installed — run: pip install pypdf")
    except Exception as e:
        log.warning(f"  pypdf extraction error: {e}")

    # Method 2: pdfplumber
    try:
        import pdfplumber as _pdfplumber
        text_parts = []
        with _pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                text_parts.append(extracted if isinstance(extracted, str) else "")
        text = "\n".join(text_parts)
        if text.strip():
            log.info("  CSIS PDF: extracted via pdfplumber")
            return text
    except ImportError:
        pass
    except Exception as e:
        log.warning(f"  pdfplumber error: {e}")

    # Method 3: pdftotext (with explicit UTF-8 encoding for Windows)
    try:
        import subprocess
        result = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", path, "-"],
            capture_output=True, timeout=30,
        )
        # Decode bytes explicitly as UTF-8, ignoring unmappable chars
        stdout_text = result.stdout.decode("utf-8", errors="ignore")
        if result.returncode == 0 and stdout_text.strip():
            log.info("  CSIS PDF: extracted via pdftotext")
            return stdout_text
    except FileNotFoundError:
        pass  # pdftotext not installed
    except Exception as e:
        log.warning(f"  pdftotext error: {e}")

    log.warning(
        "  CSIS PDF: all extraction methods failed.\n"
        "  Fix: run  pip install pypdf  then restart the kernel.\n"
        "  The pipeline will continue with the 5 CSV datasets only."
    )
    return ""


def load_csis_pdf(path: str) -> pd.DataFrame:
    """
    Text-mines the CSIS 'Significant Cyber Incidents' PDF.
    Extracts month/year and searches for crypto keywords.
    Uses pure-Python pypdf first — no poppler/pdftotext required.
    """
    text = _extract_pdf_text(path)
    # Guard: ensure text is always a non-None string before regex
    if not isinstance(text, str):
        text = ""
    if not text.strip():
        log.info("  CSIS PDF: skipped (no text extracted)")
        return pd.DataFrame(columns=CANONICAL_COLS)

    # Parse paragraphs: each entry starts with "Month YYYY:"
    paragraphs = re.split(r"\n(?=[A-Z][a-z]+ \d{4}:)", text)
    records = []
    for para in paragraphs:
        m = re.match(r"([A-Z][a-z]+ (\d{4})):", para)
        if not m:
            continue
        year = int(m.group(2))
        if year < 2006 or year > 2026:
            continue

        # Look for dollar amounts to estimate records/loss
        usd_match = re.search(r"\$(\d[\d,.]*)\s*(million|billion|thousand)?", para, re.IGNORECASE)
        loss_usd = np.nan
        if usd_match:
            n = _clean_records(usd_match.group(1))
            mult = {"million": 1e6, "billion": 1e9, "thousand": 1e3}.get(
                (usd_match.group(2) or "").lower(), 1
            )
            loss_usd = n * mult if pd.notna(n) else np.nan

        sector_r = "government"
        for kw in ["bank", "finance", "financial"]:
            if kw in para.lower():
                sector_r = "financial"
        for kw in ["hospital", "health", "medical"]:
            if kw in para.lower():
                sector_r = "healthcare"
        for kw in ["crypto", "bitcoin", "exchange", "wallet", "blockchain"]:
            if kw in para.lower():
                sector_r = "financial"

        # Method detection
        method_r = "unknown"
        for kw in ["ransomware", "phishing", "ddos", "supply chain",
                   "malware", "hack", "insider"]:
            if kw in para.lower():
                method_r = kw
                break

        records.append({
            "year"             : year,
            "entity"           : para[:60].strip(),
            "sector_raw"       : sector_r,
            "sector"           : categorise_sector(sector_r),
            "method_raw"       : method_r,
            "method_category"  : categorise_method(method_r),
            "records_affected" : loss_usd,
            "is_crypto"        : flag_crypto(para),
            "source_id"        : "CSIS",
        })

    log.info(f"  CSIS: {len(records)} incidents parsed from PDF")
    return pd.DataFrame(records, columns=CANONICAL_COLS)


# ── Master loader ─────────────────────────────────────────────────────────────
def load_all_datasets(data_dir: str = "data") -> pd.DataFrame:
    """
    Loads all six data sources, de-duplicates, and returns a unified DataFrame.
    """
    loaders = [
        (load_iib,  "Balloon_Race_Data_Breaches_-_LATEST_-_breaches.csv"),
        (load_hhs,  "Cyber_Security_Breaches.csv"),
        (load_dbn,  "Data_BreachesN_new.csv"),
        (load_dben, "Data_Breaches_EN_V2_2004_2017_20180220.csv"),
        (load_df1,  "df_1.csv"),
        (load_csis_pdf, "260306_Cyber_Events.pdf"),
    ]

    frames = []
    for loader_fn, fname in loaders:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            log.warning(f"  Missing: {fpath}")
            continue
        df = loader_fn(fpath)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "\n\n  No data files were found in: " + os.path.abspath(data_dir) +
            "\n  Please place the following files inside that folder:" +
            "\n    • Balloon_Race_Data_Breaches_-_LATEST_-_breaches.csv" +
            "\n    • Cyber_Security_Breaches.csv" +
            "\n    • Data_BreachesN_new.csv" +
            "\n    • Data_Breaches_EN_V2_2004_2017_20180220.csv" +
            "\n    • df_1.csv" +
            "\n    • 260306_Cyber_Events.pdf" +
            "\n  Then re-run:  python main.py --data_dir <path_to_folder>\n"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["year"].between(2004, 2026)]
    # Ensure is_crypto is always proper bool (concat can produce object dtype)
    combined["is_crypto"] = combined["is_crypto"].fillna(False).astype(bool)

    # Soft de-duplication: drop exact (entity, year, method_category) matches
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["entity", "year", "method_category"], keep="first"
    )
    log.info(
        f"  Combined: {len(combined)} unique records "
        f"(dropped {before - len(combined)} duplicates)"
    )
    return combined.reset_index(drop=True)
