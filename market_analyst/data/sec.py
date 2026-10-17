"""SEC EDGAR: ticker → CIK → latest 10-K plain text.

EDGAR rejects requests without a descriptive User-Agent (set SEC_USER_AGENT in .env)
and rate-limits to ~10 requests/second.
"""
import requests
import re
from bs4 import BeautifulSoup

from ..config import CACHE_DIR, SEC_USER_AGENT

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"


def _get(url: str) -> requests.Response:
    if not SEC_USER_AGENT:
        raise RuntimeError("Set SEC_USER_AGENT in .env (name + email)")
    resp = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=20)
    resp.raise_for_status()
    return resp


def ticker_to_cik(ticker: str) -> str:
    """TODO(3): load TICKER_MAP_URL, find the entry whose 'ticker' matches,
    and return its cik_str zero-padded to 10 digits (f"{cik:010d})."""
    data = _get(TICKER_MAP_URL).json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            return f"{entry['cik_str']:010d}"
    raise ValueError(f"No CIK found for ticker {ticker}")


def latest_10k_url(cik: str) -> str:
    """TODO(3): fetch SUBMISSIONS_URL. Under filings.recent, the lists 'form',
    'accessionNumber' and 'primaryDocument' line up by index — take the first
    index where form == '10-K'. Strip dashes from the accession number."""
    recent = _get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
    for form, accession, document in zip(
        recent["form"], recent["accessionNumber"], recent["primaryDocument"]
    ):
        if form == "10-K":
            return ARCHIVE_URL.format(
                cik_int=int(cik),
                accession=accession.replace("-", ""),
                document=document
            )
    raise ValueError(f"No 10-K found for CIK {cik}")

def fetch_10k_text(ticker: str) -> str:
    """Return the most recent 10-K as plain text, cached on disk.

    TODO(3): chain the helpers above, strip the HTML with BeautifulSoup
    (get_text(" ")), collapse whitespace, and cache the result at
    CACHE_DIR / f"{ticker}_10k.txt". Caching is what makes repeat runs fast.
    """
    cache_file = CACHE_DIR / f"{ticker.upper()}_10k.txt"
    if cache_file.exists():
        return cache_file.read_text()

    url = latest_10k_url(ticker_to_cik(ticker))
    html = _get(url).text
    soup = BeautifulSoup(html, "html.parser")
    # Inline-XBRL filings start with a hidden <ix:header> of machine-readable tags.
    for tag in soup.find_all("ix:header"):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(text)
    return text
