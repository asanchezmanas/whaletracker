"""
SEC EDGAR Form 4 Connector

Fetches insider transaction filings (Form 4) from SEC EDGAR.
Covers ALL insiders: CEOs, CFOs, directors, 10%+ owners.

SEC EDGAR is free, rate-limited to 10 requests/second.
Uses the full-text search index and XBRL/XML parsing.

Reference: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany
"""

import httpx
import pandas as pd
import xml.etree.ElementTree as ET
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from ..utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


# SEC requires a User-Agent identifying the requester
SEC_USER_AGENT = "WhaleTracker QDN research@whaletracker.dev"
SEC_BASE_URL = "https://efts.sec.gov/LATEST"
SEC_EDGAR_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
SEC_FULL_INDEX = "https://www.sec.gov/Archives/edgar/full-index"


class SECConnector:
    """
    Fetches SEC Form 4 insider transaction filings.
    
    Two approaches:
    1. EDGAR Full-Text Search (EFTS) — recent filings, fast
    2. Full Index — historical bulk download
    
    Rate limit: 10 req/sec (SEC enforced)
    """

    def __init__(self, user_agent: str = SEC_USER_AGENT):
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self._sec_limiter = RateLimiter.get_limiter("sec", 10.0)
        self._client = None  # Lazy-initialized persistent client

    def _get_client(self) -> httpx.Client:
        """Get or create a persistent HTTP client (reuses SSL context)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=30.0,
                headers=self.headers,
                follow_redirects=True,
            )
        return self._client

    def _rate_limit(self):
        """Enforce SEC's 10 req/sec rate limit (Shared)."""
        self._sec_limiter.wait()

    def fetch_recent_filings(
        self,
        days_back: int = 30,
        form_type: str = "4",
    ) -> pd.DataFrame:
        """
        Fetch recent Form 4 filings via EDGAR Full-Text Search.
        Best for recent data (< 1 year).
        """
        date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        date_to = datetime.now().strftime("%Y-%m-%d")

        all_filings = []
        start = 0
        page_size = 100

        logger.info(f"Fetching Form {form_type} filings from {date_from} to {date_to}")

        while True:
            self._rate_limit()

            url = f"{SEC_BASE_URL}/search-index"
            params = {
                "q": f'"form-type":"{form_type}"',
                "dateRange": "custom",
                "startdt": date_from,
                "enddt": date_to,
                "from": start,
                "size": page_size,
            }

            try:
                client = self._get_client()
                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                hits = data.get("hits", {}).get("hits", [])
                if not hits:
                    break

                for hit in hits:
                    source = hit.get("_source", {})
                    filing_url = source.get("file_url", "")
                    if filing_url:
                        all_filings.append({
                            "filing_url": filing_url,
                            "filing_date": source.get("file_date"),
                            "cik": source.get("cik"),
                            "company_name": source.get("display_names", [None])[0],
                        })

                start += page_size
                if start >= data.get("hits", {}).get("total", {}).get("value", 0):
                    break

            except Exception as e:
                logger.error(f"EFTS search failed: {e}")
                break

        return self._process_filing_list(all_filings)

    def fetch_historical_index(
        self,
        year: int,
        quarter: int,
        form_type: str = "4",
    ) -> pd.DataFrame:
        """
        Fetch filings by parsing the SEC's master.idx for a specific quarter.
        REQUIRED for 10-year backfill (EFTS is limited).
        """
        self._rate_limit()
        url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"
        
        logger.info(f"Fetching Master Index: {year} Q{quarter}")
        try:
            client = self._get_client()
            r = client.get(url)
            r.raise_for_status()
            
            # Index format: CIK|Name|Form|Date|File
            lines = r.text.split('\n')
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith('---'):
                    data_start = i + 1
                    break
            
            filing_list = []
            for line in lines[data_start:]:
                parts = line.split('|')
                if len(parts) < 5: continue
                
                if parts[2] == form_type:
                    filing_list.append({
                        "cik": parts[0],
                        "company_name": parts[1],
                        "form_type": parts[2],
                        "filing_date": parts[3],
                        "filing_url": f"https://www.sec.gov/Archives/{parts[4]}"
                    })
            
            logger.info(f"Found {len(filing_list)} {form_type} filings in {year} Q{quarter}")
            return self._process_filing_list(filing_list)
            
        except Exception as e:
            logger.error(f"Master index fetch failed for {year} Q{quarter}: {e}")
            return pd.DataFrame()

    def _process_filing_list(self, filing_list: List[Dict]) -> pd.DataFrame:
        """Helper to parse a list of filing metadata into transactions."""
        transactions = []
        for i, filing in enumerate(filing_list):
            if i % 100 == 0 and i > 0:
                logger.info(f"Parsing filing {i}/{len(filing_list)}")
            
            parsed = self._parse_form4_xml(filing)
            if parsed:
                transactions.extend(parsed)
        
        if not transactions:
            return pd.DataFrame()
            
        df = pd.DataFrame(transactions)
        return self._clean_data(df)

    def fetch_by_cik(
        self, cik: str, form_type: str = "4", max_filings: int = 40
    ) -> pd.DataFrame:
        """
        Fetch Form 4 filings for a specific company (by CIK).
        
        Args:
            cik: SEC Central Index Key (e.g., "0000320193" for Apple)
            form_type: SEC form type
            max_filings: Maximum number of XML filings to parse (default 40)
        
        Returns:
            DataFrame of insider transactions
        """
        self._rate_limit()

        # Pad CIK to 10 digits
        cik_padded = cik.zfill(10)

        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"

        try:
            client = self._get_client()
            response = client.get(url)
            response.raise_for_status()
            data = response.json()

            # Extract recent filings
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])

            # Collect Form 4 filing URLs
            filing_list = []
            for form, date, accession, doc in zip(forms, dates, accessions, primary_docs):
                if form != form_type:
                    continue
                accession_clean = accession.replace("-", "")
                filing_url = (
                    f"https://data.sec.gov/Archives/edgar/data/"
                    f"{cik_padded}/{accession_clean}/{doc}"
                )
                filing_list.append({
                    "filing_url": filing_url,
                    "filing_date": date,
                    "cik": cik,
                    "company_name": data.get("name"),
                })

            # Limit number of XML parses
            if len(filing_list) > max_filings:
                logger.info(
                    f"CIK {cik}: {len(filing_list)} Form 4s found, "
                    f"parsing most recent {max_filings}"
                )
                filing_list = filing_list[:max_filings]

            transactions = []
            for i, filing in enumerate(filing_list):
                parsed = self._parse_form4_xml(filing)
                if parsed:
                    transactions.extend(parsed)

            if not transactions:
                return pd.DataFrame()

            df = pd.DataFrame(transactions)
            return self._clean_data(df)

        except Exception as e:
            logger.error(f"Failed to fetch CIK {cik}: {e}")
            return pd.DataFrame()

    def _parse_form4_xml(self, filing_info: Dict) -> List[Dict]:
        """
        Parse a Form 4 XML filing and extract transactions.
        
        Form 4 XML structure:
        <ownershipDocument>
          <issuer><issuerCik/><issuerName/><issuerTradingSymbol/></issuer>
          <reportingOwner>
            <reportingOwnerId><rptOwnerName/></reportingOwnerId>
            <reportingOwnerRelationship/>
          </reportingOwner>
          <nonDerivativeTable>
            <nonDerivativeTransaction>
              <transactionDate><value/></transactionDate>
              <transactionCoding><transactionCode/></transactionCoding>
              <transactionAmounts>
                <transactionShares><value/></transactionShares>
                <transactionPricePerShare><value/></transactionPricePerShare>
              </transactionAmounts>
              <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value/></sharesOwnedFollowingTransaction>
              </postTransactionAmounts>
            </nonDerivativeTransaction>
          </nonDerivativeTable>
        </ownershipDocument>
        """
        self._rate_limit()

        url = filing_info.get("filing_url", "")
        if not url:
            return []

        try:
            client = self._get_client()
            response = client.get(url)
            response.raise_for_status()
            content = response.text

            root = ET.fromstring(content)

            # Namespace handling
            ns = ""
            if root.tag.startswith("{"):
                ns = root.tag.split("}")[0] + "}"

            # Issuer info
            ticker = self._get_text(root, f"{ns}issuer/{ns}issuerTradingSymbol")
            company_name = self._get_text(root, f"{ns}issuer/{ns}issuerName")
            cik = self._get_text(root, f"{ns}issuer/{ns}issuerCik") or filing_info.get("cik")

            # Owner info
            owner_name = self._get_text(
                root, f"{ns}reportingOwner/{ns}reportingOwnerId/{ns}rptOwnerName"
            )
            
            # Owner relationship
            rel = root.find(f"{ns}reportingOwner/{ns}reportingOwnerRelationship")
            title = self._extract_title(rel, ns) if rel is not None else None

            # Parse non-derivative transactions
            transactions = []
            for txn in root.findall(
                f"{ns}nonDerivativeTable/{ns}nonDerivativeTransaction"
            ):
                parsed = self._parse_transaction(txn, ns)
                if parsed:
                    parsed.update({
                        "ticker": ticker,
                        "company_name": company_name,
                        "cik": cik,
                        "insider_name": owner_name,
                        "insider_title": title,
                        "filing_date": filing_info.get("filing_date"),
                        "source": "sec_form4",
                    })
                    transactions.append(parsed)

            return transactions

        except ET.ParseError:
            logger.debug(f"XML parse error for {url}")
            return []
        except Exception as e:
            logger.debug(f"Failed to parse {url}: {e}")
            return []

    def _parse_transaction(self, txn, ns: str) -> Optional[Dict]:
        """Parse a single non-derivative transaction element."""
        try:
            date_str = self._get_text(txn, f"{ns}transactionDate/{ns}value")
            code = self._get_text(txn, f"{ns}transactionCoding/{ns}transactionCode")

            shares_str = self._get_text(
                txn, f"{ns}transactionAmounts/{ns}transactionShares/{ns}value"
            )
            price_str = self._get_text(
                txn,
                f"{ns}transactionAmounts/{ns}transactionPricePerShare/{ns}value",
            )
            ownership_str = self._get_text(
                txn,
                f"{ns}postTransactionAmounts/{ns}sharesOwnedFollowingTransaction/{ns}value",
            )

            shares = int(float(shares_str)) if shares_str else None
            price = float(price_str) if price_str else None
            ownership = int(float(ownership_str)) if ownership_str else None

            if not date_str or not code:
                return None

            return {
                "transaction_date": date_str,
                "transaction_code": code,
                "shares": shares,
                "price": price,
                "value": (shares * price) if shares and price else None,
                "ownership_after": ownership,
            }
        except (ValueError, TypeError) as e:
            logger.debug(f"Transaction parse error: {e}")
            return None

    def _extract_title(self, rel_element, ns: str) -> str:
        """Extract insider title from relationship element."""
        titles = []
        if self._get_text(rel_element, f"{ns}isDirector") == "1":
            titles.append("Director")
        if self._get_text(rel_element, f"{ns}isOfficer") == "1":
            officer_title = self._get_text(rel_element, f"{ns}officerTitle")
            titles.append(officer_title or "Officer")
        if self._get_text(rel_element, f"{ns}isTenPercentOwner") == "1":
            titles.append("10% Owner")
        if self._get_text(rel_element, f"{ns}isOther") == "1":
            other_text = self._get_text(rel_element, f"{ns}otherText")
            titles.append(other_text or "Other")
        return ", ".join(titles) if titles else "Unknown"

    @staticmethod
    def _get_text(element, path: str) -> Optional[str]:
        """Safely extract text from XML element."""
        el = element.find(path)
        if el is not None and el.text:
            return el.text.strip()
        return None

    @staticmethod
    def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize transaction data."""
        if df.empty:
            return df

        # Parse dates
        for col in ["transaction_date", "filing_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Drop rows without essential fields
        df = df.dropna(subset=["ticker", "insider_name", "transaction_date"])

        # Uppercase ticker
        df["ticker"] = df["ticker"].str.upper().str.strip()

        # Filter: only P (Purchase) and S (Sale) codes for clean signal
        df = df[df["transaction_code"].isin(["P", "S", "A", "M", "G"])].copy()

        # Sort by date
        df = df.sort_values("transaction_date").reset_index(drop=True)

        return df
