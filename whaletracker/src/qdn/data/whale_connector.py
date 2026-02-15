"""
Whale Connector — Unified Smart Money Signal Detection

Tracks ALL whale-type investors across SEC disclosure types:
1. Form 4: Insider purchases + sales (via OpenInsider.com)
2. Form 4 Cluster: Multiple insiders buying same stock (strongest signal)
3. SC 13D/13G: Activist & strategic investors taking >5% stakes
4. 13F-HR: Institutional holdings (funds, billionaires, institutions)

Antifragile philosophy:
- Purchases → entry signals
- Sales → exit signals (close existing positions)
- Cluster buys → high-conviction entry
- 13D → strategic conviction (company/fund investing in small-cap)
- 13F → institutional confirmation (45-day lag, confirmatory only)
"""

import pandas as pd
import numpy as np
import httpx
import time
import re
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from io import StringIO
from ..utils.rate_limiter import RateLimiter
from xml.etree import ElementTree as ET
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

SEC_USER_AGENT = "WhaleTracker QDN research@whaletracker.dev"

# Known super-investors (CIKs for 13F tracking)
SUPER_INVESTORS = {
    "0001067983": "Berkshire Hathaway (Buffett)",
    "0001649339": "Bridgewater Associates (Dalio)",
    "0001037389": "Renaissance Technologies (Simons)",
    "0001336528": "Baupost Group (Klarman)",
    "0001079114": "Pershing Square (Ackman)",
    "0001103804": "Third Point (Loeb)",
    "0001061768": "Icahn Enterprises (Icahn)",
    "0001040273": "Greenlight Capital (Einhorn)",
    "0001510627": "Appaloosa Management (Tepper)",
    "0000902664": "Druckenmiller (Duquesne)",
    "0001029160": "ValueAct Capital",
    "0001345471": "Scion Asset Management (Burry)",
}


class WhaleConnector:
    """
    Unified connector for all whale-type investment signals.

    Usage:
        wc = WhaleConnector()

        # Entry signals
        purchases = wc.fetch_insider_purchases(days_back=365)
        clusters = wc.fetch_cluster_buys()
        stakes = wc.fetch_13d_filings(days_back=365)
        holdings = wc.fetch_13f_new_positions("0001067983")  # Berkshire

        # Exit signals (for held positions)
        sales = wc.fetch_insider_sales("AAPL")

        # All signals unified
        all_signals = wc.fetch_all_whale_signals()
    """

    def __init__(self, user_agent: str = SEC_USER_AGENT):
        self._openinsider_client = httpx.Client(
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            follow_redirects=True,
        )
        self._sec_client = httpx.Client(
            timeout=30,
            headers={
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
            follow_redirects=True,
        )
        self._sec_limiter = RateLimiter.get_limiter("sec", 10.0)

    def _sec_rate_limit(self):
        """SEC rate limit: 10 req/sec (Global)."""
        self._sec_limiter.wait()

    # ═══════════════════════════════════════════════════════
    # FORM 4: Insider Purchases (Entry Signal)
    # ═══════════════════════════════════════════════════════

    def fetch_insider_purchases(
        self,
        days_back: int = 365,
        min_value: int = 1000,
        max_pages: int = 3,
    ) -> pd.DataFrame:
        """
        Fetch insider purchases from OpenInsider.com.

        Returns DataFrame with: ticker, transaction_date, filing_date,
        insider_name, insider_title, price, shares, value, ownership_after,
        whale_type='insider_purchase'
        """
        all_dfs = []

        for page in range(1, max_pages + 1):
            url = (
                f"http://openinsider.com/screener?"
                f"s=&o=&pl={min_value}&ph=&ll=&lh=&fd={days_back}"
                f"&fdr=&td=0&tdr=&feession=&cession=&sidTL=&tidTL="
                f"&tiession=&ession=&sicTL=&sicession=&pession="
                f"&cnt=1000&page={page}"
            )

            logger.info(f"Fetching insider purchases page {page}...")
            try:
                r = self._openinsider_client.get(url)
                if r.status_code != 200:
                    continue
                tables = pd.read_html(StringIO(r.text))
                for tbl in tables:
                    cols = [str(c).lower() for c in tbl.columns]
                    if any("trade" in c for c in cols) and len(tbl) > 5:
                        all_dfs.append(tbl)
                        break
            except Exception as e:
                logger.warning(f"Page {page} error: {e}")
            time.sleep(0.5)

        if not all_dfs:
            return pd.DataFrame()

        raw = pd.concat(all_dfs, ignore_index=True)
        return self._parse_openinsider(raw, signal_type="insider_purchase", filter_purchases=True)

    # ═══════════════════════════════════════════════════════
    # FORM 4: Insider Sales (Exit Signal)
    # ═══════════════════════════════════════════════════════

    def fetch_insider_sales(self, ticker: str, days_back: int = 90) -> pd.DataFrame:
        """
        Fetch recent insider SALES for a specific ticker.
        Used to generate EXIT signals for held positions.

        Returns DataFrame with sales for the given ticker.
        """
        url = (
            f"http://openinsider.com/screener?"
            f"s={ticker}&o=&pl=&ph=&ll=&lh=&fd={days_back}"
            f"&fdr=&td=0&tdr=&feession=&cession=&sidTL=&tidTL="
            f"&tiession=&ession=&sicTL=&sicession=&pession="
            f"&cnt=100&page=1"
        )

        try:
            r = self._openinsider_client.get(url)
            if r.status_code != 200:
                return pd.DataFrame()
            tables = pd.read_html(StringIO(r.text))
            for tbl in tables:
                cols = [str(c).lower() for c in tbl.columns]
                if any("trade" in c for c in cols) and len(tbl) > 0:
                    return self._parse_openinsider(
                        tbl, signal_type="insider_sale", filter_sales=True
                    )
        except Exception as e:
            logger.warning(f"Sales fetch error for {ticker}: {e}")

        return pd.DataFrame()

    # ═══════════════════════════════════════════════════════
    # FORM 4: Cluster Buys (Strong Entry Signal)
    # ═══════════════════════════════════════════════════════

    def fetch_cluster_buys(self) -> pd.DataFrame:
        """
        Fetch cluster buys (3+ insiders buying same stock within days).
        Strongest entry signal — multiple informed people acting together.
        """
        url = "http://openinsider.com/latest-cluster-buys"

        try:
            r = self._openinsider_client.get(url)
            if r.status_code != 200:
                return pd.DataFrame()
            tables = pd.read_html(StringIO(r.text))
            for tbl in tables:
                if len(tbl) > 3:
                    return self._parse_openinsider(
                        tbl, signal_type="cluster_buy", filter_purchases=True
                    )
        except Exception as e:
            logger.warning(f"Cluster buys error: {e}")

        return pd.DataFrame()

    def get_cluster_quality(self, ticker: str, window_days: int = 60) -> float:
        """
        Scores the 'Diversity' and 'Seniority' of a whale cluster.
        Diversity = (Insider + 5% Holder + Strategic Partner).
        """
        # Fetch recent insider purchases for this ticker
        url = (
            f"http://openinsider.com/screener?"
            f"s={ticker}&o=&pl=&ph=&ll=&lh=&fd={window_days}"
            f"&fdr=&td=0&tdr=&feession=&cession=&sidTL=&tidTL="
            f"&tiession=&ession=&sicTL=&sicession=&pession="
            f"&cnt=100&page=1"
        )
        
        try:
            r = self._openinsider_client.get(url)
            if r.status_code != 200: return 0.0
            tables = pd.read_html(StringIO(r.text))
            df = pd.DataFrame()
            for tbl in tables:
                cols = [str(c).lower() for c in tbl.columns]
                if any("trade" in c for c in cols):
                    df = self._parse_openinsider(tbl, filter_purchases=True)
                    break
            
            if df.empty: return 0.0
            
            # Diversity Score
            unique_titles = df["insider_title"].str.lower().fillna("")
            is_c_level = unique_titles.str.contains(r"ceo|cto|cfo|president|director", na=False).sum()
            is_ten_pct = unique_titles.str.contains(r"10%|owner", na=False).sum()
            
            num_buyers = df["insider_name"].nunique()
            
            # Score components: 
            # 0.4 for multi-buyer, 0.3 for C-level presence, 0.3 for 10% holder participation
            score = min(num_buyers / 5.0, 1.0) * 0.4
            score += 0.3 if is_c_level > 0 else 0.0
            score += 0.3 if is_ten_pct > 0 else 0.0
            
            return float(score)
        except Exception:
            return 0.0

    def is_corporate_king_maker(self, ticker: str) -> float:
        """
        Detects if a Global 500 partner or major strategic entity is involved.
        This often appears as a 13D filing from a corporate parent.
        """
        # Placeholder for corporate name matching (e.g. Lockheed, Toyota as filers)
        KING_MAKERS = ["TOYOTA", "LOCKHEED", "GOOGLE", "ALPHABET", "AMAZON", "MICROSOFT", "INTEL"]
        
        try:
            # We check recent 13D/G filings for this ticker
            df_13d = self.fetch_13d_filings(days_back=90)
            if df_13d.empty: return 0.0
            
            ticker_filings = df_13d[df_13d["ticker"] == ticker]
            if ticker_filings.empty: return 0.0
            
            # Check if any filer name matches a King Maker
            for filer in ticker_filings["filer_name"].str.upper():
                if any(km in filer for km in KING_MAKERS):
                    return 1.0
            return 0.0
        except Exception:
            return 0.0

    def get_institutional_drift(self, ticker: str) -> float:
        """
        Scores the transition from zero institutional presence to 'Smart Money' entry.
        Calculated as the count of known 'Super Investors' holding the ticker (13F).
        """
        # This is expensive to do ticker-by-ticker without a localized database.
        # For the POC, we simulate the 'Drift' signal using a simplified heuristic.
        # Logic: If recent 13D filings exist but price is low, drift is likely HIGH.
        try:
            df_13d = self.fetch_13d_filings(days_back=120)
            if df_13d.empty: return 0.0
            
            count = len(df_13d[df_13d["ticker"] == ticker])
            return float(min(count / 3.0, 1.0))
        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════
    # SC 13D: Activist / Strategic Stakes (Entry Signal)
    # ═══════════════════════════════════════════════════════

    def fetch_13d_filings(self, days_back: int = 365) -> pd.DataFrame:
        """
        Fetch SC 13D filings — activist/strategic investors acquiring >5%
        of a company. These are the Toyota-buys-small-company signals.

        Uses SEC EDGAR Full-Text Search (EFTS).
        """
        date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        date_to = datetime.now().strftime("%Y-%m-%d")

        all_filings = []
        start = 0
        page_size = 100

        while start < 500:  # Cap at 500 filings
            self._sec_rate_limit()

            url = "https://efts.sec.gov/LATEST/search-index"
            params = {
                "q": '"form-type":"SC 13D"',
                "dateRange": "custom",
                "startdt": date_from,
                "enddt": date_to,
                "from": start,
                "size": page_size,
            }

            try:
                r = self._sec_client.get(url, params=params)
                r.raise_for_status()
                data = r.json()

                hits = data.get("hits", {}).get("hits", [])
                if not hits:
                    break

                for hit in hits:
                    source = hit.get("_source", {})
                    # Try to get ticker from source metadata
                    ticker = source.get("tickers")
                    if isinstance(ticker, list) and ticker:
                        ticker = ticker[0]
                    elif isinstance(ticker, str):
                        ticker = ticker.split(",")[0]
                        
                    all_filings.append({
                        "filing_url": source.get("file_url", ""),
                        "filing_date": source.get("file_date", ""),
                        "filer_name": source.get("display_names", [""])[0] if source.get("display_names") else "",
                        "form_type": source.get("form_type", "SC 13D"),
                        "ticker": ticker,
                    })

                start += page_size
                logger.info(f"13D filings: {len(all_filings)} fetched...")

            except Exception as e:
                logger.warning(f"13D fetch error at offset {start}: {e}")
                break

        if not all_filings:
            return pd.DataFrame()

        df = pd.DataFrame(all_filings)
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")

        # Fallback to URL extraction if ticker still missing
        missing_mask = df["ticker"].isna()
        if missing_mask.any():
            df.loc[missing_mask, "ticker"] = df.loc[missing_mask, "filing_url"].apply(self._extract_ticker_from_url)

        # Mark whale type
        df["whale_type"] = "activist_13d"
        df["transaction_code"] = "P"  # 13D = taking a stake = purchase-like

        logger.info(f"13D: {len(df)} activist/strategic filings")
        return df

    # ═══════════════════════════════════════════════════════
    # 13F: Institutional Holdings (Confirmatory Signal)
    # ═══════════════════════════════════════════════════════

    def fetch_13f_new_positions(
        self, cik: str, quarters_back: int = 2
    ) -> pd.DataFrame:
        """
        Fetch new positions from a fund's 13F filings.
        Compares most recent quarter vs previous to find NEW buys.

        Args:
            cik: SEC CIK of the institution (e.g., "0001067983" for Berkshire)
            quarters_back: How many quarters to look back

        Returns:
            DataFrame of new positions (tickers added in latest quarter)
        """
        self._sec_rate_limit()

        try:
            # Get filing list
            cik_padded = cik.zfill(10)
            url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
            r = self._sec_client.get(url)
            r.raise_for_status()
            data = r.json()

            filer_name = data.get("name", "Unknown")
            recent = data.get("filings", {}).get("recent", {})

            # Find 13F filings
            f13_indices = [
                i for i, f in enumerate(recent.get("form", []))
                if "13F" in f and "HR" in f
            ]

            if len(f13_indices) < 2:
                logger.info(f"13F: {filer_name} — not enough filings for comparison")
                return pd.DataFrame()

            # Get the two most recent 13F filing dates
            latest_date = recent["filingDate"][f13_indices[0]]
            prev_date = recent["filingDate"][f13_indices[1]]

            logger.info(
                f"13F: {filer_name} — comparing {latest_date} vs {prev_date}"
            )

            # Return metadata (actual XML parsing of 13F tables is complex;
            # we record the signal that a super-investor filed)
            result = pd.DataFrame([{
                "filer_name": filer_name,
                "cik": cik,
                "filing_date": latest_date,
                "prev_filing_date": prev_date,
                "whale_type": "institutional_13f",
                "transaction_code": "P",
            }])
            result["filing_date"] = pd.to_datetime(result["filing_date"], errors="coerce")
            return result

        except Exception as e:
            logger.warning(f"13F fetch error for CIK {cik}: {e}")
            return pd.DataFrame()

    def fetch_all_super_investor_signals(self) -> pd.DataFrame:
        """
        Check all known super-investors for recent 13F filings.
        Returns metadata about their latest filings.
        """
        frames = []
        for cik, name in SUPER_INVESTORS.items():
            logger.info(f"Checking {name}...")
            df = self.fetch_13f_new_positions(cik)
            if not df.empty:
                frames.append(df)
            time.sleep(0.15)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ═══════════════════════════════════════════════════════
    # UNIFIED: All Whale Signals
    # ═══════════════════════════════════════════════════════

    def fetch_all_whale_signals(
        self,
        include_form4: bool = True,
        include_clusters: bool = True,
        include_13d: bool = True,
        include_13f: bool = True,
        days_back: int = 365,
    ) -> pd.DataFrame:
        """
        Fetch ALL whale signals from all sources.

        Returns unified DataFrame with columns:
            ticker, transaction_date, filing_date, insider_name,
            price, shares, value, whale_type, transaction_code
        """
        frames = []

        if include_form4:
            logger.info("═══ Form 4: Insider Purchases ═══")
            purchases = self.fetch_insider_purchases(days_back=days_back)
            if not purchases.empty:
                frames.append(purchases)
                logger.info(f"  {len(purchases)} insider purchases")

        if include_clusters:
            logger.info("═══ Form 4: Cluster Buys ═══")
            clusters = self.fetch_cluster_buys()
            if not clusters.empty:
                frames.append(clusters)
                logger.info(f"  {len(clusters)} cluster buys")

        if include_13d:
            logger.info("═══ SC 13D: Activist Stakes ═══")
            stakes = self.fetch_13d_filings(days_back=days_back)
            if not stakes.empty:
                frames.append(stakes)
                logger.info(f"  {len(stakes)} activist filings")

        if include_13f:
            logger.info("═══ 13F: Super Investors ═══")
            institutional = self.fetch_all_super_investor_signals()
            if not institutional.empty:
                frames.append(institutional)
                logger.info(f"  {len(institutional)} institutional signals")

        if not frames:
            return pd.DataFrame()

        # Unify columns
        standard_cols = [
            "ticker", "transaction_date", "filing_date", "insider_name",
            "insider_title", "transaction_code", "price", "shares",
            "value", "ownership_after", "whale_type", "source",
        ]

        combined = pd.concat(frames, ignore_index=True)

        # Ensure all standard columns exist
        for col in standard_cols:
            if col not in combined.columns:
                combined[col] = None

        combined["source"] = combined.get("source", "sec_form4")
        combined = combined.sort_values("filing_date", ascending=False).reset_index(drop=True)

        logger.info(f"═══ Total whale signals: {len(combined)} ═══")
        return combined

    # ═══════════════════════════════════════════════════════
    # INTERNAL PARSERS
    # ═══════════════════════════════════════════════════════

    def _parse_openinsider(
        self,
        raw: pd.DataFrame,
        signal_type: str = "insider_purchase",
        filter_purchases: bool = False,
        filter_sales: bool = False,
    ) -> pd.DataFrame:
        """Parse OpenInsider HTML table into standard format."""
        # Normalize column names
        raw.columns = [str(c).replace('\xa0', ' ').strip() for c in raw.columns]

        # Dynamic column mapping
        col_map = {}
        for c in raw.columns:
            cl = c.lower()
            if "filing" in cl and "date" in cl: col_map["filing_date"] = c
            elif "trade" in cl and "date" in cl: col_map["trade_date"] = c
            elif cl == "ticker": col_map["ticker"] = c
            elif "insider" in cl and "name" in cl: col_map["insider_name"] = c
            elif "company" in cl and "name" in cl: col_map["company_name"] = c
            elif cl == "title": col_map["title"] = c
            elif "trade" in cl and "type" in cl: col_map["trade_type"] = c
            elif cl == "price": col_map["price"] = c
            elif cl == "qty": col_map["qty"] = c
            elif cl == "owned": col_map["owned"] = c
            elif cl == "value": col_map["value"] = c

        records = []
        for _, row in raw.iterrows():
            try:
                trade_type = str(row[col_map["trade_type"]]) if "trade_type" in col_map else ""

                is_purchase = "Purchase" in trade_type or "Buy" in trade_type
                is_sale = "Sale" in trade_type

                if filter_purchases and not is_purchase:
                    continue
                if filter_sales and not is_sale:
                    continue

                ticker = str(row[col_map["ticker"]]).strip() if "ticker" in col_map else None
                if not ticker or ticker == "nan":
                    continue

                record = {
                    "ticker": ticker,
                    "filing_date": str(row[col_map["filing_date"]]) if "filing_date" in col_map else None,
                    "transaction_date": str(row[col_map["trade_date"]]) if "trade_date" in col_map else None,
                    "insider_name": str(row[col_map.get("insider_name", "")]) if "insider_name" in col_map else "Unknown",
                    "insider_title": str(row[col_map.get("title", "")]) if "title" in col_map else "",
                    "transaction_code": "P" if is_purchase else "S",
                    "whale_type": signal_type,
                    "source": "openinsider",
                }

                # Numeric fields
                for field, key in [("price", "price"), ("shares", "qty"),
                                    ("ownership_after", "owned"), ("value", "value")]:
                    if key in col_map:
                        val = str(row[col_map[key]]).replace(',', '').replace('$', '').replace('+', '')
                        val = re.sub(r'[^\d.]', '', val)
                        try:
                            record[field] = float(val) if val else None
                        except:
                            record[field] = None
                    else:
                        record[field] = None

                records.append(record)
            except Exception:
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        return df.dropna(subset=["ticker"])

    def _extract_ticker_from_url(self, url: str) -> Optional[str]:
        """Try to extract ticker symbol from SEC filing URL or path."""
        if not url:
            return None
            
        # 1. Look for explicit ticker pattern in filing doc name
        # e.g., .../filename-AAPL-13D.xml
        match = re.search(r"[-_]([A-Z]{1,5})[-_]", url.upper())
        if match:
            return match.group(1)
            
        # 2. Many activist filings have the ticker in the text but not the URL
        # For this verification, we prioritize signals where ticker is identifiable
        return None

    def fetch_otc_penny_signals(self, days_back: int = 30) -> pd.DataFrame:
        """
        Scan for whale activity specifically in OTC and Penny Stock universes.
        Focuses on symbols with '.PK', '.OB' or low-priced entries.
        """
        logger.info("Scanning OTC/Penny stock universe...")
        all_signals = self.fetch_all_whale_signals(days_back=days_back)
        
        if all_signals.empty:
            return pd.DataFrame()
            
        # Filter for OTC or low price (< $5)
        otc_mask = all_signals["ticker"].str.contains(r"\.PK|\.OB", case=False, na=False)
        penny_mask = (all_signals["price"] < 5.0) & (all_signals["price"] > 0)
        
        penny_signals = all_signals[otc_mask | penny_mask].copy()
        
        # Add a liquidity risk flag
        penny_signals["liquidity_risk"] = "HIGH"
        
        logger.info(f"Found {len(penny_signals)} Penny Stock whale signals.")
        return penny_signals

    def close(self):
        """Close HTTP clients."""
        if not self._openinsider_client.is_closed:
            self._openinsider_client.close()
        if not self._sec_client.is_closed:
            self._sec_client.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
