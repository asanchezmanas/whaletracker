import logging
import sys
import os
from pprint import pprint

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from qdn.data.gov_connector import GovContractConnector

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_gov_connector():
    gov = GovContractConnector()
    
    # Test 1: Lockheed Martin (LMT)
    print("\n--- Testing LMT (Lockheed Martin) ---")
    recipient = gov.resolve_recipient_name("LMT")
    print(f"Ticker LMT resolved to: {recipient}")
    
    velocity = gov.get_company_contract_velocity("LMT", recipient)
    print("Contract Velocity Snapshot:")
    pprint(velocity)
    
    # Test 2: Boeing (BA)
    print("\n--- Testing BA (Boeing) ---")
    backlog = gov.get_backlog_estimate("BA")
    print("Backlog Estimate:")
    pprint(backlog)
    
    # Test 3: Search Discovery
    print("\n--- Searching for 'Palantir' ---")
    search = gov.search_recipients("Palantir")
    print(f"Found {len(search)} matches. Top result: {search[0].get('recipient_name') if search else 'None'}")

if __name__ == "__main__":
    test_gov_connector()
