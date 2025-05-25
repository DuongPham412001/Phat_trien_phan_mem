import argparse
import time
import random
import pandas as pd
from pytrends.request import TrendReq

def fetch_google_trends(keywords, timeframe, sleept):
    pytrends = TrendReq(hl='vi-VN', tz=360)
    results = {}
    
    for keyword in keywords:
        print(f"Fetching data for: {keyword}")
        pytrends.build_payload([keyword], timeframe=timeframe, geo='VN')
        interest_over_time = None
        
        while interest_over_time is None:
            try:
                interest_over_time = pytrends.interest_over_time()
            except Exception as e:
                print(f"Error encountered: {e}. Retrying...")
                time.sleep(random.uniform(5, 15))
        
        results[keyword] = interest_over_time
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Fetch Google Trends data for given keywords.")
    parser.add_argument("--keywords", nargs='+', required=True, help="List of keywords to fetch trends for.")
    parser.add_argument("--timeframe", default='today 12-m', help="Timeframe for Google Trends data (e.g., 'today 12-m', 'now 7-d').")
    parser.add_argument("--sleept", default=60)
    args = parser.parse_args()
    trends_data = fetch_google_trends(args.keywords, args.timeframe, args.sleept)
    
    for keyword, df in trends_data.items():
        filename = f"interest_over_time_{keyword.replace(' ', '_')}.csv"
        df.to_csv(filename)
        print(f"Saved: {filename}")
    
    print("Data fetched and saved successfully.")

if __name__ == "__main__":
    main()
