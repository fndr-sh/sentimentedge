import os
import sqlite3
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    newsapi_key = os.getenv('NEWSAPI_KEY')
    
    if not newsapi_key:
        logger.error("NEWSAPI_KEY not found in environment variables")
        return None, None
    
    return gemini_api_key, newsapi_key

def create_database():
    """Create SQLite database and tables"""
    conn = sqlite3.connect('sentimentedge.db')
    cursor = conn.cursor()
    
    # Create prices table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
    ''')
    
    # Create headlines table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS headlines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            date TEXT,
            headline TEXT,
            url TEXT,
            source TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database and tables created successfully")

def download_price_data():
    """Download 2 years of daily OHLCV data for tickers"""
    tickers = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    logger.info(f"Downloading price data from {start_date.date()} to {end_date.date()}")
    
    all_data = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                continue
                
            # Reset index to make date a column
            data.reset_index(inplace=True)
            data['ticker'] = ticker
            
            # Select and rename columns
            data = data[['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
            
            # Convert date to string
            data['date'] = data['date'].dt.strftime('%Y-%m-%d')
            
            all_data.append(data)
            logger.info(f"Downloaded {len(data)} rows of data for {ticker}")
            
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {str(e)}")
            continue
    
    if not all_data:
        logger.error("No price data downloaded")
        return None
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Save to database
    conn = sqlite3.connect('sentimentedge.db')
    combined_data.to_sql('prices', conn, if_exists='replace', index=False)
    conn.close()
    
    logger.info(f"Saved {len(combined_data)} rows of price data to database")
    return combined_data

def download_news_data(newsapi_key):
    """Download news headlines for each ticker"""
    newsapi = NewsApiClient(api_key=newsapi_key)
    
    # Mapping of tickers to company names
    company_names = {
        'AAPL': 'Apple',
        'TSLA': 'Tesla', 
        'NVDA': 'Nvidia',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google'
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    logger.info(f"Downloading news data from {start_date.date()} to {end_date.date()}")
    
    all_headlines = []
    
    for ticker, company_name in company_names.items():
        try:
            # Search for news
            response = newsapi.get_everything(
                q=company_name,
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt',
                page_size=100  # Maximum per page
            )
            
            articles = response.get('articles', [])
            logger.info(f"Found {len(articles)} articles for {ticker} ({company_name})")
            
            for article in articles:
                headline_data = {
                    'ticker': ticker,
                    'date': article['publishedAt'][:10],  # Extract date part
                    'headline': article['title'],
                    'url': article['url'],
                    'source': article['source']['name']
                }
                all_headlines.append(headline_data)
                
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            continue
    
    if not all_headlines:
        logger.error("No news data downloaded")
        return None
    
    # Convert to DataFrame
    headlines_df = pd.DataFrame(all_headlines)
    
    # Save to database
    conn = sqlite3.connect('sentimentedge.db')
    headlines_df.to_sql('headlines', conn, if_exists='replace', index=False)
    conn.close()
    
    logger.info(f"Saved {len(headlines_df)} headlines to database")
    return headlines_df

def print_summary():
    """Print summary statistics"""
    conn = sqlite3.connect('sentimentedge.db')
    cursor = conn.cursor()
    
    # Total rows in prices table
    cursor.execute("SELECT COUNT(*) FROM prices")
    total_prices = cursor.fetchone()[0]
    
    # Total rows in headlines table
    cursor.execute("SELECT COUNT(*) FROM headlines")
    total_headlines = cursor.fetchone()[0]
    
    # Date range of price data
    cursor.execute("SELECT MIN(date), MAX(date) FROM prices")
    price_date_range = cursor.fetchone()
    
    # Number of headlines per ticker
    cursor.execute("SELECT ticker, COUNT(*) as count FROM headlines GROUP BY ticker ORDER BY ticker")
    headlines_per_ticker = cursor.fetchall()
    
    conn.close()
    
    print("\n" + "="*50)
    print("PIPELINE SUMMARY")
    print("="*50)
    print(f"Total rows in prices table: {total_prices:,}")
    print(f"Total rows in headlines table: {total_headlines:,}")
    
    if price_date_range[0] and price_date_range[1]:
        print(f"Price data date range: {price_date_range[0]} to {price_date_range[1]}")
    else:
        print("No price data found")
    
    print("\nHeadlines per ticker:")
    for ticker, count in headlines_per_ticker:
        print(f"  {ticker}: {count:,}")
    print("="*50)

def main():
    """Main pipeline function"""
    logger.info("Starting sentiment edge pipeline")
    
    try:
        # Load environment variables
        gemini_key, newsapi_key = load_environment()
        if not newsapi_key:
            logger.error("Failed to load required API keys")
            return
        
        # Create database
        create_database()
        
        # Download price data
        price_data = download_price_data()
        
        # Download news data
        headlines_data = download_news_data(newsapi_key)
        
        # Print summary
        print_summary()
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
