import os
import sqlite3
import pandas as pd
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return None
    
    return gemini_api_key

def setup_database():
    """Add sentiment columns to headlines table if they don't exist"""
    conn = sqlite3.connect('sentimentedge.db')
    cursor = conn.cursor()
    
    # Add sentiment columns to headlines table
    try:
        cursor.execute("ALTER TABLE headlines ADD COLUMN sentiment TEXT")
        logger.info("Added sentiment column to headlines table")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            logger.info("Sentiment column already exists")
        else:
            logger.error(f"Error adding sentiment column: {e}")
    
    try:
        cursor.execute("ALTER TABLE headlines ADD COLUMN confidence REAL")
        logger.info("Added confidence column to headlines table")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            logger.info("Confidence column already exists")
        else:
            logger.error(f"Error adding confidence column: {e}")
    
    try:
        cursor.execute("ALTER TABLE headlines ADD COLUMN reason TEXT")
        logger.info("Added reason column to headlines table")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            logger.info("Reason column already exists")
        else:
            logger.error(f"Error adding reason column: {e}")
    
    # Create daily_sentiment table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_sentiment (
            ticker TEXT,
            date TEXT,
            sentiment_score REAL,
            headline_count INTEGER,
            PRIMARY KEY (ticker, date)
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database setup completed")

def get_unanalyzed_headlines():
    """Get headlines that don't have sentiment analysis yet"""
    conn = sqlite3.connect('sentimentedge.db')
    
    query = """
        SELECT ticker, date, headline, url, source
        FROM headlines 
        WHERE sentiment IS NULL OR confidence IS NULL OR reason IS NULL
        ORDER BY date DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Found {len(df)} unanalyzed headlines")
    return df

def analyze_sentiment_batch(headlines: List[str], gemini_api_key: str):
    """Analyze sentiment for a batch of headlines using Gemini API"""
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # System instruction
    system_instruction = """You are a financial sentiment analyzer. For each headline I give you, respond with ONLY a JSON array of objects with these exact keys: 
    - sentiment (must be exactly "bullish", "bearish", or "neutral")
    - confidence (a float from 0.0 to 1.0) 
    - reason (max 8 words explaining why)
    
    Example format: [{"sentiment": "bullish", "confidence": 0.8, "reason": "positive earnings report"}]"""
    
    # Prepare headlines text
    headlines_text = "\n".join([f"{i+1}. {headline}" for i, headline in enumerate(headlines)])
    
    prompt = f"""Analyze the sentiment of these financial news headlines:\n\n{headlines_text}\n\nRespond with only a JSON array."""
    
    try:
        response = model.generate_content(system_instruction + "\n\n" + prompt)
        response_text = response.text.strip()
        
        # Clean up response text
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        results = json.loads(response_text)
        
        # Validate results
        if not isinstance(results, list):
            logger.warning("API response is not a list, wrapping in list")
            results = [results]
        
        # Validate each result
        validated_results = []
        for result in results:
            if isinstance(result, dict) and 'sentiment' in result and 'confidence' in result and 'reason' in result:
                if result['sentiment'] in ['bullish', 'bearish', 'neutral']:
                    validated_results.append(result)
                else:
                    logger.warning(f"Invalid sentiment value: {result['sentiment']}")
            else:
                logger.warning(f"Invalid result format: {result}")
        
        return validated_results
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Response text: {response_text if 'response_text' in locals() else 'No response text'}")
        return []
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return []

def update_headlines_sentiment(headlines_data: pd.DataFrame, sentiments: List[Dict[str, Any]]):
    """Update sentiment analysis results in the database"""
    if len(headlines_data) != len(sentiments):
        logger.error("Mismatch between headlines and sentiment results")
        return
    
    conn = sqlite3.connect('sentimentedge.db')
    cursor = conn.cursor()
    
    for (_, row), sentiment in zip(headlines_data.iterrows(), sentiments):
        try:
            cursor.execute("""
                UPDATE headlines 
                SET sentiment = ?, confidence = ?, reason = ?
                WHERE ticker = ? AND date = ? AND headline = ?
            """, (
                sentiment.get('sentiment'),
                sentiment.get('confidence'),
                sentiment.get('reason'),
                row['ticker'],
                row['date'],
                row['headline']
            ))
        except Exception as e:
            logger.error(f"Error updating headline for {row['ticker']} on {row['date']}: {e}")
    
    conn.commit()
    conn.close()
    logger.info(f"Updated {len(sentiments)} headlines with sentiment analysis")

def calculate_daily_sentiment():
    """Calculate daily sentiment scores for each ticker"""
    conn = sqlite3.connect('sentimentedge.db')
    
    query = """
        SELECT ticker, date, sentiment, confidence
        FROM headlines 
        WHERE sentiment IS NOT NULL AND confidence IS NOT NULL
        ORDER BY ticker, date
    """
    
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        logger.warning("No sentiment data found for daily calculation")
        conn.close()
        return
    
    # Convert sentiment to numeric direction
    sentiment_map = {'bullish': 1, 'bearish': -1, 'neutral': 0}
    df['direction'] = df['sentiment'].map(sentiment_map)
    df['weighted_sentiment'] = df['direction'] * df['confidence']
    
    # Group by ticker and date
    daily_scores = df.groupby(['ticker', 'date']).agg({
        'weighted_sentiment': 'mean',
        'direction': 'count'  # This will be the headline count
    }).reset_index()
    
    daily_scores.columns = ['ticker', 'date', 'sentiment_score', 'headline_count']
    
    # Save to daily_sentiment table
    daily_scores.to_sql('daily_sentiment', conn, if_exists='replace', index=False)
    conn.close()
    
    logger.info(f"Calculated daily sentiment scores for {len(daily_scores)} ticker-date combinations")

def print_summary():
    """Print summary statistics"""
    conn = sqlite3.connect('sentimentedge.db')
    
    # Average sentiment per ticker
    query = """
        SELECT ticker, AVG(sentiment_score) as avg_sentiment, COUNT(*) as days
        FROM daily_sentiment
        GROUP BY ticker
        ORDER BY avg_sentiment DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print("\n" + "="*50)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("="*50)
    
    if not df.empty:
        print("Average sentiment score per ticker:")
        for _, row in df.iterrows():
            print(f"  {row['ticker']}: {row['avg_sentiment']:.3f} (over {row['days']} days)")
    else:
        print("No sentiment data available")
    
    print("="*50)

def main():
    """Main sentiment analysis function"""
    logger.info("Starting sentiment analysis pipeline")
    
    try:
        # Load environment
        gemini_api_key = load_environment()
        if not gemini_api_key:
            logger.error("Failed to load GEMINI_API_KEY")
            return
        
        # Setup database
        setup_database()
        
        # Get unanalyzed headlines
        unanalyzed_df = get_unanalyzed_headlines()
        
        if unanalyzed_df.empty:
            logger.info("No new headlines to analyze")
        else:
            # Process headlines in batches of 20
            batch_size = 20
            total_headlines = len(unanalyzed_df)
            processed_count = 0
            
            for i in range(0, total_headlines, batch_size):
                batch = unanalyzed_df.iloc[i:i+batch_size]
                headlines = batch['headline'].tolist()
                
                logger.info(f"Processing batch {i//batch_size + 1}: {len(headlines)} headlines")
                
                # Analyze sentiment
                sentiments = analyze_sentiment_batch(headlines, gemini_api_key)
                
                if sentiments:
                    # Update database
                    update_headlines_sentiment(batch.iloc[:len(sentiments)], sentiments)
                    processed_count += len(sentiments)
                else:
                    logger.warning(f"Failed to analyze batch {i//batch_size + 1}")
                
                # Small delay to avoid rate limiting
                import time
                time.sleep(1)
            
            logger.info(f"Processed {processed_count} headlines out of {total_headlines}")
        
        # Calculate daily sentiment scores
        calculate_daily_sentiment()
        
        # Print summary
        print_summary()
        
        logger.info("Sentiment analysis pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Sentiment analysis pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
