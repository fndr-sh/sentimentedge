import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(layout="wide", page_title="SentimentEdge", page_icon="📈")

def load_data(ticker, start_date, end_date):
    """Load data from SQLite database for selected ticker and date range"""
    conn = sqlite3.connect('sentimentedge.db')
    
    # Load price data
    price_query = """
    SELECT ticker, date, open, high, low, close, volume
    FROM prices
    WHERE ticker = ? AND date BETWEEN ? AND ?
    ORDER BY date
    """
    prices_df = pd.read_sql_query(price_query, conn, params=(ticker, start_date, end_date))
    
    # Load sentiment data
    sentiment_query = """
    SELECT ticker, date, sentiment_score, headline_count
    FROM daily_sentiment
    WHERE ticker = ? AND date BETWEEN ? AND ?
    ORDER BY date
    """
    sentiment_df = pd.read_sql_query(sentiment_query, conn, params=(ticker, start_date, end_date))
    
    # Load headlines
    headlines_query = """
    SELECT date, headline, sentiment, confidence
    FROM headlines
    WHERE ticker = ? AND date BETWEEN ? AND ?
    AND sentiment IS NOT NULL AND confidence IS NOT NULL
    ORDER BY date DESC
    LIMIT 20
    """
    headlines_df = pd.read_sql_query(headlines_query, conn, params=(ticker, start_date, end_date))
    
    conn.close()
    
    return prices_df, sentiment_df, headlines_df

def load_model():
    """Load the trained model"""
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please run model.py first.")
        return None

def create_features_for_prediction(prices_df, sentiment_df):
    """Create features for the latest data point"""
    if prices_df.empty or sentiment_df.empty:
        return None
    
    # Get the latest data
    latest_price = prices_df.iloc[-1]
    latest_sentiment = sentiment_df.iloc[-1]
    
    # Calculate features
    sentiment_3d_avg = sentiment_df.tail(3)['sentiment_score'].mean()
    sentiment_7d_avg = sentiment_df.tail(7)['sentiment_score'].mean()
    
    # Price momentum (5-day)
    if len(prices_df) >= 5:
        close_5d_ago = prices_df.iloc[-5]['close']
        price_momentum_5d = ((latest_price['close'] - close_5d_ago) / close_5d_ago) * 100
    else:
        price_momentum_5d = 0
    
    # Volume change
    if len(prices_df) >= 2:
        volume_prev_day = prices_df.iloc[-2]['volume']
        volume_change = ((latest_price['volume'] - volume_prev_day) / volume_prev_day) * 100
    else:
        volume_change = 0
    
    features = {
        'sentiment_score': latest_sentiment['sentiment_score'],
        'sentiment_3d_avg': sentiment_3d_avg,
        'sentiment_7d_avg': sentiment_7d_avg,
        'headline_count': latest_sentiment['headline_count'],
        'price_momentum_5d': price_momentum_5d,
        'volume_change': volume_change
    }
    
    return pd.DataFrame([features])

def make_prediction(model, features_df):
    """Make prediction using the loaded model"""
    if model is None or features_df is None:
        return None, None
    
    try:
        prediction = model.predict(features_df)[0]
        prediction_proba = model.predict_proba(features_df)[0]
        confidence = max(prediction_proba) * 100
        
        direction = "UP" if prediction == 1 else "DOWN"
        return direction, confidence
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def create_price_sentiment_chart(prices_df, sentiment_df, ticker):
    """Create the main price vs sentiment chart"""
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        subplot_titles=[f"{ticker} Price vs AI Sentiment Score"]
    )
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=prices_df['date'],
            y=prices_df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ),
        secondary_y=False
    )
    
    # Add sentiment bars
    colors = ['green' if score >= 0 else 'red' for score in sentiment_df['sentiment_score']]
    
    fig.add_trace(
        go.Bar(
            x=sentiment_df['date'],
            y=sentiment_df['sentiment_score'],
            name='Sentiment Score',
            marker_color=colors,
            opacity=0.7
        ),
        secondary_y=True
    )
    
    # Set axes labels
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Stock Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
    
    # Update layout
    fig.update_layout(
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def main():
    """Main Streamlit app"""
    # Sidebar
    st.sidebar.title("SentimentEdge")
    st.sidebar.markdown("**AI Financial Intelligence**")
    
    # Ticker selection
    ticker = st.sidebar.selectbox(
        "Select Ticker",
        ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"],
        index=0
    )
    
    # Date range selection
    default_end = datetime.now()
    default_start = default_end - timedelta(days=90)
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        max_value=default_end
    )
    
    end_date = st.sidebar.date_input(
        "End Date", 
        value=default_end,
        max_value=datetime.now()
    )
    
    # Convert dates to strings
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Load data
    prices_df, sentiment_df, headlines_df = load_data(ticker, start_str, end_str)
    
    # Load model
    model = load_model()
    
    # Main content
    st.title(f"📈 SentimentEdge Dashboard")
    
    if prices_df.empty or sentiment_df.empty:
        st.warning("No data available for the selected ticker and date range.")
        return
    
    # Row 1 - Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    
    # Current sentiment score
    latest_sentiment = sentiment_df.iloc[-1]['sentiment_score']
    col1.metric(
        "Current Sentiment",
        f"{latest_sentiment:.2f}",
        delta=None
    )
    
    # Prediction
    features_df = create_features_for_prediction(prices_df, sentiment_df)
    prediction, confidence = make_prediction(model, features_df)
    
    if prediction:
        col2.metric(
            "Prediction",
            prediction,
            delta=None
        )
        
        col3.metric(
            "Model Confidence",
            f"{confidence:.1f}%",
            delta=None
        )
    else:
        col2.metric("Prediction", "N/A")
        col3.metric("Model Confidence", "N/A")
    
    # Headlines analyzed
    total_headlines = len(headlines_df)
    col4.metric(
        "Headlines Analyzed",
        total_headlines,
        delta=None
    )
    
    # Row 2 - Main Chart
    st.plotly_chart(
        create_price_sentiment_chart(prices_df, sentiment_df, ticker),
        use_container_width=True
    )
    
    # Row 3 - Tabs
    tab1, tab2 = st.tabs(["Recent Headlines", "Model Insights"])
    
    with tab1:
        st.subheader("Recent Headlines")
        if not headlines_df.empty:
            # Format the dataframe for display
            display_df = headlines_df.copy()
            display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
            display_df['confidence'] = display_df['confidence'].round(2)
            display_df.columns = ['Date', 'Headline', 'Sentiment', 'Confidence']
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No headlines available for the selected period.")
    
    with tab2:
        st.subheader("Model Performance")
        
        # Try to load and display the HTML files
        try:
            with open('roc_curve.html', 'r', encoding='utf-8') as f:
                roc_html = f.read()
            st.markdown("### ROC Curve Comparison")
            components.html(roc_html, height=600, scrolling=True)
        except FileNotFoundError:
            st.info("ROC curve chart not found. Please run model.py first.")
        except UnicodeDecodeError:
            st.info("ROC curve chart has encoding issues. Please regenerate with model.py.")
        
        try:
            with open('feature_importance.html', 'r', encoding='utf-8') as f:
                feature_html = f.read()
            st.markdown("### Feature Importance")
            components.html(feature_html, height=500, scrolling=True)
        except FileNotFoundError:
            st.info("Feature importance chart not found. Please run model.py first.")
        except UnicodeDecodeError:
            st.info("Feature importance chart has encoding issues. Please regenerate with model.py.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: gray; padding: 20px;">'
        'Built with Gemini AI · scikit-learn · Streamlit'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
