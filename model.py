import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_data():
    """Load and merge prices and daily_sentiment data from database"""
    conn = sqlite3.connect('sentimentedge.db')
    
    # Join prices and daily_sentiment tables
    query = """
    SELECT 
        p.ticker,
        p.date,
        p.open,
        p.high,
        p.low,
        p.close,
        p.volume,
        COALESCE(ds.sentiment_score, 0) as sentiment_score,
        COALESCE(ds.headline_count, 0) as headline_count
    FROM prices p
    LEFT JOIN daily_sentiment ds ON p.ticker = ds.ticker AND p.date = ds.date
    ORDER BY p.ticker, p.date
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert date column
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df)} rows of data for {df['ticker'].nunique()} tickers")
    return df

def create_features(df):
    """Create features for each ticker"""
    features_list = []
    
    for ticker in df['ticker'].unique():
        ticker_data = df[df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date')
        
        # Sentiment rolling averages
        ticker_data['sentiment_3d_avg'] = ticker_data['sentiment_score'].rolling(window=3, min_periods=1).mean()
        ticker_data['sentiment_7d_avg'] = ticker_data['sentiment_score'].rolling(window=7, min_periods=1).mean()
        
        # Price momentum (5-day)
        ticker_data['close_5d_ago'] = ticker_data['close'].shift(5)
        ticker_data['price_momentum_5d'] = ((ticker_data['close'] - ticker_data['close_5d_ago']) / ticker_data['close_5d_ago']) * 100
        
        # Volume change
        ticker_data['volume_prev_day'] = ticker_data['volume'].shift(1)
        ticker_data['volume_change'] = ((ticker_data['volume'] - ticker_data['volume_prev_day']) / ticker_data['volume_prev_day']) * 100
        
        # Target variable (1 if tomorrow's close > today's close, else 0)
        ticker_data['close_next_day'] = ticker_data['close'].shift(-1)
        ticker_data['target'] = (ticker_data['close_next_day'] > ticker_data['close']).astype(int)
        
        features_list.append(ticker_data)
    
    # Combine all ticker data
    features_df = pd.concat(features_list, ignore_index=True)
    
    # Select feature columns
    feature_columns = [
        'sentiment_score',
        'sentiment_3d_avg', 
        'sentiment_7d_avg',
        'headline_count',
        'price_momentum_5d',
        'volume_change',
        'target'
    ]
    
    final_df = features_df[feature_columns + ['ticker', 'date']].copy()
    
    return final_df

def split_data_chronologically(df):
    """Split data chronologically (80% train, 20% test)"""
    # Sort by date
    df_sorted = df.sort_values('date')
    
    # Remove rows with NaN values
    df_clean = df_sorted.dropna()
    print(f"After removing NaN values: {len(df_clean)} rows remaining")
    
    # Get unique dates and split them
    unique_dates = df_clean['date'].unique()
    split_idx = int(len(unique_dates) * 0.8)
    
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]
    
    train_data = df_clean[df_clean['date'].isin(train_dates)]
    test_data = df_clean[df_clean['date'].isin(test_dates)]
    
    print(f"Train set: {len(train_data)} rows ({len(train_dates)} dates)")
    print(f"Test set: {len(test_data)} rows ({len(test_dates)} dates)")
    
    # Prepare features and target
    feature_cols = ['sentiment_score', 'sentiment_3d_avg', 'sentiment_7d_avg', 
                   'headline_count', 'price_momentum_5d', 'volume_change']
    
    X_train = train_data[feature_cols]
    y_train = train_data['target']
    X_test = test_data[feature_cols]
    y_test = test_data['target']
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_models(X_train, X_test, y_train, y_test):
    """Train RandomForest and XGBoost models"""
    models = {}
    predictions = {}
    
    # RandomForest
    print("\nTraining RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    models['RandomForest'] = rf
    predictions['RandomForest'] = {
        'y_pred': rf_pred,
        'y_pred_proba': rf_pred_proba
    }
    
    # XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = {
        'y_pred': xgb_pred,
        'y_pred_proba': xgb_pred_proba
    }
    
    return models, predictions

def evaluate_models(y_test, predictions):
    """Evaluate models and print metrics"""
    results = {}
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    for model_name, pred_data in predictions.items():
        print(f"\n{model_name} Results:")
        print("-" * 40)
        
        # Accuracy
        accuracy = accuracy_score(y_test, pred_data['y_pred'])
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, pred_data['y_pred']))
        
        # AUC-ROC
        auc = roc_auc_score(y_test, pred_data['y_pred_proba'])
        print(f"AUC-ROC: {auc:.4f}")
        
        results[model_name] = {
            'accuracy': accuracy,
            'auc': auc
        }
    
    return results

def create_roc_curve(y_test, predictions):
    """Create and save ROC curve chart"""
    fig = go.Figure()
    
    colors = ['blue', 'red']
    
    for i, (model_name, pred_data) in enumerate(predictions.items()):
        fpr, tpr, _ = roc_curve(y_test, pred_data['y_pred_proba'])
        auc = roc_auc_score(y_test, pred_data['y_pred_proba'])
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc:.3f})',
            line=dict(color=colors[i], width=2)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='black', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=800,
        height=600
    )
    
    fig.write_html('roc_curve.html')
    print("ROC curve saved as roc_curve.html")

def create_feature_importance_chart(rf_model, feature_cols):
    """Create and save feature importance chart"""
    importances = rf_model.feature_importances_
    
    fig = px.bar(
        x=importances,
        y=feature_cols,
        orientation='h',
        title='Random Forest Feature Importances',
        labels={'x': 'Importance', 'y': 'Features'}
    )
    
    fig.update_layout(
        width=800,
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    fig.write_html('feature_importance.html')
    print("Feature importance chart saved as feature_importance.html")

def save_best_model(models, results):
    """Save the best model based on AUC-ROC score"""
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_model = models[best_model_name]
    best_auc = results[best_model_name]['auc']
    
    # Save with pickle
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"\n" + "="*60)
    print("BEST MODEL")
    print("="*60)
    print(f"Winner: {best_model_name}")
    print(f"Reason: Highest AUC-ROC score ({best_auc:.4f})")
    print(f"Model saved as: best_model.pkl")
    
    return best_model_name, best_auc

def main():
    """Main modeling pipeline"""
    print("Starting sentiment edge modeling pipeline")
    print("="*60)
    
    # Load and merge data
    df = load_and_merge_data()
    
    # Create features
    features_df = create_features(df)
    print(f"Features created for {len(features_df)} rows")
    
    # Split data
    X_train, X_test, y_train, y_test, feature_cols = split_data_chronologically(features_df)
    
    # Train models
    models, predictions = train_models(X_train, X_test, y_train, y_test)
    
    # Evaluate models
    results = evaluate_models(y_test, predictions)
    
    # Create charts
    create_roc_curve(y_test, predictions)
    create_feature_importance_chart(models['RandomForest'], feature_cols)
    
    # Save best model
    best_model_name, best_auc = save_best_model(models, results)
    
    print("\nModeling pipeline completed successfully!")

if __name__ == "__main__":
    main()
