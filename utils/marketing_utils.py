import pandas as pd
import numpy as np
from textblob import TextBlob
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def analyze_sentiment(text):
    """Analyse le sentiment d'un texte avec TextBlob"""
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    if sentiment.polarity > 0.1:
        sentiment_label = "Positif 😊"
        color = "green"
    elif sentiment.polarity < -0.1:
        sentiment_label = "Négatif 😠"
        color = "red"
    else:
        sentiment_label = "Neutre 😐"
        color = "gray"
    
    return {
        'polarity': sentiment.polarity,
        'subjectivity': sentiment.subjectivity,
        'label': sentiment_label,
        'color': color
    }

def calculate_clv(avg_order_value, purchase_frequency, customer_lifespan, profit_margin=0.3, acquisition_cost=0):
    """Calcule la Customer Lifetime Value"""
    annual_revenue = avg_order_value * purchase_frequency * 12
    total_revenue = annual_revenue * customer_lifespan
    gross_profit = total_revenue * profit_margin
    clv = gross_profit - acquisition_cost
    return clv, gross_profit, annual_revenue

def simulate_ad_auction(budget, competitors=3):
    """Simule une enchère publicitaire programmatique"""
    np.random.seed(42)
    competitor_bids = np.random.exponential(0.5, competitors) * budget
    
    user_bid = budget
    all_bids = list(competitor_bids) + [user_bid]
    winning_bid = max(all_bids)
    
    return {
        'user_bid': user_bid,
        'competitor_bids': competitor_bids,
        'winning_bid': winning_bid,
        'user_won': user_bid == winning_bid,
        'all_bids': all_bids
    }

def generate_customer_data(n_customers=300):
    """Génère des données clients simulées pour les démos"""
    np.random.seed(42)
    
    data = {
        'customer_id': range(1, n_customers + 1),
        'age': np.random.normal(45, 15, n_customers),
        'income': np.random.normal(50000, 20000, n_customers),
        'spending_score': np.random.normal(50, 25, n_customers),
        'days_since_last_purchase': np.random.exponential(30, n_customers),
        'total_purchases': np.random.poisson(15, n_customers)
    }
    
    df = pd.DataFrame(data)
    
    # Ajuster les valeurs dans des plages réalistes
    df['age'] = df['age'].clip(18, 80).astype(int)
    df['income'] = df['income'].clip(20000, 150000).astype(int)
    df['spending_score'] = df['spending_score'].clip(1, 100).astype(int)
    df['days_since_last_purchase'] = df['days_since_last_purchase'].clip(1, 365).astype(int)
    df['total_purchases'] = df['total_purchases'].clip(1, 50).astype(int)
    
    return df

def perform_customer_segmentation(df, n_clusters=4, features=None):
    """Effectue une segmentation client avec K-means"""
    if features is None:
        features = ['age', 'income', 'spending_score']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['segment'] = kmeans.fit_predict(X_scaled)
    
    return df, kmeans

def get_segment_strategies(segments):
    """Retourne des stratégies marketing par segment"""
    strategies = {
        0: "🎯 **Segment Basique** : Campagnes d'acquisition, offres découverte",
        1: "💰 **Segment Valeur** : Programmes fidélité, ventes croisées", 
        2: "⭐ **Segment Premium** : Services personnalisés, produits exclusifs",
        3: "⚠️ **Segment À Risque** : Campagnes de réactivation, enquêtes de satisfaction",
        4: "🚀 **Segment Croissance** : Upselling, programmes ambassadeurs",
        5: "🆕 **Segment Nouveaux** : Onboarding, éducation produit"
    }
    
    return {k: strategies[k] for k in segments if k in strategies}