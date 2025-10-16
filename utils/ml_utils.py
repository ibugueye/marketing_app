import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

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

def predict_clv(customer_data):
    """Prédit la valeur à vie du client basée sur les données"""
    # Modèle simplifié pour la démo
    clv = (
        customer_data['income'] * 0.001 +
        customer_data['spending_score'] * 10 +
        (365 - customer_data['days_since_last_purchase']) * 0.1 +
        customer_data['total_purchases'] * 5
    )
    return clv

def perform_clustering(df, n_clusters=4):
    """Effectue un clustering K-means sur les données clients"""
    features = ['age', 'income', 'spending_score']
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[features])
    return df, kmeans