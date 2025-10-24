import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import graphviz
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


from utils.ml_utils import *
from utils.marketing_utils import *

# Configuration de la page
st.set_page_config(
    page_title="AI Marketing Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger le CSS personnalisé
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Navigation dans la sidebar
def main():
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Choisissez une section :",
        [
            "🏠 Accueil", 
            "🤖 ML Fundamentals", 
            "🎯 Problèmes Marketing",
            "📢 Capter l'Attention",
            "🚀 Cas Pratiques"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**AI Marketing Explorer** v1.0\n\n"
        "Explorez l'IA dans le marketing à travers des démonstrations interactives."
    )

    # Router vers la page sélectionnée
    if page == "🏠 Accueil":
        show_homepage()
    elif page == "🤖 ML Fundamentals":
        show_ml_fundamentals()
    elif page == "🎯 Problèmes Marketing":
        show_marketing_problems()
    elif page == "📢 Capter l'Attention":
        show_attention_capture()
    elif page == "🚀 Cas Pratiques":
        show_practical_cases()

## === PAGE D'ACCUEIL ===
def show_homepage():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("🤖 AI Marketing Explorer")
        st.subheader("Maîtrisez l'Intelligence Artificielle pour Transformer votre Marketing")
        
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 20px 0;'>
        <h3 style='color: #1f77b4;'>🎯 Objectif de cette Application</h3>
        <p>Cette application vous guide à travers les concepts clés de l'IA appliquée au marketing, 
        avec des démonstrations interactives et des cas concrets.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://cdn.pixabay.com/photo/2019/08/06/22/48/artificial-intelligence-4389372_1280.jpg", 
                use_column_width=True)

    # Les Trois D de l'IA
    st.markdown("---")
    st.header("🧠 Les Trois D de l'IA en Marketing")
    
    tab1, tab2, tab3 = st.tabs(["🔍 Détecter", "⚖️ Délibérer", "🚀 Développer"])
    
    with tab1:
        st.subheader("Détecter - Comprendre les Patterns")
        st.markdown("""
        - Analyser le comportement des clients
        - Identifier les tendances émergentes
        - Détecter les anomalies et opportunités
        """)
        st.info("Exemple : Reconnaissance des intentions d'achat par navigation web")
        
    with tab2:
        st.subheader("Délibérer - Prendre les Meilleures Décisions")
        st.markdown("""
        - Évaluer les options optimales
        - Prédire les résultats
        - Optimiser les ressources
        """)
        st.info("Exemple : Choix du canal marketing le plus efficace pour un segment")
        
    with tab3:
        st.subheader("Développer - Améliorer en Continu")
        st.markdown("""
        - Ajuster les stratégies en temps réel
        - Personnaliser l'expérience client
        - Innover constamment
        """)
        st.info("Exemple : Adaptation automatique des campagnes basée sur les performances")

    # Pourquoi l'IA en Marketing ?
    st.markdown("---")
    
    st.header("💡 Pourquoi l'IA en Marketing ?")
    
    

    cols = st.columns(4)
    benefits = [
    ("🤖", "Automatisation", "Libérez du temps pour la stratégie"),
    ("📊", "Data-Driven", "Décisions basées sur les données"),
    ("🎯", "Personnalisation", "Expériences sur mesure à grande échelle"),
    ("⚡", "Avantage Concurrentiel", "Restez en avance sur le marché")
]
    
    for col, (icon, title, desc) in zip(cols, benefits):
        with col:
            # st.metric(title, desc, icon)
            st.markdown(
            f"""
            <div style='
                text-align: center; 
                padding: 1rem; 
                border-radius: 10px; 
                background: #f8f9fa; 
                border: 1px solid #e9ecef;
                margin: 0.5rem;
                height: 180px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            '>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>{icon}</div>
                <h4 style='margin: 0.5rem 0; color: #2c3e50; font-size: 1.1rem;'>{title}</h4>
                <p style='font-size: 0.85rem; margin: 0; color: #6c757d; line-height: 1.4;'>{desc}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

## === FONCTIONS ML FONDAMENTALS ===
def show_ml_fundamentals():
    st.title("🤖 Fondamentaux du Machine Learning")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Concepts de Base", 
        "🎯 Apprentissage Supervisé", 
        "🔍 Apprentissage Non-Supervisé",
        "🧠 Réseaux Neuronaux"
    ])
    
    with tab1:
        show_ml_concepts()
    
    with tab2:
        show_supervised_learning()
    
    with tab3:
        show_unsupervised_learning()
    
    with tab4:
        show_neural_networks()

def show_ml_concepts():
    st.header("📚 Qu'est-ce que le Machine Learning ?")
    
    st.markdown("""
    > **« Tous les modèles sont faux, mais certains sont utiles. »** - George Edward Pelham Box
    
    Le Machine Learning (ML) est une technologie qui permet aux systèmes **d'apprendre à partir de données** 
    sans être explicitement programmés pour chaque tâche.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Différence avec les Statistiques Traditionnelles")
        st.markdown("""
        - **Statistiques** : Comprendre les données, tester des hypothèses
        - **ML** : Prédire des résultats, automatiser des décisions
        - **IA** : Résoudre des problèmes complexes de manière "intelligente"
        """)
    
    with col2:
        st.subheader("📈 Évolution des Données vs Complexité")
        data = {
            "Approche": ["Règles Métier", "Statistiques", "Machine Learning", "Deep Learning"],
            "Volume Données": [1, 3, 8, 10],
            "Complexité Problème": [2, 5, 8, 10]
        }
        df = pd.DataFrame(data)
        
        fig = px.scatter(df, x="Volume Données", y="Complexité Problème", 
                        text="Approche", size=[20, 30, 40, 50],
                        title="Évolution des Approches Data")
        st.plotly_chart(fig, use_container_width=True)

def show_supervised_learning():
    st.header("🎯 Apprentissage Supervisé")
    
    st.subheader("Classification vs Régression")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Classification** : Catégoriser dans des classes")
        st.markdown("""
        - Email : Spam / Non-Spam
        - Client : Fidèle / À risque
        - Produit : Populaire / Niche
        """)
        
        # Démo Classification
        st.subheader("🎮 Démo : Arbre de Décision pour Assurance Moto")
        
        age = st.slider("Âge du client", 18, 70, 30)
        ville = st.selectbox("Type de ville", ["Rural", "Urbain"])
        score_credit = st.slider("Score de crédit", 300, 850, 650)
        
        # Logique de décision simplifiée
        if age < 25:
            decision = "❌ Risque élevé - Assurance refusée"
            reason = "Jeunes conducteurs à haut risque"
        elif ville == "Rural" and score_credit < 600:
            decision = "⚠️ Conditionnel - Prime majorée"
            reason = "Zone rurale + score de crédit faible"
        else:
            decision = "✅ Accepté - Prime standard"
            reason = "Profil favorable"
        
        st.success(f"**Décision :** {decision}")
        st.write(f"**Raison :** {reason}")
        
        # Visualisation de l'arbre
        st.subheader("📊 Structure de l'Arbre de Décision")
        dot = graphviz.Digraph()
        dot.edge('Profil Client', 'Âge < 25?')
        dot.edge('Âge < 25?', 'Refusé', label='Oui')
        dot.edge('Âge < 25?', 'Type Ville?', label='Non')
        dot.edge('Type Ville?', 'Score Crédit < 600?', label='Rural')
        dot.edge('Type Ville?', 'Accepté', label='Urbain')
        dot.edge('Score Crédit < 600?', 'Conditionnel', label='Oui')
        dot.edge('Score Crédit < 600?', 'Accepté', label='Non')
        
        st.graphviz_chart(dot)
    
    with col2:
        st.success("**Régression** : Prédire une valeur numérique")
        st.markdown("""
        - Prévision des ventes
        - Prédiction du CLV (Customer Lifetime Value)
        - Estimation du prix optimal
        """)
        
        # Démo Régression
        st.subheader("📈 Prédiction du CLV (Customer Lifetime Value)")
        
        avg_purchase = st.slider("Panier moyen (€)", 50, 500, 150)
        frequency = st.slider("Achats par mois", 0.5, 10.0, 2.0)
        retention = st.slider("Taux de rétention annuel (%)", 10, 90, 60) / 100
        
        # Calcul CLV simplifié
        clv = (avg_purchase * frequency * 12) * (retention / (1 - retention))
        
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode = "number+delta",
            value = clv,
            number = {'prefix': "€"},
            title = {"text": "CLV Prédit<br><span style='font-size:0.8em;color:gray'>Valeur à Vie du Client</span>"},
            domain = {'row': 0, 'column': 0}
        ))
        
        fig.update_layout(
            grid = {'rows': 1, 'columns': 1, 'pattern': "independent"},
            height=200
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique des composantes du CLV
        components = {
            'Composante': ['Panier Moyen', 'Fréquence', 'Rétention'],
            'Valeur': [avg_purchase, frequency * 12, retention * 100]
        }
        df_comp = pd.DataFrame(components)
        
        fig_bar = px.bar(df_comp, x='Composante', y='Valeur', 
                        title="Composantes du CLV",
                        color='Composante')
        st.plotly_chart(fig_bar, use_container_width=True)

def show_unsupervised_learning():
    st.header("🔍 Apprentissage Non-Supervisé")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Clustering - Segmentation Client")
        
        # Générer des données clients simulées
        np.random.seed(42)
        n_customers = 200
        
        data = {
            'age': np.random.normal(45, 15, n_customers),
            'income': np.random.normal(50000, 20000, n_customers),
            'spending_score': np.random.normal(50, 25, n_customers)
        }
        df = pd.DataFrame(data)
        df['age'] = df['age'].clip(18, 80)
        df['income'] = df['income'].clip(20000, 100000)
        df['spending_score'] = df['spending_score'].clip(1, 100)
        
        # Application k-means simplifié
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=4, random_state=42)
        df['cluster'] = kmeans.fit_predict(df[['age', 'income', 'spending_score']])
        
        fig = px.scatter_3d(df, x='age', y='income', z='spending_score',
                           color='cluster', title="Segmentation Client 3D",
                           labels={'age': 'Âge', 'income': 'Revenu (€)', 'spending_score': 'Score Dépenses'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Segments identifiés :**
        - 🔵 Jeunes à fort potentiel
        - 🟢 Clients fidèles moyens
        - 🟡 Seniors aisés
        - 🔴 Clients à risque
        """)
    
    with col2:
        st.subheader("🛒 Analyse d'Association - Panier d'Achat")
        
        st.markdown("Découvrez quels produits sont souvent achetés ensemble :")
        
        products = st.multiselect(
            "Sélectionnez des produits dans le panier :",
            ["Lait", "Pain", "Œufs", "Fromage", "Café", "Beurre", "Jus d'orange", "Céréales"],
            default=["Lait", "Pain"]
        )
        
        # Règles d'association simulées
        rules = {
            ("Lait", "Pain"): {"support": 0.15, "confidence": 0.7},
            ("Lait", "Œufs"): {"support": 0.12, "confidence": 0.6},
            ("Pain", "Beurre"): {"support": 0.08, "confidence": 0.5},
            ("Café", "Lait"): {"support": 0.10, "confidence": 0.65}
        }
        
        if len(products) >= 2:
            st.subheader("📈 Règles d'Association Trouvées")
            
            found_rules = []
            for rule, metrics in rules.items():
                if all(p in products for p in rule):
                    found_rules.append({
                        'Règle': f"{rule[0]} → {rule[1]}",
                        'Support': f"{metrics['support']*100:.1f}%",
                        'Confiance': f"{metrics['confidence']*100:.1f}%"
                    })
            
            if found_rules:
                df_rules = pd.DataFrame(found_rules)
                st.dataframe(df_rules, use_container_width=True)
                
                # Visualisation
                fig_rules = px.bar(df_rules, x='Règle', y='Confiance',
                                 title="Confiance des Règles d'Association",
                                 color='Support')
                st.plotly_chart(fig_rules, use_container_width=True)
            else:
                st.warning("Aucune règle forte trouvée pour cette combinaison.")

def show_neural_networks():
    st.header("🧠 Réseaux Neuronaux et Deep Learning")
    
    st.markdown("""
    Les réseaux neuronaux imitent le fonctionnement du cerveau humain avec des « neurones » artificiels 
    organisés en couches.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏗️ Architecture d'un Réseau Neuronal")
        
        # Visualisation interactive du réseau
        layers = st.slider("Nombre de couches cachées", 1, 5, 3)
        neurons = st.slider("Neurones par couche", 2, 20, 8)
        
        # Créer une visualisation simplifiée
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Couche d'entrée
        input_neurons = 4
        for i in range(input_neurons):
            circle = plt.Circle((1, i+1), 0.1, color='blue', alpha=0.6)
            ax.add_patch(circle)
            plt.text(1, i+1, f"I{i+1}", ha='center', va='center', fontsize=8)
        
        # Couches cachées
        for layer in range(layers):
            x_pos = 2 + layer
            for i in range(neurons):
                circle = plt.Circle((x_pos, i+1), 0.1, color='green', alpha=0.6)
                ax.add_patch(circle)
                plt.text(x_pos, i+1, f"H{layer+1}_{i+1}", ha='center', va='center', fontsize=6)
        
        # Couche de sortie
        output_neurons = 2
        for i in range(output_neurons):
            circle = plt.Circle((2 + layers, i+1.5), 0.1, color='red', alpha=0.6)
            ax.add_patch(circle)
            plt.text(2 + layers, i+1.5, f"O{i+1}", ha='center', va='center', fontsize=8)
        
        ax.set_xlim(0, 3 + layers)
        ax.set_ylim(0, max(input_neurons, neurons, output_neurons) + 1)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title(f"Réseau Neuronal : {layers} couches cachées")
        st.pyplot(fig)
    
    with col2:
        st.subheader("🎯 Application : Optimisation d'Email Marketing")
        
        st.markdown("""
        **Comment un réseau neuronal choisit le meilleur objet d'email :**
        """)
        
        user_segment = st.selectbox("Segment client :", 
                                  ["Nouveaux clients", "Clients fidèles", "Clients inactifs"])
        
        # Simulation de prédiction
        if user_segment == "Nouveaux clients":
            best_subject = "🎁 Bienvenue ! Profitez de -20% sur votre première commande"
            confidence = 0.87
        elif user_segment == "Clients fidèles":
            best_subject = "👑 Offre exclusive pour nos meilleurs clients"
            confidence = 0.92
        else:
            best_subject = "📱 Nous vous avons manqué ? Re-découvrez nos nouveautés"
            confidence = 0.78
        
        st.success(f"**Objet recommandé :** {best_subject}")
        st.metric("Confiance de la prédiction", f"{confidence*100:.1f}%")
        
        # Graphique de performance
        segments = ["Nouveaux", "Fidèles", "Inactifs"]
        scores = [0.87, 0.92, 0.78]
        
        fig_perf = px.bar(x=segments, y=scores, 
                         title="Performance des Objets par Segment",
                         labels={'x': 'Segment', 'y': 'Taux d\'Ouverture Prédit'})
        fig_perf.update_traces(marker_color=['blue', 'green', 'orange'])
        st.plotly_chart(fig_perf, use_container_width=True)

## === CONTINUATION DANS LE PROCHAIN MESSAGE ===
# Les autres fonctions (show_marketing_problems, show_attention_capture, etc.) 
# seront définies dans la suite du code
## === SECTION PROBLEMES MARKETING ===
## === SECTION PROBLEMES MARKETING ===
def show_marketing_problems():
    st.title("🎯 Résoudre les Problèmes Marketing avec l'IA")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Évolution du Marketing", 
        "🎢 Parcours Client", 
        "💰 Calculateur CLV",
        "🔍 Redéfinir les Problèmes"
    ])
    
    with tab1:
        show_marketing_evolution()
    
    with tab2:
        show_customer_journey()
    
    with tab3:
        show_clv_calculator()
    
    with tab4:
        show_problem_reframing()

def show_marketing_evolution():
    st.header("📈 L'Évolution du Marketing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Des 4P aux 4C")
        st.markdown("""
        **Marketing Traditionnel (4P) :**
        - **Produit** : Caractéristiques techniques
        - **Prix** : Coût de production + marge
        - **Place** : Canaux de distribution
        - **Promotion** : Publicité de masse
        
        **Marketing Moderne (4C) :**
        - **Client** : Besoins et expériences
        - **Coût** : Valeur perçue
        - **Convenance** : Facilité d'achat
        - **Communication** : Dialogue interactif
        """)
        
        # Timeline interactive
        st.subheader("🕰️ Évolution Chronologique")
        
        eras = {
            "1950-1980": "Marketing de Masse\n• Publicité TV/Radio\n• One-size-fits-all",
            "1980-2000": "Marketing Relationnel\n• Bases de données\n• Fidélisation",
            "2000-2015": "Marketing Digital\n• Sites web\n• Email marketing\n• SEO",
            "2015-Présent": "Marketing Intelligent\n• IA et ML\n• Personalisation\n• Prédiction"
        }
        
        selected_era = st.selectbox("Choisissez une période :", list(eras.keys()))
        st.info(f"**{selected_era}**\n\n{eras[selected_era]}")

    with col2:
        st.subheader("📊 Les Défis des Marketeurs Modernes")
        
        challenges = [
            "📱 Multiplicité des canaux",
            "⏱️ Attentes de réponse immédiate", 
            "🎯 Personnalisation à grande échelle",
            "📈 Mesure du ROI précis",
            "🔮 Prédiction des tendances",
            "🤖 Automatisation intelligente"
        ]
        
        for challenge in challenges:
            st.write(f"- {challenge}")
        
        # Graphique des préoccupations
        st.subheader("📋 Préoccupations des Marketeurs")
        
        concerns_data = {
            'Préoccupation': ['Qualification leads', 'Engagement', 'Conversion', 
                             'Fidélisation', 'CLV', 'ROI'],
            'Importance': [8.5, 9.2, 9.0, 8.0, 7.5, 9.5]
        }
        df_concerns = pd.DataFrame(concerns_data)
        
        fig = px.bar(df_concerns, x='Importance', y='Préoccupation', 
                    orientation='h', title="Priorités Marketing",
                    color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

def show_customer_journey():
    st.header("🎢 Parcours Client Non-Linéaire")
    
    st.markdown("""
    Le parcours client moderne n'est plus linéaire mais un écosystème complexe 
    où les clients naviguent entre différents touchpoints.
    """)
    
    # Création du graphique de parcours client interactif
    fig = go.Figure()
    
    # Points du parcours
    journey_points = {
        'Prise de Conscience': {'x': 1, 'y': 5, 'color': 'blue'},
        'Consideration': {'x': 2, 'y': 3, 'color': 'green'},
        'Achat': {'x': 3, 'y': 5, 'color': 'orange'},
        'Expérience': {'x': 4, 'y': 2, 'color': 'red'},
        'Fidélité': {'x': 5, 'y': 4, 'color': 'purple'},
        'Advocacy': {'x': 6, 'y': 6, 'color': 'brown'}
    }
    
    # Connexions non-linéaires
    connections = [
        ('Prise de Conscience', 'Consideration'),
        ('Consideration', 'Achat'),
        ('Achat', 'Expérience'),
        ('Expérience', 'Fidélité'),
        ('Fidélité', 'Advocacy'),
        ('Advocacy', 'Prise de Conscience'),  # Boucle
        ('Consideration', 'Expérience'),      # Saut
        ('Fidélité', 'Consideration')         #Retour
    ]
    
    # Ajouter les connexions
    for start, end in connections:
        fig.add_trace(go.Scatter(
            x=[journey_points[start]['x'], journey_points[end]['x']],
            y=[journey_points[start]['y'], journey_points[end]['y']],
            mode='lines',
            line=dict(color='gray', width=2, dash='dot'),
            showlegend=False
        ))
    
    # Ajouter les points
    for point, info in journey_points.items():
        fig.add_trace(go.Scatter(
            x=[info['x']],
            y=[info['y']],
            mode='markers+text',
            marker=dict(size=30, color=info['color']),
            text=[point],
            textposition="middle center",
            name=point,
            hovertemplate=f"<b>{point}</b><extra></extra>"
        ))
    
    fig.update_layout(
        title="Parcours Client Dynamique - Modèle de Brian Solis",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=500,
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Sélecteur d'étapes pour voir les actions IA
    st.subheader("🎯 Actions IA par Étape")
    
    selected_stage = st.selectbox(
        "Sélectionnez une étape du parcours :",
        list(journey_points.keys())
    )
    
    ia_actions = {
        'Prise de Conscience': [
            "🎯 Publicité programmatique ciblée",
            "🔍 Optimisation SEO avec NLP",
            "📱 Campagnes social media intelligentes"
        ],
        'Consideration': [
            "🤖 Chatbots pour qualification",
            "📧 Personalisation d'emails",
            "🎯 Retargeting dynamique"
        ],
        'Achat': [
            "💰 Pricing dynamique",
            "📦 Recommandations de produits",
            "⚡ Optimisation du checkout"
        ],
        'Expérience': [
            "😊 Analyse de sentiment",
            "🔧 Support automatisé",
            "⭐ Personalisation post-achat"
        ],
        'Fidélité': [
            "📊 Prédiction de churn",
            "🎁 Programmes de fidélité intelligents",
            "🔔 Alertes de ré-engagement"
        ],
        'Advocacy': [
            "🌟 Détection d'influenceurs",
            "📢 Génération de contenu UGC",
            "🔍 Surveillance de réputation"
        ]
    }
    
    st.info(f"**Actions IA pour '{selected_stage}':**")
    for action in ia_actions[selected_stage]:
        st.write(f"- {action}")

def show_clv_calculator():
    st.header("💰 Calculateur de Customer Lifetime Value (CLV)")
    
    st.markdown("""
    La Valeur à Vie du Client (CLV) mesure le profit total qu'un client génère 
    pendant toute sa relation avec votre entreprise.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Paramètres du Calcul")
        
        # Inputs utilisateur
        avg_order_value = st.number_input(
            "Panier moyen (€)", 
            min_value=10, 
            max_value=1000, 
            value=150,
            help="Montant moyen dépensé par commande"
        )
        
        purchase_frequency = st.slider(
            "Fréquence d'achat (commandes/mois)",
            min_value=0.1,
            max_value=20.0,
            value=2.0,
            step=0.1
        )
        
        customer_lifespan = st.slider(
            "Durée de vie moyenne (années)",
            min_value=0.5,
            max_value=10.0,
            value=3.0,
            step=0.5
        )
        
        profit_margin = st.slider(
            "Marge bénéficiaire moyenne (%)",
            min_value=5,
            max_value=50,
            value=30
        ) / 100
        
        acquisition_cost = st.number_input(
            "Coût d'acquisition client (CAC) (€)",
            min_value=0,
            max_value=500,
            value=50
        )
    
    with col2:
        st.subheader("📈 Résultats")
        
        # Calculs
        annual_revenue = avg_order_value * purchase_frequency * 12
        total_revenue = annual_revenue * customer_lifespan
        gross_profit = total_revenue * profit_margin
        clv = gross_profit - acquisition_cost
        cac_ratio = clv / acquisition_cost if acquisition_cost > 0 else 0
        
        # Affichage des métriques
        st.metric("CLV Brut", f"€{gross_profit:,.0f}")
        st.metric("CLV Net", f"€{clv:,.0f}")
        st.metric("Ratio CLV/CAC", f"{cac_ratio:.1f}x")
        
        # Interprétation du ratio
        if cac_ratio > 3:
            st.success("✅ Excellent ratio CLV/CAC")
        elif cac_ratio > 1:
            st.warning("⚠️ Ratio acceptable mais perfectible")
        else:
            st.error("❌ Problématique : CLV < CAC")
    
    # Visualisations
    st.subheader("📊 Analyse du CLV")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Breakdown du CLV
        components = {
            'Composante': ['Revenu Annuel', 'Durée de Vie', 'Marge', 'CAC'],
            'Valeur': [annual_revenue, customer_lifespan, profit_margin, -acquisition_cost],
            'Type': ['Revenu', 'Temps', 'Pourcentage', 'Coût']
        }
        df_components = pd.DataFrame(components)
        
        fig_breakdown = px.bar(df_components, x='Composante', y='Valeur',
                              color='Type', title="Décomposition du CLV",
                              color_discrete_map={'Revenu': 'blue', 'Temps': 'green', 
                                                'Pourcentage': 'orange', 'Coût': 'red'})
        st.plotly_chart(fig_breakdown, use_container_width=True)
    
    with col4:
        # Projection temporelle
        years = list(range(1, int(customer_lifespan) + 1))
        cumulative_profit = [annual_revenue * profit_margin * year - acquisition_cost for year in years]
        
        fig_projection = px.line(
            x=years, y=cumulative_profit,
            title="CLV Cumulatif dans le Temps",
            labels={'x': 'Années', 'y': 'Profit Cumulatif (€)'}
        )
        fig_projection.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_projection, use_container_width=True)
    
    # Recommandations basées sur le CLV
    st.subheader("🎯 Recommandations Stratégiques")
    
    if clv > 1000:
        st.success("""
        **Stratégie Premium :**
        - Investissez dans la fidélisation
        - Développez des programmes VIP
        - Personnalisation haut de gamme
        """)
    elif clv > 100:
        st.info("""
        **Stratégie Croissance :**
        - Optimisez l'acquisition
        - Améliorez l'expérience client
        - Développez les ventes croisées
        """)
    else:
        st.warning("""
        **Stratégie Efficiency :**
        - Réduisez le CAC
        - Augmentez la fréquence d'achat
        - Travaillez sur la rétention
        """)

def show_problem_reframing():
    st.header("🔍 Redéfinir les Problèmes Marketing")
    
    st.markdown("""
    > **« Si j'avais une heure pour résoudre un problème, je passerais 55 minutes à réfléchir au problème 
    > et 5 minutes à réfléchir aux solutions. »** - Albert Einstein
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚪 L'Exemple de l'Ascenseur Lent")
        
        st.markdown("""
        **Problème initial :** "L'ascenseur est trop lent"
        
        **Solutions envisagées :**
        - Remplacer le moteur (€€€€)
        - Installer un nouvel ascenseur (€€€€€)
        - Optimiser l'algorithme (€€)
        
        **Redéfinition du problème :** "Les gens s'ennuient en attendant"
        
        **Solution innovante :** Installer des miroirs dans l'ascenseur (€)
        """)
        
        st.success("**Résultat :** Les plaintes ont diminué de 80%")
        
        # Application interactive
        st.subheader("🎮 Redéfinissez Votre Problème")
        
        user_problem = st.text_area(
            "Décrivez votre problème marketing :",
            "Mes campagnes email ont un faible taux d'ouverture"
        )
        
        if st.button("🔍 Redéfinir le Problème"):
            st.info("**Questions pour redéfinir le problème :**")
            st.write("1. Quel est le vrai objectif derrière ce problème ?")
            st.write("2. Pourquoi les clients ne lisent-ils pas nos emails ?")
            st.write("3. Que cherchent-ils vraiment à accomplir ?")
            st.write("4. Comment pourrions-nous communiquer cette information autrement ?")
    
    with col2:
        st.subheader("🔄 Cadres de Redéfinition")
        
        framework = st.selectbox(
            "Choisissez un cadre de réflexion :",
            ["Les 5 Pourquoi", "Inversion", "Changement de Perspective"]
        )
        
        if framework == "Les 5 Pourquoi":
            st.markdown("""
            **Exemple : Taux de conversion faible**
            1. Pourquoi ? → Le processus d'achat est compliqué
            2. Pourquoi ? → Trop d'étapes de validation
            3. Pourquoi ? → Craintes de fraude excessives
            4. Pourquoi ? → Système de détection obsolète
            5. Pourquoi ? → Pas d'investissement en tech
            """)
            
        elif framework == "Inversion":
            st.markdown("""
            **Au lieu de :** "Comment augmenter nos ventes ?"
            **Demandez :** "Comment pourrions-nous perdre tous nos clients ?"
            
            **Réponses possibles :**
            - Ignorer leurs feedbacks
            - Rendre le site inaccessible
            - Augmenter les prix sans valeur ajoutée
            """)
            
        else:  # Changement de Perspective
            st.markdown("""
            **Imaginez que vous êtes :**
            - Un client de 70 ans vs 20 ans
            - Un concurrent
            - Un influenceur dans votre domaine
            
            **Que verriez-vous différemment ?**
            """)
        
        # Matrice de redéfinition
        st.subheader("📋 Matrice de Redéfinition")
        
        problems = [
            "Faible engagement social media",
            "Taux d'abandon panier élevé", 
            "Désabonnements emails fréquents",
            "Faible rétention clients"
        ]
        
        reframed = [
            "Comment créer du contenu que les gens VEULENT partager?",
            "Comment rendre l'achat plus agréable que l'abandon?",
            "Comment apporter tellement de valeur qu'ils auraient peur de manquer quelque chose?",
            "Comment devenir indispensable dans leur vie quotidienne?"
        ]
        
        df_reframe = pd.DataFrame({
            'Problème Initial': problems,
            'Problème Redéfini': reframed
        })
        
        st.dataframe(df_reframe, use_container_width=True)

## === SECTION CAPTURER L'ATTENTION ===
def show_attention_capture():
    st.title("📢 Utiliser l'IA pour Capter l'Attention")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Recherche Marketing", 
        "🎯 Segmentation", 
        "😊 Analyse de Sentiment",
        "⚡ Publicité Programmatique"
    ])
    
    with tab1:
        show_marketing_research()
    
    with tab2:
        show_customer_segmentation()
    
    with tab3:
        show_sentiment_analysis()
    
    with tab4:
        show_programmatic_advertising()

def show_marketing_research():
    st.header("🔍 Recherche Marketing Intelligente")
    
    st.markdown("""
    L'IA transforme la recherche marketing en analysant des volumes massifs de données 
    pour identifier des insights actionnables.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Exemple : Lucy (IBM Watson)")
        
        st.markdown("""
        **Capacités :**
        - Analyse de données structurées et non-structurées
        - Réponses à des questions complexes en langage naturel
        - Identification de patterns invisibles à l'œil humain
        """)
        
        # Simulation de questions à Lucy
        st.subheader("💬 Posez une Question à Lucy")
        
        research_question = st.selectbox(
            "Choisissez une question de recherche :",
            [
                "Quels sont les segments émergents pour nos produits?",
                "Comment notre marque est-elle perçue vs nos concurrents?",
                "Quelles fonctionnalités les clients souhaitent-ils?",
                "Quels canaux sont les plus efficaces pour atteindre les millennials?"
            ]
        )
        
        if st.button("🔄 Analyser avec Lucy"):
            with st.spinner("Lucy analyse les données..."):
                import time
                time.sleep(2)
                
                st.success("**Analyse de Lucy :**")
                st.write("""
                - **Segments identifiés :** 3 nouveaux clusters détectés
                - **Sentiment global :** Positif (72%) avec opportunités d'amélioration
                - **Recommandations :** 
                  - Cibler les professionnels jeunes urbains
                  - Développer la gamme premium
                  - Renforcer la présence sur TikTok
                """)
    
    with col2:
        st.subheader("📊 Sources de Données Analysées")
        
        data_sources = [
            ("📱 Médias Sociaux", "Analyse de sentiment, tendances émergentes"),
            ("🌐 Reviews en Ligne", "Feedback produit, points de douleur"),
            ("📈 Données de Vente", "Patterns d'achat, saisonnalité"),
            ("🔍 Données Web", "Comportement navigation, taux de conversion"),
            ("📋 Enquêtes", "Perceptions, préférences déclarées"),
            ("📞 Service Client", "Problèmes récurrents, demandes")
        ]
        
        for source, description in data_sources:
            with st.expander(f"{source} - {description}"):
                st.write(f"**Applications IA :** Classification automatique, analyse thématique, prédiction de tendances")
        
        # Visualisation des insights
        st.subheader("📈 Insights Détectés")
        
        insights_data = {
            'Insight': ['Nouveau besoin mobile', 'Prix perçu élevé', 'Demande sustainability', 'Service client lent'],
            'Confiance': [85, 78, 92, 67],
            'Impact': [8, 6, 7, 9]
        }
        df_insights = pd.DataFrame(insights_data)
        
        fig = px.scatter(df_insights, x='Impact', y='Confiance', size='Confiance',
                        text='Insight', title="Matrice Impact vs Confiance des Insights",
                        color='Confiance', color_continuous_scale='Viridis')
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

def show_customer_segmentation():
    st.header("🎯 Segmentation Client par IA")
    
    st.markdown("""
    L'IA permet une segmentation dynamique et multi-dimensionnelle des clients 
    basée sur leur comportement réel plutôt que des caractéristiques démographiques simples.
    """)
    
    # Génération de données clients
    df = generate_customer_data(500)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Paramètres de Segmentation")
        
        n_clusters = st.slider("Nombre de segments", 2, 8, 4)
        
        segmentation_type = st.radio(
            "Type de segmentation :",
            ["Comportementale", "Valeur", "Engagement", "Mixte"]
        )
        
        # Features selon le type
        if segmentation_type == "Comportementale":
            features = ['spending_score', 'total_purchases', 'days_since_last_purchase']
        elif segmentation_type == "Valeur":
            features = ['income', 'spending_score', 'total_purchases']
        elif segmentation_type == "Engagement":
            features = ['days_since_last_purchase', 'total_purchases', 'spending_score']
        else:  # Mixte
            features = ['age', 'income', 'spending_score', 'total_purchases']
        
        # Application du clustering
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['segment'] = kmeans.fit_predict(X_scaled)
        
        # Description des segments
        segment_profiles = df.groupby('segment').agg({
            'age': 'mean',
            'income': 'mean', 
            'spending_score': 'mean',
            'total_purchases': 'mean',
            'days_since_last_purchase': 'mean'
        }).round(1)
        
        st.subheader("📋 Profils des Segments")
        st.dataframe(segment_profiles, use_container_width=True)
    
    with col2:
        st.subheader("📊 Visualisation des Segments")
        
        # Choix des axes pour la visualisation
        x_axis = st.selectbox("Axe X", features, index=0)
        y_axis = st.selectbox("Axe Y", features, index=1)
        
        fig = px.scatter(df, x=x_axis, y=y_axis, color='segment',
                        title=f"Segmentation Client - {segmentation_type}",
                        hover_data=['age', 'income'],
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommandations par segment
        st.subheader("🎯 Stratégies par Segment")
        
        segment_strategies = {
            0: "🎯 **Segment Basique** : Campagnes d'acquisition, offres découverte",
            1: "💰 **Segment Valeur** : Programmes fidélité, ventes croisées", 
            2: "⭐ **Segment Premium** : Services personnalisés, produits exclusifs",
            3: "⚠️ **Segment À Risque** : Campagnes de réactivation, enquêtes de satisfaction"
        }
        
        for segment, strategy in list(segment_strategies.items())[:n_clusters]:
            st.write(strategy)
        
        # Téléchargement des segments
        st.download_button(
            label="📥 Télécharger les Segments",
            data=df.to_csv(index=False),
            file_name="segments_clients.csv",
            mime="text/csv"
        )


def show_sentiment_analysis():
    st.markdown("<h2 class='section-header'>😊 Analyse de Sentiment par IA</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>
    <p>Analyse automatique des opinions clients à partir de commentaires, reviews et conversations sociales.</p>
    </div>
    """, unsafe_allow_html=True)
    
    text_input = st.text_area("Entrez un texte à analyser :", "Ce produit est vraiment excellent !")
    
    if st.button("Analyser le sentiment"):
        # Analyse simplifiée
        if "excellent" in text_input.lower() or "super" in text_input.lower():
            st.success("**Sentiment : Positif 😊**")
        elif "mauvais" in text_input.lower() or "nul" in text_input.lower():
            st.error("**Sentiment : Négatif 😠**")
        else:
            st.info("**Sentiment : Neutre 😐**")

def show_programmatic_advertising():
    st.header("⚡ Publicité Programmatique avec IA")
    
    st.markdown("""
    La publicité programmatique utilise l'IA pour automatiser l'achat d'espaces publicitaires 
    en temps réel, optimisant le ROI grâce au machine learning.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Comment ça marche ?")
        
        st.markdown("""
        **Processus en Temps Réel (RTB - Real-Time Bidding) :**
        
        1. **Utilisateur** visite un site web
        2. **SSP** (Sell-Side Platform) envoie une opportunité d'impression
        3. **DSP** (Demand-Side Platform) évalue la valeur de l'utilisateur
        4. **Enchères** ont lieu en millisecondes
        5. **Meilleure offre** remporte l'impression
        6. **Publicité** s'affiche pour l'utilisateur
        """)
        
        # Simulation d'enchère
        st.subheader("🎮 Simulateur d'Enchère")
        
        user_segment = st.selectbox(
            "Segment utilisateur cible :",
            ["Jeunes actifs urbains", "Familles suburbanes", "Retraités aisés", "Étudiants"]
        )
        
        campaign_budget = st.slider("Budget de campagne (€)", 100, 5000, 1000)
        
        if st.button("🎯 Lancer l'Enchère"):
            from utils.marketing_utils import simulate_ad_auction
            
            auction_result = simulate_ad_auction(campaign_budget)
            
            st.info(f"**Segment :** {user_segment}")
            st.metric("Votre offre", f"€{auction_result['user_bid']:.2f}")
            st.metric("Offre gagnante", f"€{auction_result['winning_bid']:.2f}")
            
            if auction_result['user_won']:
                st.success("🎉 Vous avez remporté l'enchère !")
                st.balloons()
            else:
                st.error("💸 Vous avez perdu l'enchère...")
            
            # Visualisation des offres
            bids_df = pd.DataFrame({
                'Enchérisseur': ['Vous'] + [f'Concurrent {i+1}' for i in range(len(auction_result['competitor_bids']))],
                'Offre': [auction_result['user_bid']] + list(auction_result['competitor_bids'])
            })
            
            fig_bids = px.bar(bids_df, x='Enchérisseur', y='Offre', 
                             title="Comparaison des Offres",
                             color='Offre', color_continuous_scale='Viridis')
            st.plotly_chart(fig_bids, use_container_width=True)
    
    with col2:
        st.subheader("📊 Optimisation par IA")
        
        st.markdown("""
        **Comment l'IA optimise les campagnes :**
        
        - **Bid Shading** : Ajustement automatique des offres
        - **Audience Targeting** : Identification des profils à fort potentiel
        - **Creative Optimization** : Test automatique des visuels
        - **Budget Pacing** : Répartition optimale du budget dans le temps
        """)
        
        # Dashboard de performance
        st.subheader("📈 Performance Campagne")
        
        # Données simulées de performance
        days = list(range(1, 31))
        performance_data = {
            'Jour': days,
            'Impressions': [1000 + i*50 + np.random.normal(0, 100) for i in days],
            'CTR': [0.02 + i*0.0005 + np.random.normal(0, 0.005) for i in days],
            'Coût': [200 + i*10 + np.random.normal(0, 20) for i in days],
            'Conversions': [10 + i*0.5 + np.random.normal(0, 3) for i in days]
        }
        
        df_perf = pd.DataFrame(performance_data)
        df_perf['CPA'] = df_perf['Coût'] / df_perf['Conversions']
        df_perf['ROAS'] = (df_perf['Conversions'] * 50) / df_perf['Coût']  #假设每笔转换价值50€
        
        # Métriques KPIs
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            st.metric("CTR Moyen", f"{df_perf['CTR'].mean():.2%}")
        with col4:
            st.metric("CPA Moyen", f"€{df_perf['CPA'].mean():.1f}")
        with col5:
            st.metric("ROAS Moyen", f"{df_perf['ROAS'].mean():.2f}x")
        with col6:
            st.metric("Coût Total", f"€{df_perf['Coût'].sum():.0f}")
        
        # Graphique d'évolution
        metric_choice = st.selectbox("Metrique à visualiser :", 
                                   ['CTR', 'CPA', 'ROAS', 'Impressions'])
        
        fig_trend = px.line(df_perf, x='Jour', y=metric_choice,
                           title=f"Évolution du {metric_choice}")
        st.plotly_chart(fig_trend, use_container_width=True)

## === SECTION CAS PRATIQUES ===
def show_practical_cases():
    st.title("🚀 Cas Pratiques & Études de Cas")
    
    tab1, tab2, tab3 = st.tabs([
        "🏆 Success Stories", 
        "🛠️ Mise en Œuvre", 
        "🔮 Tendances Futures"
    ])
    
    with tab1:
        show_success_stories()
    
    with tab2:
        show_implementation()
    
    with tab3:
        show_future_trends()

def show_success_stories():
    st.header("🏆 Études de Cas Réelles")
    
    case_studies = [
        {
            "company": "🎯 Netflix",
            "title": "Système de Recommandation",
            "challenge": "Garder les utilisateurs engagés avec du contenu pertinent",
            "solution": "Algorithmes de recommendation basés sur le comportement de visionnage",
            "results": "80% du contenu visionné provient des recommendations",
            "ia_techniques": ["Filtrage collaboratif", "Deep Learning", "Analyse de séquences"]
        },
        {
            "company": "🛍️ Amazon",
            "title": "Optimisation des Prix Dynamiques",
            "challenge": "Maximiser le revenue tout en restant compétitif",
            "solution": "Système de pricing dynamique basé sur la demande, la concurrence et le comportement client",
            "results": "Augmentation de 25% du revenue sur les produits concernés",
            "ia_techniques": ["Régression", "Optimisation", "Analyse en temps réel"]
        },
        {
            "company": "☕ Starbucks",
            "title": "Personalisation des Offres",
            "challenge": "Augmenter la fréquence des visites en magasin",
            "solution": "Application mobile avec recommendations personnalisées et offres ciblées",
            "results": "40% d'augmentation des transactions via l'app mobile",
            "ia_techniques": ["Segmentation", "Règles d'association", "Analyse géospatiale"]
        }
    ]
    
    for case in case_studies:
        with st.expander(f"{case['company']} - {case['title']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Défi")
                st.write(case['challenge'])
                
                st.subheader("🛠️ Solution IA")
                st.write(case['solution'])
                
            with col2:
                st.subheader("📊 Résultats")
                st.success(case['results'])
                
                st.subheader("🧠 Techniques IA Utilisées")
                for technique in case['ia_techniques']:
                    st.write(f"- {technique}")
    
    # Analyse interactive d'un cas
    st.header("🎮 Analysez Votre Propre Cas")
    
    user_challenge = st.selectbox(
        "Sélectionnez un défi marketing :",
        [
            "Faible taux de conversion site web",
            "Désabonnements emails élevés", 
            "Coût d'acquisition client trop élevé",
            "Faible engagement sur les réseaux sociaux"
        ]
    )
    
    if st.button("💡 Générer des Solutions IA"):
        solutions = {
            "Faible taux de conversion site web": [
                "🤖 Chatbot de qualification pour guider les visiteurs",
                "🎯 Personalisation du contenu basée sur le comportement",
                "⚡ Optimisation A/B testing automatisé des landing pages"
            ],
            "Désabonnements emails élevés": [
                "📊 Segmentation avancée pour un contenu plus pertinent",
                "⏰ Optimisation du timing d'envoi par machine learning",
                "🎨 Génération IA d'objets et contenu personnalisés"
            ],
            "Coût d'acquisition client trop élevé": [
                "🎯 Identification des canaux les plus efficaces par attribution IA",
                "💰 Optimisation des enchères publicitaires programmatiques",
                "🔮 Prédiction des clients à forte valeur potentielle"
            ],
            "Faible engagement sur les réseaux sociaux": [
                "😊 Analyse de sentiment pour comprendre les préférences",
                "🕒 Optimisation du calendrier de publication par IA",
                "🎨 Génération automatique de contenu visuel engageant"
            ]
        }
        
        st.success(f"**Solutions IA pour : {user_challenge}**")
        for solution in solutions[user_challenge]:
            st.write(f"- {solution}")

def show_implementation():
    st.header("🛠️ Guide de Mise en Œuvre")
    
    st.markdown("""
    Implémenter l'IA dans votre marketing nécessite une approche structurée. 
    Voici un guide étape par étape :
    """)
    
    steps = [
        {
            "step": 1,
            "title": "📊 Audit des Données",
            "description": "Évaluez la qualité et la disponibilité de vos données",
            "actions": [
                "Identifier les sources de données internes et externes",
                "Évaluer la qualité et la complétude des données",
                "Mettre en place un plan de gouvernance des données"
            ]
        },
        {
            "step": 2, 
            "title": "🎯 Définition des Cas d'Usage",
            "description": "Identifiez les problèmes business que l'IA peut résoudre",
            "actions": [
                "Prioriser les cas d'usage par impact et faisabilité",
                "Définir les métriques de succès claires",
                "Estimer le ROI potentiel"
            ]
        },
        {
            "step": 3,
            "title": "🤖 Choix des Outils & Technologies",
            "description": "Sélectionnez la stack technologique adaptée à vos besoins",
            "actions": [
                "Évaluer solutions no-code vs développement sur mesure",
                "Choisir entre cloud providers et solutions on-premise",
                "Former les équipes aux nouvelles technologies"
            ]
        },
        {
            "step": 4,
            "title": "🚀 Prototypage & Test",
            "description": "Lancez un projet pilote pour valider l'approche",
            "actions": [
                "Démarrer avec un cas d'usage simple et mesurable",
                "Tester sur un segment limité de clients",
                "Itérer rapidement basé sur les feedbacks"
            ]
        },
        {
            "step": 5,
            "title": "📈 Scale & Industrialisation",
            "description": "Étendez la solution à l'ensemble de l'organisation",
            "actions": [
                "Automatiser les processus de données",
                "Former les équipes métier à l'utilisation des outils",
                "Mettre en place un monitoring continu des performances"
            ]
        }
    ]
    
    for step in steps:
        with st.expander(f"Étape {step['step']}: {step['title']}"):
            st.write(step['description'])
            st.write("**Actions concrètes :**")
            for action in step['actions']:
                st.write(f"- {action}")
    
    # Roadmap personnalisée
    st.subheader("🗓️ Générateur de Roadmap IA")
    
    company_size = st.selectbox("Taille de votre entreprise :", 
                               ["Startup (<50)", "PME (50-500)", "ETI (500-5000)", "Grande Entreprise (>5000)"])
    
    maturity = st.slider("Maturité Data actuelle (1=débutant, 10=avancé)", 1, 10, 3)
    
    if st.button("🎯 Générer Ma Roadmap"):
        st.success("**Votre Roadmap Personnalisée :**")
        
        if maturity <= 3:
            st.write("**Phase 1 (3-6 mois) - Fondations :**")
            st.write("- Mettre en place un CRM et outils analytics de base")
            st.write("- Former les équipes aux concepts data de base")
            st.write("- Identifier 1-2 cas d'usage simples (ex: segmentation basique)")
        
        if 4 <= maturity <= 7:
            st.write("**Phase 2 (6-12 mois) - Expérimentation :**")
            st.write("- Implémenter des outils no-code d'IA marketing")
            st.write("- Lancer des pilotes sur la personalisation email")
            st.write("- Explorer l'analyse de sentiment sur les réseaux sociaux")
        
        if maturity >= 8:
            st.write("**Phase 3 (12+ mois) - Industrialisation :**")
            st.write("- Développer des modèles prédictifs customisés")
            st.write("- Automatiser les processus de décision marketing")
            st.write("- Mettre en place une plateforme data marketing unifiée")

def show_future_trends():
    st.header("🔮 Tendances Futures de l'IA en Marketing")
    
    st.markdown("""
    L'IA continue d'évoluer à un rythme rapide. Voici les tendances qui façonneront 
    le marketing de demain :
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Technologies Émergentes")
        
        trends = [
            {
                "name": "🧠 Generative AI",
                "description": "Création automatique de contenu textuel et visuel",
                "impact": "Élevé",
                "timeline": "Maintenant - 2 ans"
            },
            {
                "name": "🔍 AI Multimodale", 
                "description": "Analyse simultanée de texte, image, audio et vidéo",
                "impact": "Élevé",
                "timeline": "2-3 ans"
            },
            {
                "name": "⚡ Edge AI",
                "description": "Traitement IA en local sur les devices pour plus de rapidité",
                "impact": "Moyen",
                "timeline": "3-5 ans"
            },
            {
                "name": "🤖 Autonomous Marketing",
                "description": "Systèmes marketing entièrement automatisés et auto-optimisants",
                "impact": "Très élevé", 
                "timeline": "5+ ans"
            }
        ]
        
        for trend in trends:
            with st.expander(f"{trend['name']} - Impact: {trend['impact']}"):
                st.write(trend['description'])
                st.write(f"**Horizon :** {trend['timeline']}")
    
    with col2:
        st.subheader("📊 Impact sur les Métiers Marketing")
        
        roles_impact = {
            "📢 Responsable Publicité": "80% de tâches automatisables",
            "✍️ Rédacteur Content": "60% de création assistée par IA", 
            "📈 Analyste Data": "Focus sur l'interprétation vs la collecte",
            "🎯 Responsable Segmentation": "Segmentation dynamique en temps réel",
            "💬 Community Manager": "IA d'assistance pour le scale"
        }
        
        for role, impact in roles_impact.items():
            st.metric(role, impact)
        
        # Visualisation de l'adoption
        st.subheader("📈 Courbe d'Adoption de l'IA Marketing")
        
        adoption_data = {
            'Année': [2020, 2021, 2022, 2023, 2024, 2025, 2026],
            'Adoption (%)': [15, 25, 35, 48, 62, 75, 85]
        }
        df_adoption = pd.DataFrame(adoption_data)
        
        fig_adoption = px.line(df_adoption, x='Année', y='Adoption (%)',
                              title="Adoption de l'IA dans le Marketing",
                              markers=True)
        st.plotly_chart(fig_adoption, use_container_width=True)
    
    # Préparations recommandées
    st.subheader("🎯 Comment Se Préparer ?")
    
    preparations = [
        "**Formation Continue** : Restez à jour sur les nouvelles technologies IA",
        "**Culture Data** : Développez une mindset data-driven dans toute l'organisation", 
        "**Agilité** : Soyez prêt à tester rapidement et itérer",
        "**Éthique** : Mettez en place des guidelines pour l'IA responsable",
        "**Partenariats** : Collaborez avec des experts IA et startups innovantes"
    ]
    
    for prep in preparations:
        st.write(f"- {prep}")
    
    # Quiz interactif
    st.subheader("🎓 Testez Vos Connaissances")
    
    quiz_questions = [
        {
            "question": "Quel type d'apprentissage est utilisé pour la segmentation client?",
            "options": ["Supervisé", "Non-supervisé", "Par renforcement", "Semi-supervisé"],
            "answer": "Non-supervisé"
        },
        {
            "question": "Que mesure la CLV?",
            "options": [
                "Le coût d'acquisition client", 
                "La valeur totale d'un client sur sa relation avec l'entreprise",
                "Le taux de conversion moyen",
                "La satisfaction client"
            ],
            "answer": "La valeur totale d'un client sur sa relation avec l'entreprise"
        }
    ]
    
    score = 0
    for i, q in enumerate(quiz_questions):
        st.write(f"**Question {i+1}:** {q['question']}")
        user_answer = st.radio(f"Choisissez une réponse:", q['options'], key=f"q{i}")
        
        if user_answer == q['answer']:
            score += 1
            st.success("✅ Correct!")
        elif user_answer:
            st.error(f"❌ Incorrect. La réponse est : {q['answer']}")
    
    if st.button("📊 Voir mon score"):
        st.success(f"Votre score : {score}/{len(quiz_questions)}")
        if score == len(quiz_questions):
            st.balloons()
            st.success("🎉 Excellent! Vous maîtrisez les concepts clés!")

# Lancer l'application
if __name__ == "__main__":
    main()
def show_marketing_problems():
    st.title("🎯 Résoudre les Problèmes Marketing avec l'IA")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Évolution du Marketing", 
        "🎢 Parcours Client", 
        "💰 Calculateur CLV",
        "🔍 Redéfinir les Problèmes"
    ])
    
    with tab1:
        show_marketing_evolution()
    
    with tab2:
        show_customer_journey()
    
    with tab3:
        show_clv_calculator()
    
    with tab4:
        show_problem_reframing()


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

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='
        text-align: center;
        padding: 2rem 0;
        color: #666;
        font-size: 0.9rem;
    '>
        <div style='margin-bottom: 0.5rem;'>
            <strong>📧 Contactez-moi</strong>
        </div>
        <div style='margin-bottom: 0.5rem;'>
            Amiharbi Eyeug• 
            <a href='mailto: ibugueye@ngorweb.com' style='color: #ff4b4b; text-decoration: none;'>
                ibugueye@ngorweb.com
            </a>
        </div>
        <div style='margin-bottom: 0.5rem;'>
            <a href='https://www.linkedin.com/in/ibucoumba/' style='color: #ff4b4b; text-decoration: none; margin: 0 10px;'>
                LinkedIn
            </a>
            •
            <a href='https://github.com/ibugueye' style='color: #ff4b4b; text-decoration: none; margin: 0 10px;'>
                GitHub
            </a>
            •
            <a href='https://ngorweb.com' style='color: #ff4b4b; text-decoration: none; margin: 0 10px;'>
                Portfolio
            </a>
        </div>
        <div style='font-size: 0.8rem; color: #999;'>
            © 2024 Amiharbi Eyeug Xataxeli. Tous droits réservés.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


