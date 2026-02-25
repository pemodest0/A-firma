#!/usr/bin/env python3
"""Script para treinar e validar o classificador de regimes."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.regime_classifier import train_regime_classifier, evaluate_classifier

# Diretórios
OUT = Path('results/duffing_validation')
OUT.mkdir(parents=True, exist_ok=True)

def plot_feature_distributions(df, features, target):
    """Plota distribuição das features por classe."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(features):
        if i >= len(axes):
            break
        sns.boxplot(data=df, x='gt_lyap', y=feature, ax=axes[i])
        axes[i].set_title(f'Distribuição de {feature}')
    
    plt.tight_layout()
    plt.savefig(OUT / 'feature_distributions.png')
    plt.close()

def main():
    # Carregar dados do sweep
    df = pd.read_csv(OUT / 'duffing_sweep.csv')
    
    # Preparar features
    features = ['entropy', 'alpha', 'n_clusters', 'lyap']
    X = df[features].fillna(0).values
    y = df['gt_lyap'].values
    
    # Plotar distribuições
    import seaborn as sns
    plot_feature_distributions(df, features, 'gt_lyap')  # usando classificação baseada no expoente de Lyapunov
    
    print("\nDistribuição das classes:")
    print(pd.Series(y).value_counts())
    
    # Dividir em treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Treinar classificador
    print("Treinando classificador...")
    clf, scaler = train_regime_classifier(X_train, y_train)
    
    # Avaliar performance
    print("\nAvaliando no conjunto de teste:")
    evaluate_classifier(clf, X_test, y_test, scaler)
    
    # Salvar modelo treinado
    import joblib
    joblib.dump((clf, scaler), OUT / 'regime_classifier.joblib')
    print(f"\nModelo salvo em {OUT/'regime_classifier.joblib'}")

if __name__ == '__main__':
    main()