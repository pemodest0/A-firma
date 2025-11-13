#!/usr/bin/env python3
"""Classificador de regimes do sistema Duffing usando múltiplas features."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def train_regime_classifier(X, y):
    """Treina classificador com features normalizadas e cross-validation."""
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Grid search para melhores parâmetros
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None]
    }
    
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        clf, param_grid,
        cv=5, scoring='f1_weighted',
        n_jobs=-1
    )
    
    # Treinar com os melhores parâmetros
    grid_search.fit(X_scaled, y)
    best_clf = grid_search.best_estimator_
    
    # Avaliar com cross-validation
    cv_scores = cross_val_score(best_clf, X_scaled, y, cv=5, scoring='f1_weighted')
    print("\nCross-validation scores:", cv_scores)
    print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print("\nBest parameters:", grid_search.best_params_)
    
    return best_clf, scaler

def classify_regime(features, clf, scaler):
    """Classifica regime usando modelo treinado."""
    X_scaled = scaler.transform(features.reshape(1, -1))
    return clf.predict(X_scaled)[0]

def evaluate_classifier(clf, X, y, scaler):
    """Avalia performance do classificador."""
    X_scaled = scaler.transform(X)
    y_pred = clf.predict(X_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    feature_importance = pd.DataFrame({
        'feature': ['entropy', 'alpha', 'n_clusters'],
        'importance': clf.feature_importances_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))