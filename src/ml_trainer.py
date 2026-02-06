"""
ML Trainer Module for Phase 3: Train AI Brain

This script:
1. Loads labeled dataset from Phase 2
2. Trains XGBoost/RandomForest model
3. Evaluates model performance
4. Exports trained model for real-time prediction
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

# ML Libraries
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[!] scikit-learn not installed. Run: pip install scikit-learn")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[!] xgboost not installed. Run: pip install xgboost")

# ============================================
# CONFIGURATION
# ============================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Features to use for training
FEATURE_COLUMNS = [
    'rsi',
    'ema50_slope',
    'ema200_slope',
    'price_vs_ema50',
    'price_vs_ema200',
    'vol_ratio',
    'bb_width',
    'has_pattern',
    'pattern_matches_trend',
]

# Target column
TARGET_COLUMN = 'outcome_binary'


class MLTrainer:
    """Trains and evaluates ML models for trade prediction."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_columns = FEATURE_COLUMNS
        print("[*] ML Trainer initialized")
    
    def load_dataset(self, filename='elliott_signals.csv'):
        """Load labeled dataset from CSV."""
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"[!] Dataset not found: {filepath}")
            print("    Run data_miner first to generate the dataset.")
            return None
        
        df = pd.read_csv(filepath)
        print(f"[+] Loaded {len(df)} samples from {filename}")
        
        # Show class balance
        wins = (df['outcome'] == 'WIN').sum()
        losses = (df['outcome'] == 'LOSS').sum()
        print(f"    Class Balance: {wins} Wins ({wins/len(df)*100:.1f}%) / {losses} Losses ({losses/len(df)*100:.1f}%)")
        
        return df
    
    def prepare_features(self, df):
        """Prepare feature matrix X and target vector y."""
        # Select features
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Target
        y = df[TARGET_COLUMN].values
        
        print(f"[*] Feature Matrix: {X.shape}")
        print(f"    Features: {list(X.columns)}")
        
        return X, y
    
    def train_model(self, X, y, model_type='xgboost'):
        """Train ML model."""
        print(f"\n[*] Training {model_type.upper()} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"    Train: {len(X_train)} | Test: {len(X_test)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select model
        if model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:
            # Fallback to Gradient Boosting
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        print("\n=== Model Performance ===")
        print(f"    Accuracy:  {accuracy_score(y_test, y_pred)*100:.1f}%")
        print(f"    Precision: {precision_score(y_test, y_pred, zero_division=0)*100:.1f}%")
        print(f"    Recall:    {recall_score(y_test, y_pred, zero_division=0)*100:.1f}%")
        print(f"    F1 Score:  {f1_score(y_test, y_pred, zero_division=0)*100:.1f}%")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n    Confusion Matrix:")
        print(f"    TN: {cm[0,0]} | FP: {cm[0,1]}")
        print(f"    FN: {cm[1,0]} | TP: {cm[1,1]}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5)
        print(f"\n    Cross-Val Accuracy: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")
        
        # Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            print("\n=== Feature Importance ===")
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for _, row in importance.iterrows():
                bar = '█' * int(row['importance'] * 50)
                print(f"    {row['feature']:20} {row['importance']:.3f} {bar}")
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
        }
    
    def save_model(self, filename='elliott_model.pkl'):
        """Save trained model and scaler to disk."""
        if self.model is None:
            print("[!] No model to save")
            return
        
        model_path = os.path.join(MODEL_DIR, filename)
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\n[+] Model saved to {model_path}")
        print(f"[+] Scaler saved to {scaler_path}")
        
        # Save feature list
        features_path = os.path.join(MODEL_DIR, 'features.txt')
        with open(features_path, 'w') as f:
            f.write('\n'.join(self.feature_columns))
        print(f"[+] Features saved to {features_path}")
    
    def predict_proba(self, features_dict):
        """
        Predict win probability for a single signal.
        
        Args:
            features_dict: Dictionary with feature values
        
        Returns:
            float: Probability of WIN (0.0 to 1.0)
        """
        if self.model is None:
            return 0.5  # Default probability if no model
        
        # Create feature vector
        X = np.array([[features_dict.get(f, 0) for f in self.feature_columns]])
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prob = self.model.predict_proba(X_scaled)[0, 1]
        
        return prob


def load_trained_model():
    """Load a previously trained model."""
    model_path = os.path.join(MODEL_DIR, 'elliott_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    features_path = os.path.join(MODEL_DIR, 'features.txt')
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        print("[!] Trained model not found. Run training first.")
        return None
    
    trainer = MLTrainer()
    
    with open(model_path, 'rb') as f:
        trainer.model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        trainer.scaler = pickle.load(f)
    
    with open(features_path, 'r') as f:
        trainer.feature_columns = f.read().strip().split('\n')
    
    print("[+] Loaded trained model from disk")
    return trainer


def run_training():
    """Main function to run the training process."""
    if not SKLEARN_AVAILABLE:
        print("[!] scikit-learn is required. Install with: pip install scikit-learn")
        return
    
    print("\n" + "="*60)
    print("  PHASE 3: ML BRAIN TRAINING")
    print("="*60)
    
    trainer = MLTrainer()
    
    # Step 1: Load dataset
    df = trainer.load_dataset()
    if df is None or len(df) < 10:
        print("[!] Not enough data for training. Need at least 10 samples.")
        return
    
    # Step 2: Prepare features
    X, y = trainer.prepare_features(df)
    
    # Step 3: Train model
    model_type = 'xgboost' if XGBOOST_AVAILABLE else 'gradient_boosting'
    metrics = trainer.train_model(X, y, model_type=model_type)
    
    # Step 4: Save model
    trainer.save_model()
    
    print("\n[*] Training complete!")
    return trainer


if __name__ == '__main__':
    run_training()
