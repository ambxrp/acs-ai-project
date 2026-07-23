import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class ModelManager:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.feature_names = ['COUNCIL_DIST', 'Month', 'Week_of_Year', 'Temperature', 'Precipitation']
        
        # Dict of trained models
        self.models = {
            'stray': None,
            'total': None,
            'aggressive': None,
            'strain': None
        }
        
    def train(self, training_df, n_estimators=100, random_seed=42):
        """Train Random Forest models on the dataset"""
        X = training_df[self.feature_names]
        
        # Define target variables
        targets = {
            'stray': training_df['Weekly_Stray_Count'],
            'total': training_df['Weekly_Call_Count'],
            'aggressive': training_df['Weekly_Aggressive_Count'],
            'strain': training_df['Capacity_Strain_Score']
        }
        
        print(f"Training models on {len(training_df)} records...")
        for key in self.models.keys():
            print(f"Training Random Forest Regressor for target: '{key}'...")
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_seed,
                n_jobs=-1
            )
            rf.fit(X, targets[key])
            self.models[key] = rf
            print(f"✅ Target '{key}' trained successfully. R^2 score: {rf.score(X, targets[key]):.4f}")
            
    def save_models(self):
        """Save trained models to disk"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        for key, model in self.models.items():
            if model is None:
                print(f"⚠️ Model '{key}' is not trained. Skipping save.")
                continue
            path = os.path.join(self.model_dir, f"{key}_rf_model.pkl")
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved model '{key}' to {path}")

    def load_models(self):
        """Load models from disk"""
        success = True
        for key in self.models.keys():
            path = os.path.join(self.model_dir, f"{key}_rf_model.pkl")
            if not os.path.exists(path):
                print(f"❌ Model file not found: {path}")
                success = False
                continue
            with open(path, 'rb') as f:
                self.models[key] = pickle.load(f)
            print(f"Loaded model '{key}' from {path}")
        return success

    def predict(self, district_id, month, week_of_year, temperature, precipitation):
        """Run predictions for district and weather scenario"""
        # Check model initialization
        for key, model in self.models.items():
            if model is None:
                raise ValueError(f"Model '{key}' is not trained or loaded. Call train() or load_models() first.")
                
        # Create input dataframe
        input_data = pd.DataFrame(
            [[district_id, month, week_of_year, temperature, precipitation]],
            columns=self.feature_names
        )
        
        predictions = {}
        for key, model in self.models.items():
            pred_val = model.predict(input_data)[0]
            # Clean predictions and clip strain score between 0 and 100
            if key == 'strain':
                predictions[key] = round(float(np.clip(pred_val, 0.0, 100.0)), 2)
            else:
                predictions[key] = int(round(float(max(0.0, pred_val))))
                
        return {
            'predicted_stray_count': predictions['stray'],
            'predicted_total_count': predictions['total'],
            'predicted_aggressive_count': predictions['aggressive'],
            'predicted_capacity_strain_score': predictions['strain']
        }
