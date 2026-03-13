import pickle
import pandas as pd
import numpy as np

# Load the trained XGBoost model
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get feature importance scores
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

print("="*70)
print("XGBOOST FEATURE IMPORTANCE (WEIGHTS)")
print("="*70)

print("\nModel Type:", type(model))

# Try to get feature importance using sklearn interface
try:
    # Most XGBoost classifiers have feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        print("\n1. FEATURE IMPORTANCE (Built-in sklearn interface):")
        print("-" * 70)
        
        # Create dataframe for better display
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Percentage': (importances / importances.sum() * 100)
        }).sort_values('Importance', ascending=False)
        
        print("\nRanked by Importance:")
        for idx, row in importance_df.iterrows():
            bar = '█' * int(row['Percentage'] / 2)
            print(f"{row['Feature']:20s}: {row['Importance']:.4f} ({row['Percentage']:6.2f}%) {bar}")
        
        print("\n" + "="*70)
        print("TOP 3 MOST IMPORTANT FEATURES:")
        print("="*70)
        for i, (idx, row) in enumerate(importance_df.head(3).iterrows(), 1):
            print(f"{i}. {row['Feature']:20s}: {row['Percentage']:6.2f}% importance")
            
    else:
        print("Model doesn't have feature_importances_ attribute")
        
except Exception as e:
    print(f"Error accessing feature importance: {e}")

# Try alternative method using booster
try:
    print("\n\n2. DETAILED XGBOOST METRICS:")
    print("="*70)
    
    # Get booster
    booster = model.get_booster()
    
    # Get importance scores
    importance_types = ['weight', 'gain', 'cover']
    
    for imp_type in importance_types:
        try:
            scores = booster.get_score(importance_type=imp_type)
            
            if scores:
                print(f"\n{imp_type.upper()} - ", end="")
                if imp_type == 'weight':
                    print("Number of times feature is used in splits")
                elif imp_type == 'gain':
                    print("Average gain/improvement per split")
                elif imp_type == 'cover':
                    print("Average number of samples affected")
                print("-" * 70)
                
                # Map f0, f1, etc to feature names
                feature_scores = {}
                for key, value in scores.items():
                    # Extract feature index from fX format
                    if key.startswith('f'):
                        try:
                            idx = int(key[1:])
                            if idx < len(feature_names):
                                feature_scores[feature_names[idx]] = value
                        except:
                            pass
                
                # Sort and display
                if feature_scores:
                    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                    total = sum(feature_scores.values())
                    
                    for feature, score in sorted_features:
                        percentage = (score / total * 100) if total > 0 else 0
                        print(f"{feature:20s}: {score:10.2f} ({percentage:6.2f}%)")
                else:
                    print("No feature scores found with proper mapping")
            else:
                print(f"\n{imp_type.upper()}: No scores available")
        except Exception as e:
            print(f"\n{imp_type.upper()}: Error - {e}")
            
except Exception as e:
    print(f"Error accessing booster: {e}")

print("\n")
