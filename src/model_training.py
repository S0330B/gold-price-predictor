from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn import metrics
import joblib

def train_model(X, y, save_path='models/GoldPricePredictor'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    r2 = metrics.r2_score(y_test, predictions)
    print(f"R² Score: {r2:.4f}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Model saved at {save_path}")
    
    return model, X_test, y_test, predictions