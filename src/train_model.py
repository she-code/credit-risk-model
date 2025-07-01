import mlflow
import mlflow.sklearn
import pytest
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def load_and_prepare_data():
    """Load and prepare the dataset for modeling"""
    # Load processed data with target variable
    df = pd.read_csv('../data/processed/data_processed.csv')
    
    # Select only the engineered numeric features (from Task 3-4)
    feature_columns = [
        'num__Amount',
        'num__Value', 
        'num__CustomerId_TransactionCount',
        'num__CustomerId_TotalAmount',
        'num__CustomerId_AvgAmount',
        'num__CustomerId_AmountStd',
        'num__CustomerId_AmountSkew',
        'num__ProductCategory_TransactionCount',
        'num__ProductCategory_TotalAmount',
        'num__ProductCategory_AvgAmount',
        'num__ProductCategory_AmountStd',
        'num__ProductCategory_AmountSkew',
        'num__TransactionHour',
        'num__TransactionDay',
        'num__TransactionDayOfWeek',
        'num__TransactionMonth',
        'num__TransactionYear',
        'num__IsWeekend'
    ]
    
    # Verify all required columns exist
    available_features = [col for col in feature_columns if col in df.columns]
    if len(available_features) != len(feature_columns):
        missing = set(feature_columns) - set(available_features)
        raise ValueError(f"Missing expected features: {missing}")
    
    # Select features and target
    X = df[available_features]
    y = df['is_high_risk']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, params=None, search_method='grid'):
    """Train and evaluate a single model"""
    with mlflow.start_run(run_name=model_name, nested=True):
        # Hyperparameter tuning
        if params:
            if search_method == 'grid':
                clf = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1)
            else:
                clf = RandomizedSearchCV(model, params, cv=5, n_iter=10, 
                                      scoring='roc_auc', random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            best_model = clf.best_estimator_
        else:
            best_model = model.fit(X_train, y_train)
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Log parameters and metrics
        if params and search_method == 'grid':
            mlflow.log_params(clf.best_params_)
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(best_model, model_name)
        
        # Print summary
        print(f"\n{model_name} Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return best_model, metrics

def main():
    # Initialize MLflow
    mlflow.set_tracking_uri("file:./mlruns")  # Local tracking
    mlflow.set_experiment("Credit_Risk_Modeling")
    
    # Load and prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_prepare_data()
    
    # Define models and parameters
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        }
    }
    
    # Train and evaluate models
    best_metrics = {}
    best_model = None
    best_model_name = ''
    
    with mlflow.start_run(run_name="Parent_Run"):
        # Log data preprocessing details
        mlflow.log_param("n_features", X_train_scaled.shape[1])
        mlflow.log_param("scaler", "StandardScaler")
        
        for model_name, config in models.items():
            trained_model, metrics = train_and_evaluate_model(
                config['model'], 
                model_name, 
                X_train_scaled, y_train, 
                X_test_scaled, y_test,
                config['params']
            )
            
            # Track best model
            if not best_model or metrics['roc_auc'] > best_metrics['roc_auc']:
                best_model = trained_model
                best_metrics = metrics
                best_model_name = model_name
        
        # Register best model
        if best_model:
            mlflow.sklearn.log_model(best_model, "best_model")
            
            # Register in Model Registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
            registered_model = mlflow.register_model(model_uri, "CreditRiskModel")
            print(f"\nBest model registered: {best_model_name}")
            print(f"Model URI: {model_uri}")
            
            # Log best model info
            mlflow.log_param("best_model", best_model_name)
            mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})

if __name__ == "__main__":
    main()