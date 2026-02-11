
from typing import Dict, Any
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from stock_prediction.config import settings
from stock_prediction.src.utils import setup_logger

logger = setup_logger("models")

def train_models(X_train, y_train) -> Dict[str, Any]:
    """
    Train Random Forest, XGBoost, and SVM models.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        
    Returns:
        Dictionary of trained models.
    """
    models = {}
    
    logger.info("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=settings.RANDOM_STATE,
        class_weight='balanced' # Handle Class Imbalance
    )
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    logger.info("Random Forest trained.")
    
    logger.info("Training XGBoost Classifier...")
    # Calculate scale_pos_weight for XGBoost
    # count(negative) / count(positive)
    num_positive = y_train.sum()
    num_negative = len(y_train) - num_positive
    scale_pos_weight = num_negative / num_positive if num_positive > 0 else 1.0
    
    xgb_model = XGBClassifier(
        n_estimators=100, 
        random_state=settings.RANDOM_STATE, 
        eval_metric='mlogloss',
        scale_pos_weight=scale_pos_weight # Handle Class Imbalance
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    logger.info(f"XGBoost trained with scale_pos_weight: {scale_pos_weight:.2f}")
    
    logger.info("Training Support Vector Machine...")
    svm_model = SVC(
        kernel='rbf', 
        random_state=settings.RANDOM_STATE,
        probability=True, # Enable probability for ROC curve if needed
        class_weight='balanced' # Handle Class Imbalance
    )
    svm_model.fit(X_train, y_train)
    models['SVM'] = svm_model
    models['SVM'] = svm_model
    logger.info("SVM trained.")
    
    # --- Senior Trader Ensemble ---
    logger.info("Training Ensemble Voting Classifier...")
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('svm', svm_model)
        ],
        voting='soft' # Average probabilities
    )
    ensemble.fit(X_train, y_train)
    models['Ensemble'] = ensemble
    logger.info("Ensemble Voting Classifier trained.")
    
    return models

def save_models(models: Dict[str, Any], output_dir=None):
    """
    Save trained models to disk.
    """
    if output_dir is None:
        # Default to a 'models' dir in Data dir or similar? 
        # Implementation plan didn't specify model save path structure details
        # Let's put them in a 'models' folder under 'data' or similar for now
        output_dir = settings.DATA_DIR / "models"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, model in models.items():
        filename = output_dir / f"{name.replace(' ', '_').lower()}_model.joblib"
        logger.info(f"Saving {name} to {filename}")
        joblib.dump(model, filename)
