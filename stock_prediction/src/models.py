
from typing import Dict, Any
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
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
    
    # CMS-Optimization: Use LinearSVC for large datasets (>10k samples)
    if len(X_train) > 10000:
        logger.info(f"Dataset size {len(X_train)} > 10000. Using LinearSVC for performance.")
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        
        # LinearSVC doesn't support predict_proba, so we wrap it in CalibratedClassifierCV
        linear_svc = LinearSVC(
            dual=False, # Prefer dual=False when n_samples > n_features
            random_state=settings.RANDOM_STATE,
            class_weight='balanced',
            max_iter=1000
        )
        svm_model = CalibratedClassifierCV(linear_svc)
        svm_model.fit(X_train, y_train)
        models['SVM'] = svm_model
        logger.info("LinearSVM (Calibrated) trained successfully.")
        
    else:
        # Use Standard RBF SVM for smaller datasets
        svm_model = SVC(
            kernel='rbf', 
            random_state=settings.RANDOM_STATE,
            probability=True, # Enable probability for ROC curve if needed
            class_weight='balanced' # Handle Class Imbalance
        )
        svm_model.fit(X_train, y_train)
        models['SVM'] = svm_model
        logger.info("Standard RBF SVM trained.")
    
    # --- Phase 16: Deep Learning & Boosting Expansion ---
    
    # --- Phase 16: Deep Learning & Boosting Expansion ---
    
    logger.info("Training Gradient Boosting Classifier...")
    # Optimized GBM: More estimators, slower learning rate, subsampling
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=settings.RANDOM_STATE
    )
    gb_model.fit(X_train, y_train)
    models['Gradient Boosting'] = gb_model
    logger.info("Gradient Boosting trained (Optimized).")
    
    logger.info("Training MLP (Neural Network)...")
    # Optimized MLP: Scaling + Regularization
    # MLP requires scaling!
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    
    mlp_model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01, # Stronger L2 Regularization
            batch_size='auto',
            learning_rate_init=0.001,
            max_iter=1000, # More iterations for convergence
            early_stopping=True,
            validation_fraction=0.1,
            random_state=settings.RANDOM_STATE
        )
    )
    mlp_model.fit(X_train, y_train)
    models['MLP'] = mlp_model
    logger.info("Multi-Layer Perceptron trained (Scaled + Optimized).")

    # --- Senior Trader Ensemble ---
    logger.info("Training Ensemble Voting Classifier...")
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('svm', svm_model),
            ('gb', gb_model),
            ('mlp', mlp_model)
        ],
        voting='soft' # Average probabilities
    )
    ensemble.fit(X_train, y_train)
    models['Ensemble'] = ensemble
    logger.info("Ensemble Voting Classifier trained (5 Models).")
    
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
