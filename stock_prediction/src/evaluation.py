
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from stock_prediction.src import visualization
from stock_prediction.src.utils import setup_logger

logger = setup_logger("evaluation")

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models and print metrics.
    
    Args:
        models: Dictionary of trained models.
        X_test: Test features.
        y_test: Test targets.
    """
    results = {}
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        predictions = model.predict(X_test)
        
        acc = accuracy_score(y_test, predictions)
        report_str = classification_report(y_test, predictions)
        report_dict = classification_report(y_test, predictions, output_dict=True)
        cm = confusion_matrix(y_test, predictions)
        
        logger.info(f"{name} Accuracy: {acc}")
        logger.info(f"\n{report_str}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Plot Confusion Matrix
        visualization.plot_confusion_matrix(cm, name)
        
        results[name] = {
            "accuracy": acc,
            "report_str": report_str,
            "report_dict": report_dict,
            "confusion_matrix": cm,
            "predictions": predictions
        }
    
    # Comparative Plots
    visualization.plot_model_accuracy_comparison(results)
    visualization.plot_model_metrics_comparison(results)
    visualization.plot_roc_curve_comparison(models, X_test, y_test)
        
    return results
