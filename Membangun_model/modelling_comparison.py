import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def configure_dagshub():
    dagshub_uri = os.getenv('MLFLOW_TRACKING_URI')
    dagshub_user = os.getenv('MLFLOW_TRACKING_USERNAME')
    dagshub_token = os.getenv('MLFLOW_TRACKING_PASSWORD')
    
    if dagshub_uri and dagshub_user and dagshub_token:
        mlflow.set_tracking_uri(dagshub_uri)
        return True
    else:
        return False

def load_preprocessed_data():
    X_train = pd.read_csv('diabetes_preprocessing/X_train.csv')
    X_val = pd.read_csv('diabetes_preprocessing/X_val.csv')
    y_train = pd.read_csv('diabetes_preprocessing/y_train.csv').values.ravel()
    y_val = pd.read_csv('diabetes_preprocessing/y_val.csv').values.ravel()
    return X_train, X_val, y_train, y_val

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

def plot_roc_curve(y_true, y_proba, model_name, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

def plot_feature_importance(model, feature_names, model_name, save_path):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_n = min(10, len(importances))
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    elif hasattr(model, 'coef_'):
        coef = np.abs(model.coef_[0])
        top_n = min(10, len(coef))
        indices = np.argsort(coef)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Coefficients - {model_name}')
        plt.bar(range(top_n), coef[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Absolute Coefficient')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    return None

def save_classification_report(y_true, y_pred, save_path):
    report = classification_report(y_true, y_pred, 
                                   target_names=['No Diabetes', 'Diabetes'])
    with open(save_path, 'w') as f:
        f.write(report)
    return save_path

def train_random_forest(X_train, y_train, X_val, y_val, feature_names):
    param_grid = {
        'n_estimators': [20, 30],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1, 
                               scoring='accuracy', verbose=0)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    
    with mlflow.start_run(run_name="RandomForest_Comparison"):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("smote_applied", True)
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred)
        val_rec = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_roc = roc_auc_score(y_val, y_val_proba)
        
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_precision", val_prec)
        mlflow.log_metric("val_recall", val_rec)
        mlflow.log_metric("val_f1_score", val_f1)
        mlflow.log_metric("val_roc_auc", val_roc)
        
        cm = confusion_matrix(y_val, y_val_pred)
        mlflow.log_metric("confusion_matrix_tn", int(cm[0, 0]))
        mlflow.log_metric("confusion_matrix_fp", int(cm[0, 1]))
        mlflow.log_metric("confusion_matrix_fn", int(cm[1, 0]))
        mlflow.log_metric("confusion_matrix_tp", int(cm[1, 1]))
        
        cm_path = plot_confusion_matrix(y_val, y_val_pred, "RandomForest", 
                                       "rf_confusion_matrix.png")
        mlflow.log_artifact(cm_path)
        
        roc_path = plot_roc_curve(y_val, y_val_proba, "RandomForest", 
                                 "rf_roc_curve.png")
        mlflow.log_artifact(roc_path)
        
        fi_path = plot_feature_importance(best_model, feature_names, "RandomForest",
                                         "rf_feature_importance.png")
        if fi_path:
            mlflow.log_artifact(fi_path)
        
        report_path = save_classification_report(y_val, y_val_pred,
                                                "rf_classification_report.txt")
        mlflow.log_artifact(report_path)
        
        mlflow.sklearn.log_model(best_model, "model")
        
        if os.path.exists(cm_path): os.remove(cm_path)
        if os.path.exists(roc_path): os.remove(roc_path)
        if fi_path and os.path.exists(fi_path): os.remove(fi_path)
        if os.path.exists(report_path): os.remove(report_path)
    
    print(f"Random Forest - Accuracy: {val_acc:.4f}, ROC-AUC: {val_roc:.4f}")
    return best_model, val_roc

def train_gradient_boosting(X_train, y_train, X_val, y_val, feature_names):
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5]
    }
    
    base_model = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1,
                               scoring='accuracy', verbose=0)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    
    with mlflow.start_run(run_name="GradientBoosting_Comparison"):
        mlflow.log_param("model_type", "GradientBoostingClassifier")
        mlflow.log_param("smote_applied", True)
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred)
        val_rec = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_roc = roc_auc_score(y_val, y_val_proba)
        
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_precision", val_prec)
        mlflow.log_metric("val_recall", val_rec)
        mlflow.log_metric("val_f1_score", val_f1)
        mlflow.log_metric("val_roc_auc", val_roc)
        
        cm = confusion_matrix(y_val, y_val_pred)
        mlflow.log_metric("confusion_matrix_tn", int(cm[0, 0]))
        mlflow.log_metric("confusion_matrix_fp", int(cm[0, 1]))
        mlflow.log_metric("confusion_matrix_fn", int(cm[1, 0]))
        mlflow.log_metric("confusion_matrix_tp", int(cm[1, 1]))
        
        cm_path = plot_confusion_matrix(y_val, y_val_pred, "GradientBoosting",
                                       "gb_confusion_matrix.png")
        mlflow.log_artifact(cm_path)
        
        roc_path = plot_roc_curve(y_val, y_val_proba, "GradientBoosting",
                                 "gb_roc_curve.png")
        mlflow.log_artifact(roc_path)
        
        fi_path = plot_feature_importance(best_model, feature_names, "GradientBoosting",
                                         "gb_feature_importance.png")
        if fi_path:
            mlflow.log_artifact(fi_path)
        
        report_path = save_classification_report(y_val, y_val_pred,
                                                "gb_classification_report.txt")
        mlflow.log_artifact(report_path)
        
        mlflow.sklearn.log_model(best_model, "model")
        
        if os.path.exists(cm_path): os.remove(cm_path)
        if os.path.exists(roc_path): os.remove(roc_path)
        if fi_path and os.path.exists(fi_path): os.remove(fi_path)
        if os.path.exists(report_path): os.remove(report_path)
    
    print(f"Gradient Boosting - Accuracy: {val_acc:.4f}, ROC-AUC: {val_roc:.4f}")
    return best_model, val_roc

def train_logistic_regression(X_train, y_train, X_val, y_val, feature_names):
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    base_model = LogisticRegression(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1,
                               scoring='accuracy', verbose=0)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    
    with mlflow.start_run(run_name="LogisticRegression_Comparison"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("smote_applied", True)
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred)
        val_rec = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_roc = roc_auc_score(y_val, y_val_proba)
        
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_precision", val_prec)
        mlflow.log_metric("val_recall", val_rec)
        mlflow.log_metric("val_f1_score", val_f1)
        mlflow.log_metric("val_roc_auc", val_roc)
        
        cm = confusion_matrix(y_val, y_val_pred)
        mlflow.log_metric("confusion_matrix_tn", int(cm[0, 0]))
        mlflow.log_metric("confusion_matrix_fp", int(cm[0, 1]))
        mlflow.log_metric("confusion_matrix_fn", int(cm[1, 0]))
        mlflow.log_metric("confusion_matrix_tp", int(cm[1, 1]))
        
        cm_path = plot_confusion_matrix(y_val, y_val_pred, "LogisticRegression",
                                       "lr_confusion_matrix.png")
        mlflow.log_artifact(cm_path)
        
        roc_path = plot_roc_curve(y_val, y_val_proba, "LogisticRegression",
                                 "lr_roc_curve.png")
        mlflow.log_artifact(roc_path)
        
        fi_path = plot_feature_importance(best_model, feature_names, "LogisticRegression",
                                         "lr_feature_coefficients.png")
        if fi_path:
            mlflow.log_artifact(fi_path)
        
        report_path = save_classification_report(y_val, y_val_pred,
                                                "lr_classification_report.txt")
        mlflow.log_artifact(report_path)
        
        mlflow.sklearn.log_model(best_model, "model")
        
        if os.path.exists(cm_path): os.remove(cm_path)
        if os.path.exists(roc_path): os.remove(roc_path)
        if fi_path and os.path.exists(fi_path): os.remove(fi_path)
        if os.path.exists(report_path): os.remove(report_path)
    
    print(f"Logistic Regression - Accuracy: {val_acc:.4f}, ROC-AUC: {val_roc:.4f}")
    return best_model, val_roc

def plot_model_comparison(results, save_path):
    models = list(results.keys())
    scores = list(results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Model')
    plt.ylabel('ROC-AUC Score')
    plt.title('Model Comparison - ROC-AUC Score')
    plt.ylim([0.7, 1.0])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

def main():
    is_dagshub = configure_dagshub()
    
    X_train, X_val, y_train, y_val = load_preprocessed_data()
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    print(f"After SMOTE: {X_train_resampled.shape}")
    
    feature_names = X_train.columns.tolist()
    
    experiment_name = "Diabetes_Model_Comparison_Advanced" if is_dagshub else "Diabetes_Model_Comparison"
    mlflow.set_experiment(experiment_name)
    
    results = {}
    
    _, roc_rf = train_random_forest(X_train_resampled, y_train_resampled, 
                                    X_val, y_val, feature_names)
    results['Random Forest'] = roc_rf
    
    _, roc_gb = train_gradient_boosting(X_train_resampled, y_train_resampled,
                                        X_val, y_val, feature_names)
    results['Gradient Boosting'] = roc_gb
    
    _, roc_lr = train_logistic_regression(X_train_resampled, y_train_resampled,
                                          X_val, y_val, feature_names)
    results['Logistic Regression'] = roc_lr
    
    comparison_path = plot_model_comparison(results, "model_comparison.png")
    
    with mlflow.start_run(run_name="Model_Comparison_Summary"):
        mlflow.log_param("experiment_type", "model_comparison")
        mlflow.log_param("models_compared", 3)
        mlflow.log_param("smote_applied", True)
        
        for model_name, score in results.items():
            mlflow.log_metric(f"{model_name.replace(' ', '_').lower()}_roc_auc", score)
        
        best_model = max(results, key=results.get)
        mlflow.log_param("best_model", best_model)
        mlflow.log_metric("best_roc_auc", results[best_model])
        
        mlflow.log_artifact(comparison_path)
        
        results_df = pd.DataFrame(list(results.items()), 
                                 columns=['Model', 'ROC-AUC'])
        results_df.to_csv('comparison_results.csv', index=False)
        mlflow.log_artifact('comparison_results.csv')
        
        if os.path.exists(comparison_path):
            os.remove(comparison_path)
        if os.path.exists('comparison_results.csv'):
            os.remove('comparison_results.csv')
    
    print("\nModel Comparison Results:")
    for model_name, score in results.items():
        print(f"{model_name}: {score:.4f}")
    print(f"\nBest Model: {best_model} (ROC-AUC: {results[best_model]:.4f})")
    
if __name__ == "__main__":
    main()

