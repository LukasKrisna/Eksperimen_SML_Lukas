import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
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
    print(f"Data loaded - Train: {X_train.shape}, Val: {X_val.shape}")
    return X_train, X_val, y_train, y_val

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied: {len(y_train)} -> {len(y_resampled)} samples")
    return X_resampled, y_resampled

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

def plot_roc_curve(y_true, y_proba, save_path='roc_curve.png'):
    from sklearn.metrics import auc
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
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

def save_classification_report(y_true, y_pred, save_path='classification_report.txt'):
    report = classification_report(y_true, y_pred, 
                                   target_names=['No Diabetes', 'Diabetes'])
    with open(save_path, 'w') as f:
        f.write('Classification Report\n')
        f.write('='*60 + '\n\n')
        f.write(report)
    return save_path

def plot_feature_importance(model, n_features, save_path='feature_importance.png'):
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    importances = model.feature_importances_
    top_n = min(10, n_features)
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Top {top_n} Feature Importances')
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

def perform_hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [10, 20, 30],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8]
    }
    
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best CV Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def train_with_manual_logging(X_train, y_train, X_val, y_val, 
                               use_smote=True, is_dagshub=False):
    if use_smote:
        X_train_final, y_train_final = apply_smote(X_train, y_train)
    else:
        X_train_final, y_train_final = X_train, y_train
    
    best_model, best_params = perform_hyperparameter_tuning(X_train_final, y_train_final)
    
    experiment_name = "Diabetes_Classification_Advanced_DagsHub" if is_dagshub else "Diabetes_Classification_Skilled"
    mlflow.set_experiment(experiment_name)
    
    run_name = "RandomForest_Tuned_SMOTE_Manual" if use_smote else "RandomForest_Tuned_Manual"
    
    with mlflow.start_run(run_name=run_name):
        
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("smote_applied", use_smote)
        mlflow.log_param("random_state", 42)
        
        for param, value in best_params.items():
            mlflow.log_param(param, value)
        
        best_model.fit(X_train_final, y_train_final)
        
        y_train_pred = best_model.predict(X_train_final)
        y_val_pred = best_model.predict(X_val)
        y_val_proba = best_model.predict_proba(X_val)[:, 1]
        
        train_accuracy = accuracy_score(y_train_final, y_train_pred)
        mlflow.log_metric("train_accuracy", train_accuracy)
        
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_precision", val_precision)
        mlflow.log_metric("val_recall", val_recall)
        mlflow.log_metric("val_f1_score", val_f1)
        
        val_roc_auc = roc_auc_score(y_val, y_val_proba)
        mlflow.log_metric("val_roc_auc", val_roc_auc)
        
        
        cm_path = plot_confusion_matrix(y_val, y_val_pred, 
                                        save_path='confusion_matrix.png')
        mlflow.log_artifact(cm_path)
        
        roc_path = plot_roc_curve(y_val, y_val_proba, 
                                  save_path='roc_curve.png')
        mlflow.log_artifact(roc_path)
        
        report_path = save_classification_report(y_val, y_val_pred,
                                                save_path='classification_report.txt')
        mlflow.log_artifact(report_path)
        
        fi_path = plot_feature_importance(best_model, X_train.shape[1],
                                         save_path='feature_importance.png')
        mlflow.log_artifact(fi_path)
        
        cm = confusion_matrix(y_val, y_val_pred)
        mlflow.log_metric("confusion_matrix_tn", int(cm[0, 0]))
        mlflow.log_metric("confusion_matrix_fp", int(cm[0, 1]))
        mlflow.log_metric("confusion_matrix_fn", int(cm[1, 0]))
        mlflow.log_metric("confusion_matrix_tp", int(cm[1, 1]))
        
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importances.to_csv('feature_importances.csv', index=False)
        mlflow.log_artifact('feature_importances.csv')
        
        mlflow.log_param("train_samples", len(X_train_final))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("n_features", X_train.shape[1])
        
        print(f"\nResults - Acc: {val_accuracy:.4f}, Prec: {val_precision:.4f}, " +
              f"Rec: {val_recall:.4f}, F1: {val_f1:.4f}, ROC-AUC: {val_roc_auc:.4f}")
        
        run = mlflow.active_run()
        print(f"Run ID: {run.info.run_id}")
        
        for temp_file in ['confusion_matrix.png', 'roc_curve.png', 
                         'classification_report.txt', 'feature_importance.png',
                         'feature_importances.csv']:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    return best_model

def main():
    is_dagshub = configure_dagshub()
    X_train, X_val, y_train, y_val = load_preprocessed_data()
    
    model = train_with_manual_logging(
        X_train, y_train, X_val, y_val,
        use_smote=True,
        is_dagshub=is_dagshub
    )

if __name__ == "__main__":
    main()

