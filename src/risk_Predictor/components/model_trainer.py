import warnings
warnings.filterwarnings( "ignore",category=UserWarning,message="pkg_resources is deprecated as an API")
import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_curve,
    auc
)
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from collections import Counter
from risk_predictor import logger
from risk_predictor.entity.config_entity import  ModelTrainerConfig
from pathlib import Path



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        os.makedirs(self.config.root_dir, exist_ok=True)
        os.makedirs(self.config.reports_dir, exist_ok=True)
        os.makedirs(self.config.plots_dir, exist_ok=True)
        


        self.models = {
            "Logistic Regression": {
                "model": OneVsRestClassifier(LogisticRegression(max_iter=2000,class_weight='balanced')),
                "params": {
                    "estimator__C": [0.01, 0.1, 1, 10],
                    "estimator__solver": ['lbfgs','saga'],
                    "estimator__penalty": ['l2']
                }
            },     
            "Decision Tree": {
                "model": DecisionTreeClassifier(class_weight='balanced'),
                "params": {
                    'max_depth': [3, 5, 10],
                    'criterion': ['gini', 'entropy']
                }
            },
            "Random Forest": {
                "model": RandomForestClassifier(class_weight='balanced'),
                "params": {
                    'n_estimators': [100, 200],
                    'max_depth': [5, 10, None]
                }
            },
            "XGBoost": {
                "model": XGBClassifier(eval_metric='logloss',objective="multi:softmax", num_class=3),
                "params": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5, 6]
                }
            },
            "AdaBoost": {
                "model": AdaBoostClassifier(),
                "params": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.5, 1.0]
                }
            },
            "KNN": {
                "model": KNeighborsClassifier(),
                "params": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                }
            }
        }


    def _save_report_json(self, name, report):
        report_path = Path(self.config.reports_dir) / f"{name}_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        logger.info(f"📄 Saved report JSON: {report_path}")

    def _save_report_csv(self, name, report):
        df = pd.DataFrame(report).transpose()
        csv_path = Path(self.config.reports_dir) / f"{name}_report.csv"
        df.to_csv(csv_path, index=True)
        logger.info(f"📊 Saved report CSV: {csv_path}")

    def _save_confusion_matrix(self, name, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        cm_path = Path(self.config.plots_dir) / f"{name}_cm.png"
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"📉 Saved confusion matrix plot: {cm_path}")

    def _save_roc_curve(self, name, y_true, y_pred_proba, classes):
        y_true_bin = label_binarize(y_true, classes=classes)
        if y_true_bin.shape[1] == 1:
            return  

        plt.figure()
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"ROC Curve - {name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        roc_path = Path(self.config.plots_dir) / f"{name}_roc.png"
        plt.savefig(roc_path)
        plt.close()
        logger.info(f"📈 Saved ROC curve plot: {roc_path}")

    def train_and_evaluate(self, x_train, y_train, x_test, y_test):
        results = {}
        best_classifier, best_name, best_f1,best_recall,best_precision = None, None, 0, 0, 0

        classes = np.unique(y_train)

        for name, mp in self.models.items():
            logger.info(f"🔹 Training {name}...")

            clf = GridSearchCV(
                mp["model"], mp["params"], 
                cv=5, 
                n_jobs=-1, 
                verbose=1
            )
            clf.fit(x_train, y_train)

            best_model = clone(mp["model"]).set_params(**clf.best_params_)
            best_model.fit(x_train, y_train)

            y_pred = best_model.predict(x_test)
            report = classification_report(y_test, y_pred, output_dict=True)

            # Add best params to the report
            report["best_params"] = clf.best_params_
            

            # Save reports
            self._save_report_json(name, report)
            self._save_report_csv(name, report)

            # Save confusion matrix
            self._save_confusion_matrix(name, y_test, y_pred)

            # ROC curve (if proba available)
            if hasattr(best_model, "predict_proba"):
                y_pred_proba = best_model.predict_proba(x_test)
                self._save_roc_curve(name, y_test, y_pred_proba, classes)

            # Store metrics
            results[name] = {
                "accuracy" :accuracy_score(y_test, y_pred),
                "precision": report['weighted avg']['precision'],
                "recall": report['weighted avg']['recall'],
                "f1": report['weighted avg']['f1-score'],
                "model": best_model
            }

            logger.info(f"{name} → Accuracy={results[name]['accuracy']}, Precision={results[name]['precision']:.2f}, Recall={results[name]['recall']:.2f}, F1={results[name]['f1']:.2f}")

            if results[name]["f1"] !=1 :
                if results[name]["f1"] > best_f1:
                        best_f1 = results[name]["f1"]
                        best_precision = results[name]['precision']
                        best_recall = results[name]['recall']
                        best_classifier = best_model
                        best_name = name

        if best_classifier:
            joblib.dump(best_classifier, self.config.best_model_path)
             
             # Save model into best_model_path
            logger.info(f"Best model: {best_name} (F1={best_f1:.2f}, Precision={best_precision:.2f}, Recall={best_recall:.2f}) saved at {self.config.best_model_path}")
        else:
            logger.warning(" No suitable model found!")

        return best_classifier, best_name