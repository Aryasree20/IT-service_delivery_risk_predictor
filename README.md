# End_to-End_IT_Services_Delivery_RiskPredictor_using_MLFlow_DVC_CICD_Pipeline


## 🎯 Objective

The goal of this project is to build and deploy a Delivery Risk Prediction System that classifies IT projects into High, Medium, or Low risk.

The workflow integrates:

  - MLflow for experiment tracking

  - DVC for dataset & pipeline versioning

  - CI/CD with Docker & GitHub Actions for deployment to AWS


## 📌 Project Workflow Diagram  


## ⚙️ Project Architecture

  ### 1️⃣ Setup & Configuration

        - config.yaml → project paths and global settings

        - params.yaml → model hyperparameters

        - Entity classes + Config Manager for structured configs

        - Pipeline stages: ingestion, preprocessing, training, evaluation

        - Defined in dvc.yaml and triggered via main.py


  ### 2️⃣ Model Development

        The Delivery Risk Prediction System follows a modular pipeline, where each stage ensures data quality, balanced learning, and robust evaluation before deployment. 

        🗂️ 2.1 Data Ingestion
          - Loads raw project data  

        ⚙️ 2.2 Data Preprocessing
          - check missing values, inconsistent formats, and ensures clean schema. 
          - Encoding & Scaling
          - Drop unwanted columns
          - Imbalanced Data Handling using SMOTE

        🤖 2.3 Model Training
         - Trains multiple machine learning models for classification:
           - Logistic Regression
           - Decision Tree
           - Random Forest
           - XGBoost
           - daBoost
           - KNN
         - Perform Hyperparameter Tuning to optimize parameters for each model.
         - Model Selection: The best-performing model is chosen based on F1-Score and ROC AUC.

        📊 2.4 Model Evaluation
         - Evaluates trained models on the unseen test dataset.
         - Metrics considered:
           - Precision → To minimize false positives in risk classification.
           - Recall → To ensure high detection of risky projects.
           - F1-Score → Balances precision and recall.
           - ROC AUC → Measures overall model discriminative ability.  
         - Logs metrics and artifacts to MLflow for experiment tracking.


  ### 3️⃣ MLflow Experiment Tracking

       - Logs: params, metrics, models
       - DagsHub configured as remote MLflow server
       - Compare experiments via mlflow ui 


  ### 4️⃣ Data Version Control with DVC

       - Versioned datasets & pipeline artifacts
       - Commands:
         - dvc repro → re-run pipeline
         - dvc dag → visualize stages
       - Ensures reproducibility & collaboration  


  ### 5️⃣ CI/CD Deployment (Docker + GitHub Actions + AWS)

       - Designed an automated CI/CD workflow with GitHub Actions and AWS deployment
       - Configured EC2 as a self-hosted runner to execute pipeline jobs
       - Workflow tasks:
         - Build Docker image for the delivery risk prediction service
         - Upload image to Amazon ECR
         - Deploy container on EC2 by pulling from ECR for real-time inferenc
       - IAM roles applied:
         - AmazonEC2FullAccess
         - AmazonEC2ContainerRegistryFullAccess
         - Secured AWS access keys and settings through GitHub Secrets


  ### 6️⃣ Web Application (Flask)

       - UI for project risk prediction
       - Input features:
          - planned_duration_days
          - actual_duration_days
          - team_size
          - num_bugs
          - num_change_requests
          - budget_overrun_pct
       - Predicts: High, Medium, Low Risk
       - HTML + CSS frontend with Flask backend
       - Deployable locally (app.py) or on AWS


  ### ✅ Outcome

      ✔️ Automated risk prediction pipeline
      ✔️ Reproducible experiments with DVC
      ✔️ MLflow-based tracking & versioning
      ✔️ CI/CD with AWS + GitHub Actions
      ✔️ Web app for real-time predictions



  ### 🛠️ Tech Stack

       - Python | scikit-learn | Flask
       - MLflow | DVC | Docker
       - AWS EC2 & ECR | GitHub Actions
       - DagsHub (remote MLflow tracking)    





