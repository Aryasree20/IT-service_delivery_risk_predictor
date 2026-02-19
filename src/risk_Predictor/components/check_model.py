import joblib

# load model
model = joblib.load("artifacts/model_trainer/best_model.joblib")

# print model type
print("Model type:", type(model))

# print full model
print("\nModel details:")
print(model)

# if GridSearchCV
if hasattr(model, "best_estimator_"):
    print("\nBest estimator:")
    print(model.best_estimator_)

# if Pipeline
if hasattr(model, "named_steps"):
    print("\nPipeline steps:")
    print(model.named_steps)