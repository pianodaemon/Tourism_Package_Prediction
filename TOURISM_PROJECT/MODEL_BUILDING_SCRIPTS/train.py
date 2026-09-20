import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
import os, sys

_ANS_TO_THE_ULTIMATE_QUESTION_OF_LIFE = 42

# 1. Use local SQLite for fast, offline logging (No ngrok network lockup)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("tourism_package_prediction")

# 2. Authenticate Hugging Face
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set.")

api = HfApi(token=hf_token)

Xtrain_path = "hf://datasets/j4nusx/tourism/Xtrain.csv"
Xtest_path = "hf://datasets/j4nusx/tourism/Xtest.csv"
ytrain_path = "hf://datasets/j4nusx/tourism/ytrain.csv"
ytest_path = "hf://datasets/j4nusx/tourism/ytest.csv"

# Load datasets
Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest  = pd.read_csv(ytest_path).squeeze()

numeric_features = [
    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch",
]

categorical_features = []

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    remainder="passthrough"
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=_ANS_TO_THE_ULTIMATE_QUESTION_OF_LIFE)

param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 4, 8],
    'xgbclassifier__colsample_bytree': [0.4, 0.6, 0.8],
    'xgbclassifier__colsample_bylevel': [0.4, 0.6, 0.8],
    'xgbclassifier__learning_rate': [0.01, 0.1, 0.05],
    'xgbclassifier__reg_lambda': [0.4, 0.6, 0.75],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# 3. Train and log run
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    classification_threshold = 0.40

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save artifact locally
    output_dir = "TOURISM_PROJECT/MODEL_BUILDING_SCRIPTS"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "xgb_tourism_model.joblib")
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved locally at: {model_path}")

    # Push artifact to Hugging Face
    repo_id = "j4nusx/tourism-model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' found.")
    except RepositoryNotFoundError:
        print(f"Creating model repo '{repo_id}'...")
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="xgb_tourism_model.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Model successfully uploaded to Hugging Face.")

print("Execution finished successfully. Exiting cleanly.")
