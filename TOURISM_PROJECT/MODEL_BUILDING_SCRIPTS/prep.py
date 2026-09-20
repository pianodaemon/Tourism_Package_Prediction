
import pandas as pd
import sklearn

import os

from sklearn.model_selection import train_test_split
from huggingface_hub import login, HfApi

# Answer To The Ultimate Question - The Hitchhiker's Guide To The Galaxy - BBC
# https://www.youtube.com/watch?v=5ZLtcTZP2js
_ANS_TO_THE_ULTIMATE_QUESTION_OF_LIFE=42

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_PATH = "hf://datasets/j4nusx/tourism/curated_prod_taken_full.csv"
tourism_df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ----------------------------
# Define the target variable
# ----------------------------
target = "ProdTaken"

# ----------------------------
# List of numerical features
# ----------------------------
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
] # complete the code to show list of all numerical feature

# ----------------------------
# List of categorical features
# ----------------------------
categorical_features = [
    "TypeofContact_Self Enquiry",
    "Occupation_Large Business",
    "Gender_Male",
    "MaritalStatus_Single",
    "MaritalStatus_Unmarried",
    "Designation_Executive",
    "Designation_Senior Manager",
] # complete the code to show list of all categorical feature

# ----------------------------
# Combine features to form X (feature matrix)
# ----------------------------
X = tourism_df[numeric_features + categorical_features]

# ----------------------------
# Define target vector y
# ----------------------------
y = tourism_df[target]   # complete the code to select the target column from tourism_df

# ----------------------------
# Split dataset into training and test sets
# ----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=_ANS_TO_THE_ULTIMATE_QUESTION_OF_LIFE
)

# Save splits to CSV files
output_dir = "TOURISM_PROJECT/RAW_DATA"
os.makedirs(output_dir, exist_ok=True)

Xtrain_path = os.path.join(output_dir, "Xtrain.csv")
Xtest_path = os.path.join(output_dir, "Xtest.csv")
ytrain_path = os.path.join(output_dir, "ytrain.csv")
ytest_path = os.path.join(output_dir, "ytest.csv")

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

files = [Xtrain_path, Xtest_path, ytrain_path, ytest_path]

# Upload each split file to the Hugging Face dataset repository
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="j4nusx/tourism",
        repo_type="dataset",
    )
