import os
from huggingface_hub import HfApi, create_repo

api = HfApi(token=os.getenv("HF_TOKEN"))
repo_id = "j4nusx/tourism-app"

# Ensure the Space exists on HF with Docker SDK before uploading
try:
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        token=os.getenv("HF_TOKEN")
    )
    print(f"Space '{repo_id}' created successfully.")
except Exception as e:
    if "RepositoryAlreadyExistsError" in str(type(e).__name__) or "409" in str(e):
        print(f"Space '{repo_id}' already exists. Proceeding to upload.")
    else:
        print(f"Notice: {e}")

# Upload deployment files
api.upload_folder(
    folder_path="TOURISM_PROJECT/DEPLOYMENT_SCRIPTS",
    repo_id=repo_id,
    repo_type="space",
    path_in_repo="",
)
print("Upload successful!")
