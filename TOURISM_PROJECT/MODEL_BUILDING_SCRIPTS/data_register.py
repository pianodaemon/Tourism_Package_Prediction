import os

from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo

repo_id = "j4nusx/tourism"
repo_type="dataset"

# Before initializing the Hugging Face API client using the HF_TOKEN environment variable
# 1. Generate a Hugging Face access token from your Hugging Face account.
# 2. In Colab, click the Secrets tab in the left sidebar.
# 3. Create a new secret named HF_TOKEN.
# 4. Paste your Hugging Face access token as the value.
# 5. Grant notebook access to the secret.

# Initialize the Hugging Face API client using the HF_TOKEN environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the repository already exists; if not, create it
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    # complete the code to create a new public repository on Hugging Face
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Upload the local data folder to the Hugging Face dataset repository
api.upload_folder(
    folder_path='TOURISM_PROJECT/RAW_DATA',
    repo_id=repo_id,
    repo_type=repo_type,
)
