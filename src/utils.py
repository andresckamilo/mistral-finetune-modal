import modal
from pathlib import PurePosixPath

# Constants for time calculations
MINUTES = 60
HOURS = 60 * MINUTES
GPU_CONFIG = "a100:1"
TIMEOUT = 6 * HOURS

# Dictionary containing model URLs and directories
MODEL_TAR_URL = {
    "7B Base V3": {
        "url": "https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-v0.3.tar",
        "model_dir": "mistral-7b-v0-3"
    },
    "7B Instruct V3": {
        "url": "https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-Instruct-v0.3.tar",
        "model_dir": "mistral-7b-v0-3"
    },
    "8x7B Instruct V1": {
        "url": "https://models.mistralcdn.com/mixtral-8x7b-v0-1/Mixtral-8x7B-v0.1-Instruct.tar",
        "model_dir": "mixtral-8x7b-v0-1"
    },
    "8x22B Instruct V3": {
        "url": "https://models.mistralcdn.com/mixtral-8x22b-v0-3/mixtral-8x22B-Instruct-v0.3.tar",
        "model_dir": "mixtral-8x22b-v0-3"
    },
    "8x22B Base V3": {
        "url": "https://models.mistralcdn.com/mixtral-8x22b-v0-3/mixtral-8x22B-v0.3.tar",
        "model_dir": "mixtral-8x22b-v0-3"
    }
}

# Create or retrieve a volume named "mistral-finetuning"
vol = modal.Volume.from_name(
    "mistral-finetune", create_if_missing=True
)

# Configuration for volume mapping
VOLUME_CONFIG: dict[str | PurePosixPath, modal.Volume] = {"/root/content": vol}

# Create a CUDA image with necessary dependencies
cuda_img = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "curl", "wget")
    .run_commands(
        "curl -o requirements.txt https://raw.githubusercontent.com/andresckamilo/mistral-finetune/modal_repo/finetune_files/requirements.txt\
            && pip install -r requirements.txt"
    )
)

# Create a Modal app with the specified image and secrets
app = modal.App(
    'mistral-notebook',
    image=cuda_img
)

@app.function(volumes=VOLUME_CONFIG, timeout=10 * MINUTES)
def seed_volume(model: dict[str, str]):
    """
    Function to seed the volume with the specified model data.

    Args:
        model (dict[str, str]): Dictionary containing model URL and directory.
    """
    import subprocess
    from pathlib import Path
    import tarfile
    import os
    import requests
    from urllib.parse import urlparse

    def clone_github_repo(repo_url, clone_to='.'):
        """
        Clone a GitHub repository to the specified directory.

        Args:
            repo_url (str): URL of the GitHub repository.
            clone_to (str): Directory to clone the repository into.
        """
        subprocess.run(['git', 'clone', repo_url, clone_to], check=True)

    def download_file(url, save_path):
        """
        Download a file from the specified URL to the given path.

        Args:
            url (str): URL of the file to download.
            save_path (str): Path to save the downloaded file.
        """
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Ensure the request was successful

        # Save the file to the specified path
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    # Check if the repository directory exists
    repo_dir = Path('/root/content/mistral-finetune-modal')

    if not os.path.exists(repo_dir):
        # Clone the repository if it doesn't exist
        repo_url = 'https://github.com/andresckamilo/mistral-finetune-modal.git'
        clone_github_repo(repo_url, repo_dir)

    finetune_files = repo_dir / 'finetune_files'
    # Define the directories using Path
    mistral_models_dir = finetune_files / 'mistral_models'
    model_dir = mistral_models_dir / model["model_dir"]

    # Create the directories if they don't exist
    mistral_models_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Download and extract the model tar file if the directory is empty
    if not any(model_dir.iterdir()):
        tar_url = model["url"]
        tar_save_path = os.path.join(model_dir, urlparse(tar_url).path.split('/')[-1])
        download_file(tar_url, tar_save_path)
        with tarfile.open(tar_save_path, 'r') as tar_ref:
            tar_ref.extractall(model_dir)
        os.remove(tar_save_path)
    
    # Commit the volume changes
    vol.commit()
