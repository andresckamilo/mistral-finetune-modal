from modal import forward
from .utils import (
    app,
    TIMEOUT,
    seed_volume,
    VOLUME_CONFIG,
    MODEL_TAR_URL,
    GPU_CONFIG,
)

# Define the model to be used. Supported models are:
# "7B Base V3", "7B Instruct V3", "8x7B Instruct V1", "8x22B Instruct V3", "8x22B Base V3"
model = "7B Base V3"

# Token for Jupyter Notebook authentication. Change this to something non-guessable!
JUPYTER_TOKEN = "1234"

@app.function(
    concurrency_limit=1,  # Limit the concurrency to 1 to avoid multiple instances running simultaneously
    volumes=VOLUME_CONFIG,  # Configuration for volumes to be used
    timeout=TIMEOUT,  # Timeout for the function
    gpu=GPU_CONFIG,  # GPU configuration
    _allow_background_volume_commits=True,  # Allow background volume commits
    keep_warm=1,  # Keep the function warm to avoid unexpected closing
)
def run_jupyter(timeout: int, *, TOKEN: str = "1234"):
    """
    Function to run the Jupyter Notebook server.

    Args:
        timeout (int): The timeout period in seconds.
        TOKEN (str): The token for Jupyter Notebook authentication.
    """
    import os
    import subprocess
    import time

    jupyter_port = 8888  # Port for Jupyter Notebook
    os.chdir("/root/content/mistral-finetune-modal/finetune_files")  # Change directory to the project root
    with forward(jupyter_port) as tunnel:  # Forward the Jupyter port
        jupyter_process = subprocess.Popen(
            [
                "jupyter",
                "lab",
                "--no-browser",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={jupyter_port}",
                "--NotebookApp.allow_origin='*'",
                "--NotebookApp.allow_remote_access=1",
            ],
            env={**os.environ, "JUPYTER_TOKEN": TOKEN},  # Set the environment variable for Jupyter token
        )

        print(f"Jupyter available at => {tunnel.url}")

        try:
            end_time = time.time() + timeout  # Calculate the end time
            while time.time() < end_time:  # Loop until the timeout period is reached
                time.sleep(5)  # Sleep for 5 seconds
            print(f"Reached end of {timeout} second timeout period. Exiting...")
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            jupyter_process.kill()  # Kill the Jupyter process

@app.local_entrypoint()
def main(timeout: int = TIMEOUT):
    """
    Main entry point for the application.

    Args:
        timeout (int): The timeout period in seconds. Default is 24 hours.
    """
    # Seed the volume with the model data
    seed_volume.remote(MODEL_TAR_URL[model])
    # Run the Jupyter Notebook server
    run_jupyter.remote(timeout, TOKEN=JUPYTER_TOKEN)
