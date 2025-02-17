import time
import requests
from tqdm import tqdm


def simple_download(url: str, filename: str, chunk_size: int = 1024 * 1024):
    """Download the file in a single request with progress bar."""
    start_time = time.time()

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise error if request fails

    # Get the total file size from the response headers
    total_size = int(response.headers.get("Content-Length", 0))

    # Create a progress bar using tqdm
    with (
        open(filename, "wb") as file,
        tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as bar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            file.write(chunk)
            bar.update(len(chunk))  # Update the progress bar with the chunk size

    end_time = time.time()
    print(f"\nDownloaded {filename} in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    url = "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_caption/model.safetensors"
    filename = "model-simple.safetensors"
    simple_download(url, filename, chunk_size=1024 * 1024 * 1)

