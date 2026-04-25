from enum import Enum
from pathlib import Path
import requests
import tarfile

"""
Downloads the whole dataset from Hugging Face
"""
if __name__ == "__main__":

    upscale_option = "x2"  # change to X4 for the x4 dataset

    url = f"https://huggingface.co/datasets/KUOCHENG/STAR/resolve/main/data/{upscale_option}/{upscale_option}.tar.gz"

    DATA_PATH = Path(__file__).parent.parent / "dataset" / "data"
    TAR_PATH = DATA_PATH.parent / f"{upscale_option}.tar.gz"

    # download tar.gz file as stream and extract stream
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            tar.extractall(path=DATA_PATH)

    print("Done!")
