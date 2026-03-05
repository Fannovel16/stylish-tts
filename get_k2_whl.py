import requests
from bs4 import BeautifulSoup
import torch
import sys

py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
torch_ver, device = torch.__version__.split("+")
device = device[: len("cu12")] + "." + device[len("cu12") :]
device = device.replace("cu", "cuda")
base_url = "https://k2-fsa.github.io/k2/installation/pre-compiled-cuda-wheels-linux/"
index_page = BeautifulSoup(requests.get(base_url).text)
whl_url = None
for a_el in BeautifulSoup(requests.get(base_url + torch_ver).text).select("a.external"):
    _whl_url = a_el.get("href")
    if device in _whl_url and py_version in _whl_url:
        whl_url = _whl_url
        break

if whl_url is None:
    raise RuntimeError(
        f"Wheel for torch {torch_ver}, {device} on {py_version} not found"
    )

print(whl_url, end="")
