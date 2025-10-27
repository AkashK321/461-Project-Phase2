# src/scorer/url_handler/base.py (or wherever classify_url lives)

from urllib.parse import urlparse

# Common hosts (lowercase, no scheme)
_CODE_HOSTS = (
    "github.com", "gitlab.com", "bitbucket.org", "sourceforge.net",
    "codeberg.org", "gitee.com", "dev.azure.com", "azure.microsoft.com",
    "visualstudio.com",
)

# Places datasets commonly live
_DATASET_HOSTS = (
    "huggingface.co", "kaggle.com", "zenodo.org", "figshare.com",
    "osf.io", "openml.org", "archive.ics.uci.edu", "data.mendeley.com",
    "dataverse.org", "doi.org", "data.gov", "data.gov.uk",
)

# Direct-file/CDN/object-store links that are usually datasets
_STORAGE_HOSTS = (
    "s3.amazonaws.com",        # also covers bucket.s3.amazonaws.com
    "storage.googleapis.com",  # GCS
    "blob.core.windows.net",   # Azure Blob
    "drive.google.com",        # Google Drive
    "dropbox.com", "dl.dropboxusercontent.com",
    "onedrive.live.com", "1drv.ms",
)

# File extensions that strongly suggest "this is data"
_DATA_EXTS = (
    ".zip", ".tar", ".tar.gz", ".tgz", ".7z",
    ".csv", ".tsv", ".jsonl", ".json", ".parquet", ".feather",
    ".xlsx", ".xls", ".npz", ".npy", ".h5", ".hdf5", ".mat",
    ".wav", ".flac", ".mp3", ".jpg", ".jpeg", ".png", ".tiff"
)

# File extensions that look like code sources (when not on code hosts)
_CODE_EXTS = (".py", ".ipynb", ".c", ".cpp", ".h", ".hpp", ".java", ".js", 
              ".ts", ".m", ".go", ".rb", ".rs", ".php", ".scala", ".kt", ".sh")


# Extra model hosts (besides HF)
_MODEL_HOSTS = (
    "tfhub.dev",          # TensorFlow Hub
    "modelscope.cn",      # Alibaba ModelScope
)


# Fast helpers
def _endswith_any(s: str, suffixes: tuple[str, ...]) -> bool:
    s = s.lower()
    return any(s.endswith(sfx) for sfx in suffixes)

def classify_url(url: str) -> str:
    """
    Classify URL as "code", "dataset", "model", or "unknown".
    Heuristics (in order):
      1) Known code hosts => code
      2) Hugging Face paths: /datasets => dataset; otherwise => model
      3) Known model hosts (tfhub, modelscope) => model
      4) Known dataset hosts => dataset
      5) Storage/CDN & direct data-like file extensions => dataset
      6) Code-like file extensions (outside code hosts) => code
      7) Path keywords suggesting dataset => dataset
      8) Fallback: dataset (per course guidance: datasets can be any external link)
    """
    if not url:
        return "unknown"

    try:
        u = urlparse(url)
    except Exception:
        return "unknown"

    scheme = (u.scheme or "").lower()
    if scheme not in ("http", "https"):
        return "unknown"

    host = (u.netloc or "").lower()
    path = (u.path or "")
    path_l = path.lower()

    # 1) Clear code hosts
    if any(host == h or host.endswith("." + h) for h in _CODE_HOSTS):
        return "code"

    # 2) Hugging Face: /datasets => dataset; otherwise => model
    if host == "huggingface.co" or host.endswith(".huggingface.co"):
        if path_l.startswith("/datasets"):
            return "dataset"
        # exclude Spaces (apps) from model classification if you want:
        if path_l.startswith("/spaces"):
            # A Space isn't a model; consider it dataset-like content
            return "dataset"
        return "model"

    # 3) Other model hosts
    if any(host == h or host.endswith("." + h) for h in _MODEL_HOSTS):
        return "model"

    # 4) Known dataset hosts
    if any(host == h or host.endswith("." + h) for h in _DATASET_HOSTS):
        return "dataset"

    # 5) Storage/CDN or data-like file endings
    if any(host == h or host.endswith("." + h) for h in _STORAGE_HOSTS):
        return "dataset"
    if _endswith_any(path_l, _DATA_EXTS):
        return "dataset"

    # 6) Code-like file endings (outside code hosts)
    if _endswith_any(path_l, _CODE_EXTS):
        return "code"

    # 7) Path keywords that imply datasets
    if any(seg in path_l for seg in ("/dataset", "/datasets", "/data/", 
                                     "/download", "/files", "/record", "/records")):
        return "dataset"

    # 8) Fallback: treat as dataset (per Piazza: datasets can be any external link)
    return "dataset"
