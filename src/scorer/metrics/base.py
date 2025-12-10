"""
Helper function for splicing url
"""


def get_repo_id(url: str, url_type: str) -> str:
    url = url.strip().rstrip("/")
    parts = url.split("/")

    try:
        if url_type == "model":
            if "huggingface.co" in parts:
                index = parts.index("huggingface.co")
                remaining = len(parts) - (index + 1)

                if remaining >= 2:
                    # Standard: user/repo
                    repo_id = f"{parts[index + 1]}/{parts[index + 2]}"
                elif remaining == 1:
                    # Root-level model: repo
                    repo_id = parts[index + 1]
                else:
                    return None
            else:
                return None

        elif url_type == "dataset":
            if "huggingface.co" in parts:
                index = parts.index("huggingface.co")
                # Check for /datasets/ path
                if len(parts) > index + 1 and parts[index + 1] == "datasets":
                    remaining = len(parts) - (index + 2)
                    if remaining >= 2:
                        repo_id = f"{parts[index + 2]}/{parts[index + 3]}"
                    elif remaining == 1:
                        repo_id = parts[index + 2]
                    else:
                        return None
                else:
                    return None
            else:
                return None

        elif url_type == "code":
            if "github.com" in parts:
                index = parts.index("github.com")
                if len(parts) > index + 2:
                    repo_id = f"{parts[index + 1]}/{parts[index + 2]}"
                else:
                    return None
            else:
                return None

    except (ValueError, IndexError):
        print(f"Error parsing repo id for {url}")
        return None

    return repo_id
