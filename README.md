# Trustworthy Model Registry (Phase 2)

## Project Overview
This project implements a **Trustworthy Model Registry** for ACME Corporation. It provides a secure, centralized system to upload, store, score, and manage machine learning models. Unlike public registries, this system focuses on trust and reliability by automatically calculating metrics such as reproducibility, code review coverage ("reviewedness"), and maintainer activity ("bus factor") for every model.

**Key Features:**
* **Model Ingestion:** Import models directly from HuggingFace or upload custom zip packages.
* **Automated Scoring:** Evaluates models on metrics like Ramp-up, Correctness, Bus Factor, Responsiveness, and License compatibility.
* **Lineage Tracking:** Visualizes the relationship between model versions and their parents.
* **Web Interface:** A user-friendly frontend to search, view, and manage models.
* **REST API:** A fully compliant OpenAPI backend for programmatic access.

## Live Application
* **Frontend URL:** [http://ece461team11-frontend.s3-website.us-east-2.amazonaws.com](http://ece461team11-frontend.s3-website.us-east-2.amazonaws.com)

---

## Configuration

### Prerequisites
* Docker & Docker Compose
* Git
* API Keys (HuggingFace, GitHub, etc.)

### Environment Variables
To run the system, you must configure your environment variables.
1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    ```
2.  Open `.env` and populate the following required variables:
    * `HF_TOKEN`: Your HuggingFace API token (for fetching model metadata).
    * `GITHUB_TOKEN`: Your GitHub Personal Access Token (for checking repo metrics).
    * `GEN_AI_STUDIO_API_KEY`: API key for the GenAI service (used for some scoring logic).
    * `GENAI_BASE_URL`: Base URL for the GenAI provider (default provided).

---

## Installation & Local Development

This project uses **Docker** to ensure a consistent environment for development and testing.

### Running with Docker Compose (Recommended)
The easiest way to start the application and the scoring tool is via Docker Compose.

1.  **Build and Start Services:**
    ```bash
    docker-compose up -d --build
    ```
2.  **Access the Container:**
    ```bash
    docker-compose exec scorer bash
    ```
3.  **Run Commands Inside Container:**
    Once inside the shell, you can use the `./run` utility:
    * `./run install` - Install dependencies.
    * `./run test` - Run the test suite with coverage.
    * `./run urls.txt` - Process and score a list of URLs.

### Manual Local Setup (Optional)
If you prefer running without Docker:
1.  Install Python 3.11+.
2.  Install dependencies: `pip install -r requirements.txt`.
3.  Run the CLI tool: `python3 src/scorer/cli_updated.py [URL_FILE]`.

---

## Testing

We prioritize high code coverage and reliability.

* **Run All Tests:**
    ```bash
    # From inside the docker container
    ./run test
    ```
* **Test Output:**
    * Results are output to the console.
    * Detailed coverage reports are generated in the `.coverage` file.

---

## Deployment

The system is deployed automatically to **AWS** using **GitHub Actions**.

### CI/CD Pipeline
The workflow defined in `.github/workflows/cd.yml` handles the following:
1.  **Triggers:** Pushes to the `main` branch (and specific feature branches).
2.  **Backend Deployment:** * Packages the `src/` directory and `requirements.txt`.
    * Updates the AWS Lambda functions: `scorer_function` and `registry_api`.
3.  **Frontend Deployment:**
    * Syncs the `frontend/` directory to the AWS S3 bucket `ece461team11-frontend`.

### AWS Infrastructure
* **Compute:** AWS Lambda (Serverless backend).
* **Storage:** AWS S3 (Frontend hosting and artifact storage).
* **Region:** `us-east-2`.

---

## Interaction

### Using the Web Interface
Navigate to the [Frontend URL](http://ece461team11-frontend.s3-website.us-east-2.amazonaws.com) to:
* Search for models by name or regex.
* Upload new models (Zip format) or ingest from HuggingFace.
* View model details, including their calculated trust scores and lineage.

### Using the CLI (Scorer)
You can use the scorer tool to evaluate packages directly:
```bash
# Evaluate a list of GitHub or HuggingFace URLs
./run src/scorer/urls.txt