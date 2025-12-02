# Trustworthy Model Registry (Phase 2)

## Project Overview
This project implements a **Trustworthy Model Registry and Scorer**. It provides a secure, centralized system to upload, store, score, and manage machine learning models.

**Key Features:**
* **Model Ingestion:** Import models directly from HuggingFace or upload custom zip packages.
* **Automated Scoring:** Evaluates models on metrics like Ramp-up, Correctness, Bus Factor, Responsiveness, and License compatibility.
* **Lineage Tracking:** Visualizes the relationship between model versions and their parents.
* **Web Interface:** A user-friendly frontend to search, view, and manage models.
* **REST API:** A fully compliant OpenAPI backend for programmatic access.

## Live Application
* **Frontend URL:** [http://ece461team11-frontend.s3-website.us-east-2.amazonaws.com](http://ece461team11-frontend.s3-website.us-east-2.amazonaws.com)

---

## Authentication & Configuration

### Environment Variables
**No manual configuration is required.**
Users do not need to set up environment variables or `.env` files. All system configurations, API keys (e.g., HuggingFace, GitHub), and secrets are securely stored and managed directly within the **AWS Lambda** environment.

### Access Control
To ensure security, the system enforces strict authentication. You must log in before accessing any features.

* **Authentication Requirement:** Users must authenticate before accessing any aspect of the system (searching, uploading, downloading, etc.).
* **User Roles:**
    * **Admin:** Has full access to the system, including the ability to create and delete other user accounts.
    * **Regular User:** Can perform standard registry operations. These accounts must be created by an Admin; they cannot self-register.

* After a successful login, an authentication token will be created and stored by the user's browser for 10 hours. After the token expires, users must log in again.

*(Note: Use the default admin credentials provided in the project documentation to perform the initial login and user setup.)*
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
We use pytest to run all tests, and Selenium to run automated tests for the front-end UI.

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