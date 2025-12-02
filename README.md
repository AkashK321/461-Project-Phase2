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

*Note: Use the default admin credentials provided in the project documentation to perform the initial login and user setup.*
---

## System Usage

### 1. Web Interface
The primary way to interact with the registry is through the hosted web application.

* **Dashboard:** View high-level registry statistics and access navigation menus.
* **Search:** Find models using metadata queries or Regular Expressions. Supports version constraints (e.g., `^1.2.0`).
* **Upload:**
    * **Zip Upload:** Upload a zipped model package directly.
    * **Ingest:** Import a model directly from HuggingFace. The system automatically checks if the model meets the quality threshold (Score > 0.5).
* **Model Details:** View comprehensive scores (Ramp-up, Correctness, Bus Factor, etc.) and the **Lineage Graph** to see model version history.

### 2. REST API
The system exposes a fully compliant OpenAPI REST interface for programmatic access.

**Authentication:**
All API requests (except login) require an authentication token.
1.  **Login:** POST to `/authenticate` with your credentials to receive a bearer token.
2.  **Use Token:** Include the token in the `X-Authorization` header of subsequent requests.

**Key Endpoints:**
* `POST /packages`: Get a list of packages matching a search query.
* `POST /package`: Upload a new package or ingest from a public registry.
* `GET /package/{id}`: Retrieve package content and metadata.
* `GET /package/{id}/rate`: Retrieve the calculated trust scores for a specific model.
* `DELETE /reset`: **(Admin Only)** Reset the registry to its default state.

---

## Developer Setup (Docker)

While the system is designed to be used via the Web UI and API, developers can use Docker for local testing and contribution.

### Prerequisites
* Docker & Docker Compose
* Git

### Local Development Commands
The `scorer` service is containerized to ensure consistent results across environments.

1.  **Build and Start Services:**
    ```bash
    docker-compose up -d --build
    ```
2.  **Access the Development Shell:**
    ```bash
    docker-compose exec scorer bash
    ```
3.  **Available Tools:**
    Inside the container, use the `./run` utility to interact with the Python backend:
    * `./run install` - Install dependencies.
    * `./run test` - Run the full test suite with coverage reports.
    * `./run src/scorer/urls.txt` - Manually score a list of repositories via CLI.

---

## Deployment

The system is deployed automatically to **AWS** using **GitHub Actions**.

### CI/CD Pipeline
The workflow defined in `.github/workflows/cd.yml` handles the following:
1.  **Triggers:** Pushes to the `main` branch (and specific feature branches).
2.  **Backend Deployment:**
    * Packages the `src/` directory and `requirements.txt`.
    * Updates the AWS Lambda functions: `scorer_function` and `registry_api`.
3.  **Frontend Deployment:**
    * Syncs the `frontend/` directory to the AWS S3 bucket `ece461team11-frontend`.

### AWS Infrastructure
* **Compute:** AWS Lambda (Serverless backend).
* **Storage:** AWS S3 (Frontend hosting and artifact storage).
* **Region:** `us-east-2`.

---

## License
Project created for ECE 461.