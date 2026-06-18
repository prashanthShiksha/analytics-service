# Analytics Service

An orchestration and ingestion service built with FastAPI, Kafka, and Temporal. It manages dynamic data ingestion pipelines, rule-based content moderation, local NLP vector similarity matching, and fallback LLM classification.

---

## Setup & Running Guide

### Prerequisites

Ensure the following dependencies are installed and running locally:
1. **Python**: Version 3.9+
2. **PostgreSQL**: Running and initialized with the database schema
3. **Apache Kafka & Zookeeper**: Running on `localhost:9092`
4. **Temporal Server**: Running on `localhost:7233` (e.g. via `temporal server start-dev` or Docker)

### Installation

1. Clone the repository and navigate to the project root.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   Copy `.env.example` to `.env` and fill in the required configurations (such as database credentials and your OpenRouter API key):
   ```bash
   cp .env.example .env
   ```

### Temporal Setup via Docker

A [docker-compose.yaml](file:///Users/user/Documents/AI/analytics-arch/analytics_service/docker-compose.yaml) file is provided to start the local Temporal server and its UI dashboard.

To start Temporal:
```bash
docker-compose up -d
```
This launches:
* **Temporal Server** on port `7233`
* **Temporal Web UI** on port `8233`

The configuration automatically points to your local host's PostgreSQL instance (`host.docker.internal`). Ensure PostgreSQL is running and has the `temporal` and `temporal_visibility` databases created prior to starting the containers.

#### Configuration ([docker-compose.yaml](file:///Users/user/Documents/AI/analytics-arch/analytics_service/docker-compose.yaml)):
```yaml
version: '3.8'

services:
  temporal:
    image: temporalio/auto-setup:1.24
    container_name: temporal
    ports:
      - "7233:7233"
    environment:
      - DB=postgres12
      - POSTGRES_SEEDS=host.docker.internal
      - POSTGRES_USER=postgres
      - POSTGRES_PWD=postgres
      - DB_PORT=5432
      - TEMPORAL_DB=temporal
      - TEMPORAL_VISIBILITY_DB=temporal_visibility
    extra_hosts:
      - "host.docker.internal:host-gateway"

  temporal-ui:
    image: temporalio/ui:2.34.0
    container_name: temporal-ui
    ports:
      - "8233:8080"
    environment:
      - TEMPORAL_ADDRESS=temporal:7233
      - TEMPORAL_UI_PORT=8080
    depends_on:
      - temporal
```

### Running the Application

The **`main.py`** entry point supports starting different components using the `--mode` flag:

```bash
# Default – launches web server, Kafka consumer, and Temporal worker concurrently
python main.py

# Launch specific services separately
python main.py --mode web        # Starts only the FastAPI web server
python main.py --mode consumer   # Starts only the Kafka consumer
python main.py --mode worker     # Starts only the Temporal worker
```

---

## Health Endpoint

Verify that all systems are running by calling the health check API:
```bash
curl http://localhost:8000/health
```
Example response:
```json
{
  "status": "healthy",
  "consumer_running": true,
  "worker_running": true
}
```

---

## Ingesting & Testing with Kafka

To push a mock story event into the Kafka ingestion topic (`analytics.ingestion.raw`), you can use the following command:

```bash
/Users/user/.pyenv/versions/3.9.19/bin/python -c 'import json, json5, pathlib; text = pathlib.Path("/Users/user/Documents/AI/analytics-arch/analytics_service/tests/kafka_events/create/create_story.json").read_text(encoding="utf-8"); print(json.dumps(json5.loads(text), separators=(",", ":")))' | /Users/user/Documents/shikshalokam/elevate-analytics/tools/kafka_2.12-3.7.1/bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic analytics.ingestion.raw
```

---

## Thematic Classification Pipeline

Thematic classification is implemented as a gated, multi-step pipeline inside [thematic_activity.py](file:///Users/user/Documents/AI/analytics-arch/analytics_service/app/temporal/thematic_activity.py). Below are the sequential steps executed for each target text column:

1. **Read Column Config**: Resolves target columns from settings (e.g. `objective` or `challenges`) and extracts the corresponding statement texts.
2. **Discussion Splitting (Step 1b)**: For discussion submission types, splits cells using a delimiter (`|` by default) into separate statements so each point is classified individually.
3. **Word-Count & Garbage Check (Step 2)**: Checks if the statement word count is less than `MINIMUM_THEME_WORD_COUNT` (default 5) or matches repetitive spam patterns. If so, flags it as `Unknown/Unclear` and **stops** processing.
4. **Local Safety & Moderation (Step 3)**: Performs a non-LLM safety check against sensitive PII patterns (email, Indian phone numbers, Aadhaar, PAN) and abusive/profane keywords. If flagged, sets `content_quality = 'Flagged'` and **stops** processing.
5. **Fetch Approved Taxonomies (Step 5)**: Queries the database for all taxonomies/themes currently marked with an `'Approved'` status.
6. **Local Embedding Match (Step 6)**: Employs a local `SentenceTransformer` (`all-MiniLM-L6-v2`) to encode the statement and calculate cosine similarities against all approved themes.
7. **Evaluate Local Threshold (Step 7)**: If the highest cosine similarity score $\ge$ `SIMILARITY_SCORE_THRESHOLD` (default 0.65), maps the statement directly to that theme, sets `content_quality = 'Standard'`, saves the rounded score to the database, and skips the LLM call entirely.
8. **Fallback LLM Request (Step 8)**: If the local embedding match similarity is below the threshold, retrieves the active `theme_classification` prompt template from the database, builds the user prompt by inserting the statement and approved themes lists, and issues an API request to OpenRouter.
9. **Assess LLM Confidence & Finalize (Step 9)**: 
   - If the LLM successfully maps the statement with a confidence score $\ge$ `LLM_CONFIDENCE_SCORE_THRESHOLD` (default 0.8), sets `content_quality = 'Standard'` and updates the DB with the mapped theme.
   - If both the local embedding score and the LLM confidence score fall below their respective thresholds, sets `content_quality = 'Others'`.

