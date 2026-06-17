import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional
from temporalio import activity

from app.config import settings
from app.database.db import db
from app.database.operations import insert_llm_log, get_submission_type_and_payload
from app.services.pii import mask_pii_text
from app.services.image_blur import anonymize_face

logger = logging.getLogger("analytics_service.temporal.activities")

BASE_DIR = Path(__file__).resolve().parents[2]
DOWNLOADS_DIR = BASE_DIR / "downloads"
OUTPUTS_DIR = BASE_DIR / "outputs"

def _download_file(url: str, filename: str) -> Path:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = DOWNLOADS_DIR / filename
    logger.info(f"Downloading {url} to {local_path}")
    
    with urllib.request.urlopen(url, timeout=60) as response:
        with open(local_path, "wb") as f:
            f.write(response.read())
    return local_path

async def _get_prompt_version_id(conn, analysis_type: str) -> str:
    """
    Fetch the active prompt template from the database.
    The seed script populates these rows at startup; the workflow uses them directly.
    """
    row = await conn.fetchrow(
        """
        SELECT pv.id
        FROM prompt_version pv
        JOIN prompts p ON p.id = pv.prompt_id
        WHERE p.analysis_type = $1 AND pv.is_active = TRUE
        ORDER BY pv.created_at DESC
        LIMIT 1
        """,
        analysis_type,
    )
    if not row:
        raise RuntimeError(f"No active {analysis_type} prompt version found in the database.")
    return str(row["id"])

@activity.defn
async def pii_detection_activity(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Temporal activity that masks PII for specified columns in database and updates state.
    """
    submission_id = params["submission_id"]
    tenant_code = params["tenant_code"]
    target_columns = params["target_columns"]

    if not target_columns:
        return {"status": "skipped", "reason": "no columns to mask"}

    async with db.pool.acquire() as conn:
        sub_type, payload = await get_submission_type_and_payload(conn, submission_id, tenant_code)
        prompt_version_id = await _get_prompt_version_id(conn, "pii")

        updated_fields = {}
        # Mock token metrics since OpenRouter free tier doesn't always return usage reliably
        prompt_tokens_estimated = 0
        completion_tokens_estimated = 0

        for col in target_columns:
            # Map challenges/solutions or specific DB column singular name
            db_col = "challenge" if col == "challenges" and sub_type == "story" else col
            
            raw_text = payload.get(db_col)
            if not raw_text:
                continue

            try:
                masked_text = mask_pii_text(raw_text)
                updated_fields[db_col] = masked_text
                
                # Update estimations
                prompt_tokens_estimated += len(raw_text.split())
                completion_tokens_estimated += len(masked_text.split())

            except Exception as e:
                logger.error(f"PII masking failed on column {col}: {e}")
                # Log failed attempt
                await insert_llm_log(
                    conn, submission_id, tenant_code, settings.OPENROUTER_MODEL,
                    "pii", prompt_version_id,
                    prompt_tokens_estimated, completion_tokens_estimated,
                    "failed", error_message=str(e)
                )
                raise

        # Save updates to DB
        if updated_fields:
            if sub_type == "story":
                set_clauses = ", ".join(f"{col} = ${i+3}" for i, col in enumerate(updated_fields.keys()))
                values = list(updated_fields.values())
                query = f"UPDATE story_submissions SET {set_clauses}, content_masked = TRUE, updated_at = now() WHERE submission_id = $1 AND tenant_code = $2"
                await conn.execute(query, submission_id, tenant_code, *values)
            else:
                set_clauses = ", ".join(f"{col} = ${i+3}" for i, col in enumerate(updated_fields.keys()))
                values = list(updated_fields.values())
                query = f"UPDATE discussion_submissions SET {set_clauses}, content_masked = TRUE, updated_at = now() WHERE submission_id = $1 AND tenant_code = $2"
                await conn.execute(query, submission_id, tenant_code, *values)

            await insert_llm_log(
                conn, submission_id, tenant_code, settings.OPENROUTER_MODEL,
                "pii", prompt_version_id,
                prompt_tokens_estimated, completion_tokens_estimated,
                "success"
            )

        return {"status": "success", "updated_columns": list(updated_fields.keys())}


@activity.defn
async def deface_blur_activity(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Temporal activity that downloads and runs local OpenCV/ONNX face blurring on ingestion images.
    """
    submission_id = params["submission_id"]
    tenant_code = params["tenant_code"]

    async with db.pool.acquire() as conn:
        sub_type, payload = await get_submission_type_and_payload(conn, submission_id, tenant_code)
        
        image_urls = payload.get("image_urls")
        if not image_urls:
            return {"status": "skipped", "reason": "no image urls available"}

        blurred_local_paths = []
        for i, url in enumerate(image_urls):
            filename = f"{submission_id}_{tenant_code}_{i}.jpg"
            try:
                # 1. Download file locally
                local_path = _download_file(url, filename)
                
                # 2. Deface image
                OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
                output_path = OUTPUTS_DIR / f"blurred_{filename}"
                
                anonymize_face(
                    input_path=str(local_path),
                    output_path=str(output_path)
                )
                
                blurred_local_paths.append(str(output_path))
            except Exception as e:
                logger.error(f"Failed face blurring for {url}: {e}")
                raise

        # Save output paths back to DB
        # Note: In production this would upload to S3/GCS and save public URLs
        if blurred_local_paths:
            if sub_type == "story":
                await conn.execute(
                    "UPDATE story_submissions SET blur_image_urls = $3, updated_at = now() WHERE submission_id = $1 AND tenant_code = $2",
                    submission_id, tenant_code, blurred_local_paths
                )
            else:
                await conn.execute(
                    "UPDATE discussion_submissions SET blur_image_urls = $3, updated_at = now() WHERE submission_id = $1 AND tenant_code = $2",
                    submission_id, tenant_code, blurred_local_paths
                )

        return {"status": "success", "blur_paths": blurred_local_paths}


@activity.defn
async def update_status_activity(params: Dict[str, Any]) -> None:
    """
    Temporal activity to update the overall processing status of a submission in PostgreSQL.
    """
    submission_id = params["submission_id"]
    tenant_code = params["tenant_code"]
    status = params["status"]
    process_status = params.get("process_status")

    async with db.pool.acquire() as conn:
        from app.database.operations import update_submission_status
        await update_submission_status(conn, submission_id, tenant_code, status, process_status)


@activity.defn
async def fetch_pending_submissions_activity() -> List[Dict[str, Any]]:
    """
    Retrieves all submissions currently in a 'pending' state and attaches their config-driven process steps.
    """
    async with db.pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT submission_id, tenant_code, submission_type FROM submissions WHERE status = 'pending'"
        )
        results = []
        for row in rows:
            sub_id = row["submission_id"]
            tenant = row["tenant_code"]
            sub_type = row["submission_type"]
            # Load process steps dynamically from settings based on type
            process_steps = settings.get_process_config(sub_type)
            results.append({
                "submission_id": sub_id,
                "tenant_code": tenant,
                "submission_type": sub_type,
                "process_steps": process_steps
            })
        return results



