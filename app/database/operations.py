import json
import logging
from typing import Dict, Any, Optional, List
import asyncpg
from datetime import datetime

logger = logging.getLogger("analytics_service.operations")

def _normalize_string_list(value: Any) -> Optional[str]:
    """Helper to convert list or string to database TEXT."""
    if value is None:
        return None
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)

def _normalize_url_list(value: Any) -> List[str]:
    """Helper to convert list or single string URL to database TEXT[]."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(url) for url in value]
    return [str(value)]

async def upsert_metadata(conn: asyncpg.Connection, data: Dict[str, Any], tenant_code: str) -> tuple:
    """
    Safely upserts parent metadata tables: tenant, leader_category, and programs.
    Returns (program_id, leader_id) as UUIDs.
    """
    # 1. Upsert Tenant
    tenant_name = tenant_code.capitalize()
    await conn.execute(
        """
        INSERT INTO tenant (name, code)
        VALUES ($1, $2)
        ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name, updated_at = now()
        """,
        tenant_name, tenant_code
    )

    leader_id = None
    program_id = None

    # 2. Upsert Leader Category Info
    leader_info = data.get("LeaderCategoryInfo")
    if leader_info and leader_info.get("id"):
        leader_id = leader_info["id"]
        await conn.execute(
            """
            INSERT INTO leader_category (id, name, description, tenant_code)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (id) DO UPDATE SET 
                name = EXCLUDED.name, 
                description = EXCLUDED.description, 
                updated_at = now()
            """,
            leader_id,
            leader_info.get("name", ""),
            leader_info.get("description"),
            tenant_code
        )

    # 3. Upsert Programs Info (depends on leader category)
    program_info = data.get("programInfo")
    if program_info and program_info.get("id") and leader_id:
        program_id = program_info["id"]
        await conn.execute(
            """
            INSERT INTO programs (id, leaders_id, name, description, tenant_code)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (id) DO UPDATE SET 
                leaders_id = EXCLUDED.leaders_id, 
                name = EXCLUDED.name, 
                description = EXCLUDED.description, 
                updated_at = now()
            """,
            program_id,
            leader_id,
            program_info.get("name", ""),
            program_info.get("description"),
            tenant_code
        )

    return program_id, leader_id

async def delete_submission(conn: asyncpg.Connection, submission_id: str, tenant_code: str) -> bool:
    """
    Deletes a submission and cascades down to specific table rows.
    """
    result = await conn.execute(
        "DELETE FROM submissions WHERE submission_id = $1 AND tenant_code = $2",
        submission_id, tenant_code
    )
    # result string format: e.g. "DELETE 1"
    deleted = result.startswith("DELETE") and not result.endswith("0")
    if deleted:
        logger.info(f"Deleted submission {submission_id} under tenant {tenant_code}")
    else:
        logger.warning(f"Submission {submission_id} not found under tenant {tenant_code} for deletion")
    return deleted

async def insert_or_update_submission(
    conn: asyncpg.Connection,
    event_payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Performs transactional write for submission master and type-specific tables.
    """
    submission_id = str(event_payload["submissionId"])
    tenant_code = event_payload["tenantCode"]
    submission_type = event_payload["submissionType"]
    session_id = event_payload.get("sessionId")
    data = event_payload.get("data", {})

    # Start database transaction if not already handled
    async with conn.transaction():
        # Upsert parent metadata tables
        program_id, leader_id = await upsert_metadata(conn, data, tenant_code)

        # Parse submission date
        sub_date_str = data.get("submissionDate")
        if sub_date_str:
            submission_date = datetime.fromisoformat(sub_date_str.replace("Z", "+00:00"))
        else:
            submission_date = datetime.utcnow()

        # 1. Upsert Master Submission record
        # Note: status is initialized as 'pending' for new creates
        submission_uuid_row = await conn.fetchrow(
            """
            INSERT INTO submissions (
                session_id, submission_id, tenant_code, submission_type, user_id, user_name, role,
                state, district, organization, submission_date, program_id, leader_id, status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (submission_id, tenant_code) DO UPDATE SET
                session_id = COALESCE(EXCLUDED.session_id, submissions.session_id),
                user_id = EXCLUDED.user_id,
                user_name = EXCLUDED.user_name,
                role = EXCLUDED.role,
                state = EXCLUDED.state,
                district = EXCLUDED.district,
                organization = EXCLUDED.organization,
                submission_date = EXCLUDED.submission_date,
                program_id = EXCLUDED.program_id,
                leader_id = EXCLUDED.leader_id,
                status = EXCLUDED.status,
                updated_at = now()
            RETURNING id, status;
            """,
            session_id,
            submission_id,
            tenant_code,
            submission_type,
            data.get("userId"),
            data.get("userName"),
            data.get("designation"), # Designation maps to role
            data.get("state"),
            data.get("district"),
            data.get("organization"),
            submission_date,
            program_id,
            leader_id,
            "pending"
        )

        db_sub_uuid = submission_uuid_row["id"]
        db_sub_status = submission_uuid_row["status"]

        # 2. Upsert specific payload type
        normalized_type = submission_type.lower().strip()
        if "story" in normalized_type:
            # Upsert story submission
            # Update if exists, else insert
            row_exists = await conn.fetchval(
                "SELECT 1 FROM story_submissions WHERE submission_id = $1 AND tenant_code = $2",
                submission_id, tenant_code
            )
            challenges_joined = _normalize_string_list(data.get("challenges"))
            action_steps_joined = _normalize_string_list(data.get("actionSteps"))
            image_urls = _normalize_url_list(data.get("imageUrls"))
            pdf_urls = _normalize_url_list(data.get("pdfUrls"))

            if row_exists:
                await conn.execute(
                    """
                    UPDATE story_submissions SET
                        title = $3,
                        objective = $4,
                        challenge = $5,
                        action_steps = $6,
                        impact = $7,
                        duration = $8,
                        blurb = $9,
                        content = $10,
                        image_urls = $11,
                        pdf_urls = $12,
                        transcript_link = $13,
                        updated_at = now()
                    WHERE submission_id = $1 AND tenant_code = $2
                    """,
                    submission_id, tenant_code,
                    data.get("title"),
                    data.get("objective"),
                    challenges_joined,
                    action_steps_joined,
                    data.get("impact"),
                    data.get("duration"),
                    data.get("blurb"),
                    data.get("content"),
                    image_urls,
                    pdf_urls,
                    data.get("transcriptLink")
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO story_submissions (
                        submission_id, tenant_code, title, objective, challenge, action_steps,
                        impact, duration, blurb, content, image_urls, pdf_urls, transcript_link
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                    submission_id, tenant_code,
                    data.get("title"),
                    data.get("objective"),
                    challenges_joined,
                    action_steps_joined,
                    data.get("impact"),
                    data.get("duration"),
                    data.get("blurb"),
                    data.get("content"),
                    image_urls,
                    pdf_urls,
                    data.get("transcriptLink")
                )

        elif "discussion" in normalized_type:
            # Upsert discussion submission
            row_exists = await conn.fetchval(
                "SELECT 1 FROM discussion_submissions WHERE submission_id = $1 AND tenant_code = $2",
                submission_id, tenant_code
            )
            challenges_joined = _normalize_string_list(data.get("challenges"))
            solutions_joined = _normalize_string_list(data.get("solutions"))
            image_urls = _normalize_url_list(data.get("imageUrls"))
            pdf_urls = _normalize_url_list(data.get("pdfUrls"))

            if row_exists:
                await conn.execute(
                    """
                    UPDATE discussion_submissions SET
                        title = $3,
                        challenges = $4,
                        solutions = $5,
                        author = $6,
                        language = $7,
                        image_urls = $8,
                        pdf_urls = $9,
                        transcript_link = $10,
                        updated_at = now()
                    WHERE submission_id = $1 AND tenant_code = $2
                    """,
                    submission_id, tenant_code,
                    data.get("title"),
                    challenges_joined,
                    solutions_joined,
                    data.get("author"),
                    data.get("language"),
                    image_urls,
                    pdf_urls,
                    data.get("transcriptLink")
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO discussion_submissions (
                        submission_id, tenant_code, title, challenges, solutions,
                        author, language, image_urls, pdf_urls, transcript_link
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    submission_id, tenant_code,
                    data.get("title"),
                    challenges_joined,
                    solutions_joined,
                    data.get("author"),
                    data.get("language"),
                    image_urls,
                    pdf_urls,
                    data.get("transcriptLink")
                )

        logger.info(f"Successfully ingested {submission_type} submission {submission_id} under tenant {tenant_code}")
        return {
            "id": db_sub_uuid,
            "submission_id": submission_id,
            "tenant_code": tenant_code,
            "status": db_sub_status
        }

async def update_submission_status(
    conn: asyncpg.Connection,
    submission_id: str,
    tenant_code: str,
    status: str,
    process_status: Optional[Dict[str, Any]] = None
) -> None:
    """
    Updates the execution status and status metadata on the master submissions table.
    """
    if process_status:
        process_status_json = json.dumps(process_status)
        await conn.execute(
            """
            UPDATE submissions 
            SET status = $3, process_status = $4, updated_at = now()
            WHERE submission_id = $1 AND tenant_code = $2
            """,
            submission_id, tenant_code, status, process_status_json
        )
    else:
        await conn.execute(
            """
            UPDATE submissions 
            SET status = $3, updated_at = now()
            WHERE submission_id = $1 AND tenant_code = $2
            """,
            submission_id, tenant_code, status
        )

async def insert_llm_log(
    conn: asyncpg.Connection,
    submission_id: str,
    tenant_code: str,
    model_name: str,
    analysis_type: str,
    prompt_version_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    status: str,
    error_message: Optional[str] = None,
    meta_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Logs metadata about individual LLM executions to the database for token tracking.
    """
    meta_json = json.dumps(meta_data) if meta_data else None
    await conn.execute(
        """
        INSERT INTO llm_logs (
            submission_id, tenant_code, model_name, analysis_type, prompt_version_id,
            prompt_tokens, completion_tokens, status, error_message, meta_data
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """,
        submission_id,
        tenant_code,
        model_name,
        analysis_type,
        prompt_version_id,
        prompt_tokens,
        completion_tokens,
        status,
        error_message,
        meta_json
    )

async def insert_analysis_result(
    conn: asyncpg.Connection,
    submission_id: str,
    tenant_code: str,
    theme_id: Optional[str],
    analysis_type: str,
    statements: str,
    statement_type: str,
    confidence_score: Optional[float] = None,
    justification: Optional[str] = None,
    content_quality: Optional[str] = None,
    similarity_score: Optional[float] = None,
    multi_theme_mapped: bool = False,
    meta_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Saves theme/environmental extraction analysis output to database.
    """
    if similarity_score is not None:
        similarity_score = round(similarity_score, 2)
    meta_json = json.dumps(meta_data) if meta_data else None
    await conn.execute(
        """
        INSERT INTO analysis_results (
            submission_id, tenant_code, theme_id, analysis_type, statements,
            statement_type, confidence_score, justification, content_quality,
            similarity_score, multi_theme_mapped, meta_data
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        """,
        submission_id,
        tenant_code,
        theme_id,
        analysis_type,
        statements,
        statement_type,
        confidence_score,
        justification,
        content_quality,
        similarity_score,
        multi_theme_mapped,
        meta_json
    )


async def get_submission_type_and_payload(conn: asyncpg.Connection, submission_id: str, tenant_code: str) -> tuple:
    """
    Retrieves the submission type and payload details for story or discussion submissions.
    """
    sub_row = await conn.fetchrow(
        "SELECT submission_type FROM submissions WHERE submission_id = $1 AND tenant_code = $2",
        submission_id, tenant_code
    )
    if not sub_row:
        raise ValueError(f"Submission {submission_id} not found in database.")
    
    sub_type = sub_row["submission_type"].lower().strip()
    
    if "story" in sub_type:
        payload_row = await conn.fetchrow(
            """
            SELECT title, objective, challenge, action_steps, impact, duration, blurb, content, image_urls 
            FROM story_submissions 
            WHERE submission_id = $1 AND tenant_code = $2
            """,
            submission_id, tenant_code
        )
    elif "discussion" in sub_type:
        payload_row = await conn.fetchrow(
            """
            SELECT title, challenges, solutions, author, language, image_urls 
            FROM discussion_submissions 
            WHERE submission_id = $1 AND tenant_code = $2
            """,
            submission_id, tenant_code
        )
    else:
        raise ValueError(f"Unsupported submission type: {sub_type}")

    if not payload_row:
        raise ValueError(f"Payload details not found for {submission_id} under type {sub_type}.")

    return sub_type, dict(payload_row)

