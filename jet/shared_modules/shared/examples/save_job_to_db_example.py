# examples/save_job_to_db_example.py
"""
Demonstration of how to use save_job_to_db()

Assumptions:
- You already have a running PostgreSQL + pgvector instance
- DEFAULT_JOBS_DB_NAME = "jobs_db1" (or whatever you set)
- JobData dataclass / TypedDict is available from your scrapers module
"""

from datetime import datetime
from typing import cast

from jet.db.postgres.pgvector import PgVectorClient
from jet.logger import logger
from jet.shared_modules.shared.job_helpers import (
    DEFAULT_JOBS_DB_NAME,
    save_job_to_db,
)
from shared.data_types.job import JobData


def create_sample_job() -> JobData:
    """Helper to create a realistic-looking job object for testing."""
    return cast(
        JobData,
        {
            "id": "job-linkedin-ABC123XYZ789",
            "title": "Senior React Native Engineer – Remote (APAC)",
            "company": "TechScale Innovations",
            "details": (
                "We are looking for an experienced React Native developer with strong "
                "TypeScript skills to join our mobile team. Experience with offline-first "
                "apps, Redux Toolkit, and Expo is a big plus.\n\n"
                "Responsibilities:\n"
                "- Lead mobile architecture decisions\n"
                "- Mentor junior developers\n"
                "- Build high-performance cross-platform apps\n\n"
                "Requirements:\n"
                "- 5+ years React Native\n"
                "- Published apps in stores\n"
                "- Strong CI/CD knowledge"
            ),
            "posted_date": datetime(2026, 2, 10, 14, 30),
            # Optional fields many JobData implementations support:
            "url": "https://www.linkedin.com/jobs/view/1234567890",
            "location": "Remote – Singapore / Malaysia preferred",
            "salary": "$120k–$160k USD + equity",
            "tags": ["react-native", "typescript", "mobile", "remote"],
        },
    )


def main():
    # ────────────────────────────────────────────────
    # Option A: Let save_job_to_db create its own client
    # ────────────────────────────────────────────────
    logger.info("Example A: using auto-created PgVectorClient")

    sample_job = create_sample_job()

    saved_job = save_job_to_db(sample_job)

    logger.success(
        f"Job saved/updated:\n"
        f"  ID          : {saved_job['id']}\n"
        f"  Title       : {saved_job['title']}\n"
        f"  Company     : {saved_job['company']}\n"
        f"  Posted      : {saved_job['posted_date']}\n"
        f"  Details len : {len(saved_job['details'])} chars"
    )

    # ────────────────────────────────────────────────
    # Option B: Reuse an existing client (more efficient in loops)
    # ────────────────────────────────────────────────
    logger.info("Example B: reusing existing PgVectorClient")

    db_client = PgVectorClient(dbname=DEFAULT_JOBS_DB_NAME)

    # You can pass the same client many times:
    for i in range(3):
        job = create_sample_job()
        job["id"] = f"test-job-{i:03d}-{job['id'][-8:]}"

        returned_job = save_job_to_db(
            job=job,
            db_client=db_client,
            # embed_model="another-model-name"   # optional override
        )

        logger.info(
            f"Round {i + 1} → returned job title: {returned_job['title']!r} "
            f"(id: {returned_job['id']})"
        )

    db_client.close()  # good practice when you're done


if __name__ == "__main__":
    main()
