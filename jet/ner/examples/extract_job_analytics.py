import json
import shutil
from pathlib import Path
from typing import Any

from jet.adapters.llama_cpp.config import PHOENIX_BASE_URL
from jet.logger import logger
from jet_telemetry import initialize_telemetry
from shared.data_types.job_analytics import JobAnalytics

PROJECT_NAME = "extract-job-analytics"

# --- Configuration ---
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = OUTPUT_DIR / "config.json"
RESULTS_FILE = OUTPUT_DIR / "results.json"

# --- Sample Data ---
SAMPLE_JOB_TEXT = """
Senior AI Engineer - Remote
Company: TechNova Solutions
Location: United States (Remote)
Type: Full-time

About Us:
TechNova is a leading HealthTech SaaS provider revolutionizing patient data analytics. 
We are looking for an experienced AI Developer to join our backend team.

Responsibilities:
- Build scalable ML pipelines using Python and PyTorch.
- Integrate LLMs using LangChain and OpenAI API into our existing Django backend.
- Deploy models on AWS SageMaker and manage infrastructure via Terraform.
- Collaborate with frontend teams to deliver React-based dashboards.

Requirements:
- 5+ years experience in Backend Development and Data Science.
- Strong proficiency in Python, SQL, and Cloud platforms (AWS/GCP).
- Experience with Vector Databases like Pinecone or Milvus.

Compensation:
Salary range: $120,000 - $160,000 USD per year.
Posted: 2024-05-15
Source: LinkedIn
"""


def save_config(config: dict[str, Any]) -> None:
    """Save execution configuration to JSON."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2, default=str)
    logger.success(f"Configuration saved to {CONFIG_FILE}")


def save_results(data: dict[str, Any]) -> None:
    """Save extraction results to JSON."""
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.success(f"Results saved to {RESULTS_FILE}")


def main() -> None:
    initialize_telemetry(service_name=PROJECT_NAME, endpoint=PHOENIX_BASE_URL)

    from jet.ner.entity_extractor import extract_entities_from_text

    logger.info("Starting JobAnalytics Entity Extraction Demo")

    # 1. Setup Configuration
    config = {
        "model_class": "JobAnalytics",
        "temperature": 0.3,
        "input_length_chars": len(SAMPLE_JOB_TEXT),
        "output_dir": str(OUTPUT_DIR),
    }
    save_config(config)
    logger.debug(f"Input Text Length: {config['input_length_chars']} chars")

    # 2. Extract Entities
    try:
        logger.info("Calling extract_entities_from_text...")
        analytics: JobAnalytics = extract_entities_from_text(
            text=SAMPLE_JOB_TEXT,
            model_class=JobAnalytics,
            temperature=config["temperature"],
        )

        # 3. Process Results
        result_dict = analytics.model_dump(mode="json")

        logger.success("Extraction Successful!")
        logger.info("=" * 40)
        logger.info(f"Company:       {analytics.company_name}")
        logger.info(f"Industry:      {analytics.nature_of_business}")
        logger.info(f"Location:      {analytics.country_code}")
        logger.info(f"Platform:      {analytics.source_platform}")
        logger.info(f"Work Mode:     {analytics.work_mode}")
        logger.info(f"Emp Type:      {analytics.employment_type}")
        logger.info(
            f"Salary:        {analytics.salary_min} - {analytics.salary_max} {analytics.salary_currency}"
        )
        logger.info(f"Tech Stack:    {analytics.technology_stack}")
        logger.info(f"Domains:       {analytics.job_domain}")
        logger.info("=" * 40)

        # 4. Save Output
        output_data = {
            "status": "success",
            "input_preview": SAMPLE_JOB_TEXT[:200] + "...",
            "extracted_analytics": result_dict,
        }
        save_results(output_data)

    except Exception as e:
        logger.error(f"Extraction Failed: {e}")
        error_data = {
            "status": "error",
            "error_message": str(e),
        }
        save_results(error_data)
        raise


if __name__ == "__main__":
    main()
