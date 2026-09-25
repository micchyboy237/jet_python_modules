"""
Main application entry point for jet_notes.
Tracing must be initialized BEFORE any other imports that might create spans.
"""

import logging
import os
import sys

# Set cwd to this file's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add the parent of this file's parent directory to sys.path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables from .env file for development
# Check if running in development mode or if .env exists
if os.path.exists(".env.development"):
    load_dotenv(".env.development")
elif os.path.exists(".env"):
    load_dotenv(".env")

# Configure basic logging to see telemetry init logs
logging.basicConfig(level=logging.INFO)

from jet_notes.telemetry import initialize_tracing

try:
    tracer_provider = initialize_tracing(
        project_name="jet_notes",
        auto_instrument=True,
        batch=True,
    )
except Exception as e:
    logging.error(f"Critical failure in telemetry initialization: {e}")
    # Depending on strictness, you might exit here
    # sys.exit(1)


def main():
    """Main application logic."""
    from jet_notes.services.note_service import NoteService

    service = NoteService()
    print("Creating a test note...")
    note = service.create_note("Test Title", "This is a test content for tracing.")
    print(f"Note created with ID: {note.id}")


if __name__ == "__main__":
    main()
