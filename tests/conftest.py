import os

# Ensure the environment variable exists before importing modules under test
# Set PostgreSQL database URL with full credentials
os.environ.setdefault(
    "DBOS_DATABASE_URL", "postgresql://dbuser:dbpass@localhost:5432/test_db"
)
