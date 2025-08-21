import os

# Ensure the environment variable exists before importing modules under test
os.environ.setdefault(
    "DBOS_DATABASE_URL", "postgresql://testuser:testpass@example.com:5432/test_db"
)
