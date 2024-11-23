# from jet.db import reset_db, run_sql
from jet.validation import validate_sql
from jet.logger import log_sql_results


def test_correct_syntax():
    sql = """
-- Step 1: Create the users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
""".strip()
    validate_sql(sql)


def test_incorrect_syntax():
    sql = """
-- Step 1: Create the users table
CREAT TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
""".strip()
    validate_sql(sql)


if __name__ == '__main__':
    test_correct_syntax()
    test_incorrect_syntax()
