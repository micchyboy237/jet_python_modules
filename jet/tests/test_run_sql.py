from jet.db import run_sql, reset_db
from jet.logger import logger, log_sql_results

SETUP_COMMANDS = """
-- Step 1: Create the users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Step 2: Create the trigger function to set updated_at to the current timestamp
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Step 3: Create the trigger to call the function before each row update
DO
$$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM pg_trigger 
        WHERE tgname = 'set_update_timestamp'
    ) THEN
        CREATE TRIGGER set_update_timestamp
        BEFORE UPDATE ON users
        FOR EACH ROW
        EXECUTE FUNCTION update_timestamp();
    END IF;
END
$$;
""".strip()

COMMAND = """
SELECT * FROM users;
""".strip()


def test_setup():
    results = run_sql(SETUP_COMMANDS)
    log_sql_results(results)


def test_permanent():
    results = run_sql(COMMAND)
    log_sql_results(results)


def test_temporary():
    reset_results = reset_db()
    base_db_name = reset_results['db_name']
    setup_results = run_sql(SETUP_COMMANDS, options={
        "name": base_db_name,
        "connect_name": base_db_name,
    })
    base_db_name = setup_results['db_name']
    results = run_sql(COMMAND, temp=True)
    log_sql_results(results)


if __name__ == '__main__':
    test_temporary()
