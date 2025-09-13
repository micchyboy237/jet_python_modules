import json
import subprocess
import os
import psycopg2
from typing import Optional, List, Tuple
from jet.logger import logger
from jet.validation import validate_sql
from jet._types import DB_Options, DB_Result, SQLResult

CONNECT_DB_CMD = """
\c {db_name};
""".strip()
DISCONNECT_DB_CMD = """
-- KILL ALL EXISTING CONNECTION FROM ORIGINAL DB (sourcedb)
SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity 
WHERE pg_stat_activity.datname = '{db_name}' AND pid <> pg_backend_pid();
""".strip()

DROP_DB_CMD = "DROP DATABASE IF EXISTS {db_name};"
CREATE_DB_CMD = "CREATE DATABASE {db_name};"
CLONE_DB_CMD = """
-- CLONE DATABASE TO NEW ONE({to_db})
CREATE DATABASE {to_db} WITH TEMPLATE {from_db} OWNER {db_user};
""".strip()
CLONE_DB_CMD = f"""
{DISCONNECT_DB_CMD.format(db_name="{from_db}")}

{CLONE_DB_CMD}
""".strip()


def run_sql_cmd(
    command: str,
    options: DB_Options = {},
    delimeter: str = "\n\n"
) -> SQLResult:
    db_type = options.get("type", os.getenv("DB_TYPE", "postgres"))
    db_host = options.get("host", os.getenv("DB_HOST", "localhost"))
    db_port = options.get("port", os.getenv("DB_PORT", 5432))
    db_user = options.get("user", os.getenv("DB_USER", "default_user"))
    db_password = options.get("password", os.getenv("DB_PASSWORD", ""))
    db_name = options.get("name", os.getenv("DB_NAME", "postgres"))
    db_connect_name = options.get(
        "connect_name", os.getenv("DB_CONNECT_NAME", db_name))

    results: List[DB_Result] = []

    print("\n")
    logger.log("\nDB Options:")
    logger.info(json.dumps({
        "type": db_type,
        "host": db_host,
        "port": db_port,
        "user": db_user,
        "password": db_password,
        "name": db_name,
        "connect_name": db_connect_name,
    }, indent=2))
    logger.log("\nCommand:")
    logger.debug(command)

    logger.log("\nExecuting command:")
    try:
        conn = psycopg2.connect(
            dbname=db_connect_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        conn.autocommit = True

        with conn.cursor() as cursor:
            for cmd_part in command.split(delimeter):
                cmd_part = cmd_part.strip()
                if not cmd_part:
                    continue

                # Validate the command syntax
                validation_result = validate_sql(cmd_part)
                if not validation_result['passed']:
                    message = f"Invalid SQL syntax:\n{
                        validation_result['error']}"
                    results.append({
                        "success": False,
                        "command": cmd_part,
                        "message": message
                    })
                    return {
                        "success": False,
                        "db_name": db_name,
                        "command": cmd_part,
                        "message": message,
                        "results": results,
                    }

                try:
                    # Execute command
                    cursor.execute(cmd_part)
                    results.append({
                        "success": True,
                        "command": cmd_part,
                        "message": "Executed successfully"
                    })
                except psycopg2.Error as e:
                    message = str(e)
                    results.append({
                        "success": False,
                        "command": cmd_part,
                        "message": message
                    })
                    return {
                        "success": False,
                        "db_name": db_name,
                        "command": cmd_part,
                        "message": message,
                        "results": results,
                    }

        combined_message = results[-1]['message']
        combined_command = "\n".join([result['command'] for result in results])
        return {
            "success": True,
            "db_name": db_name,
            "command": combined_command,
            "message": combined_message,
            "results": results,
        }

    except psycopg2.OperationalError as e:
        return {
            "success": False,
            "db_name": db_name,
            "command": command,
            "message": str(e),
            "results": results,
        }
    finally:
        conn.close()


def manage_temp_db(
    db_name: str,
    options: DB_Options
) -> Tuple[str, int, str, Optional[str]]:
    """Creates a temporary database with a unique name and drops it after use."""
    conn = psycopg2.connect(
        dbname="postgres",
        user=options.get("user", os.getenv("DB_USER", "default_user")),
        password=options.get("password", os.getenv("DB_PASSWORD", "")),
        host=options.get("host", os.getenv("DB_HOST", "localhost")),
        port=options.get("port", 5432)
    )
    conn.autocommit = True
    temp_db_name = db_name
    n = 1

    with conn.cursor() as cursor:
        # Find an available temporary database name
        while True:
            temp_db_name = f"{db_name}_tmp{n}"
            cursor.execute(
                f"SELECT 1 FROM pg_database WHERE datname = %s;", (temp_db_name,))
            if cursor.fetchone() is None:
                break
            n += 1

    # Return temporary db name and a function to drop it after use
    return temp_db_name, 0, "Temporary database created successfully"


def clone_db(from_db: str, to_db: str, options: DB_Options = {}) -> SQLResult:
    db_user = options.get("user", os.getenv("DB_USER", "default_user"))
    db_host = options.get("host", os.getenv("DB_HOST", "localhost"))
    db_port = options.get("port", os.getenv("DB_PORT", 5432))

    # Step 1: Create the target database
    create_db_cmd = CREATE_DB_CMD.format(db_name=to_db)
    create_db_result = run_sql_cmd(create_db_cmd, options)
    if not create_db_result["success"]:
        return create_db_result

    # Step 2: Run pg_dump and psql to clone data
    dump_command = [
        "pg_dump", "-U", db_user, "-h", db_host, "-p", str(
            db_port), "-d", from_db
    ]
    restore_command = [
        "psql", "-U", db_user, "-h", db_host, "-p", str(db_port), "-d", to_db
    ]

    try:
        dump_proc = subprocess.Popen(
            dump_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        restore_proc = subprocess.Popen(
            restore_command, stdin=dump_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        dump_proc.stdout.close()
        stdout, stderr = restore_proc.communicate()

        if restore_proc.returncode != 0:
            return {
                "success": False,
                "db_name": to_db,
                "command": f"pg_dump | psql from {from_db} to {to_db}",
                "message": stderr.decode(),
                "results": [],
            }

        return {
            "success": True,
            "db_name": to_db,
            "command": f"pg_dump | psql from {from_db} to {to_db}",
            "message": "Database cloned successfully",
            "results": [],
        }

    except Exception as e:
        return {
            "success": False,
            "db_name": to_db,
            "command": f"pg_dump | psql from {from_db} to {to_db}",
            "message": str(e),
            "results": [],
        }


def drop_db(db_name: str, options: DB_Options = {}) -> SQLResult:
    db_type = options.get("type", os.getenv("DB_TYPE", "postgres"))
    command = "\n\n".join([
        DISCONNECT_DB_CMD.format(db_name=db_name),
        DROP_DB_CMD.format(db_name=db_name),
    ])
    result = run_sql_cmd(command=command, options={
        **options,
        "name": db_name,
        "connect_name": "postgres" if db_type == "postgres" else None,
    })
    logger.info(f"Temporary database '{db_name}' dropped.")
    return result


def reset_db(options: DB_Options = {}, temp: bool = False) -> SQLResult:
    db_user = options.get("user", os.getenv("DB_USER", "default_user"))
    base_db_name = options.get("name", os.getenv("DB_NAME", "default_db"))
    db_name = base_db_name

    disconnect_db_cmd = DISCONNECT_DB_CMD.format(db_name=db_name)
    drop_db_cmd = DROP_DB_CMD.format(db_name=db_name)
    create_db_cmd = CREATE_DB_CMD.format(db_name=db_name)

    if temp:
        temp_db_name, status, message = manage_temp_db(
            db_name, options)
        if status != 0:
            command = "\n".join([
                disconnect_db_cmd,
                drop_db_cmd,
            ])
            return {
                "success": False,
                "db_name": db_name,
                "command": command,
                "message": message,
                "results": [],
            }
        db_name = temp_db_name

    if base_db_name != db_name:
        # Connect to postgres
        connect_db_cmd = CONNECT_DB_CMD.format(
            db_name="postgres")
        run_sql_cmd(connect_db_cmd, options)
        # Clone base_db_name to temp_db_name
        clone_db_cmd = CLONE_DB_CMD.format(
            from_db=base_db_name, to_db=db_name, db_user=db_user)
        run_sql_cmd(clone_db_cmd, options)

    command = "\n".join([
        disconnect_db_cmd,
        drop_db_cmd,
        create_db_cmd,
    ])

    result = run_sql_cmd(command, options, delimeter=";")

    # Drop the temporary database if used
    if temp:
        drop_db(db_name)  # Ensure cleanup after use

    return result


def run_sql(
    command: str,
    options: DB_Options = {},
    temp: bool = False
) -> SQLResult:
    """
    Manages temporary database operations if `temp` is True,
    otherwise executes commands directly on the specified database.
    """
    db_user = options.get("user", os.getenv("DB_USER", "default_user"))
    base_db_name = options.get("name", os.getenv("DB_NAME", "default_db"))
    db_name = base_db_name

    # Connect to base_db_name
    connect_db_cmd = CONNECT_DB_CMD.format(
        db_name=base_db_name)
    run_sql_cmd(connect_db_cmd, options)

    if temp:
        temp_db_name, status, message = manage_temp_db(
            db_name, options)
        if status != 0:
            return status, message, None, []
        db_name = temp_db_name

    if base_db_name != db_name:
        # Connect to postgres
        connect_db_cmd = CONNECT_DB_CMD.format(
            db_name="postgres")
        run_sql_cmd(connect_db_cmd, options)
        # Clone base_db_name to temp_db_name
        clone_db_cmd = CLONE_DB_CMD.format(
            from_db=base_db_name, to_db=db_name, db_user=db_user)
        clone_result = run_sql_cmd(clone_db_cmd, options)
        if clone_result['success']:
            logger.success(clone_result['message'])
        # clone_db(from_db=base_db_name, to_db=db_name)

    result = run_sql_cmd(command, {
        **options,
        "name": db_name,
    })

    # Drop the temporary database if it was created
    if temp:
        drop_db(db_name)  # Ensure cleanup after use

    return result
