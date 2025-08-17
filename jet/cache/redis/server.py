import os
import subprocess
from jet.logger import logger


def generate_redis_files(redis_path, redis_port, generated_dir):
    # Ensure the output directory exists
    os.makedirs(generated_dir, exist_ok=True)

    # Updated file contents
    redis_conf_content = f"""bind 0.0.0.0 ::1
port {redis_port}
unixsocket "{redis_path}/tmp/redis.sock"
protected-mode yes
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
dir "{redis_path}/data"
logfile "{redis_path}/logs/redis.log"
pidfile "{redis_path}/run/redis.pid"
databases 16
appendonly yes
appendfilename "local-search-appendonly.aof"
tcp-backlog 511
maxclients 10000
"""

    start_sh_content = f"""redis-server redis.conf &
echo "Started redis server at port {redis_port}"
"""

    stop_sh_content = f"""redis-cli -p {redis_port} shutdown
"""

    setup_sh_content = f"""if [ ! -f .env ]; then
  cp -RPp .env.example .env
  echo "Copied .env.example to .env"
else
  echo ".env file already exists. Skipping copy."
fi
REDIS_PATH="{redis_path}"
REDIS_TMP="$REDIS_PATH/tmp"
REDIS_DATA="$REDIS_PATH/data"
REDIS_RUN="$REDIS_PATH/run"
REDIS_LOGS="$REDIS_PATH/logs"
REDIS_LOG_FILE="$REDIS_LOGS/redis.log"
REDIS_PID_FILE="$REDIS_RUN/redis.pid"
REDIS_IPC_SOCKET="$REDIS_TMP/redis.sock"
mkdir -p "$REDIS_DATA"
mkdir -p "$REDIS_LOGS"
mkdir -p "$REDIS_RUN"
mkdir -p "$REDIS_TMP"
if [ ! -e "$REDIS_LOG_FILE" ]; then
  touch "$REDIS_LOG_FILE"
  echo "Created log file at $REDIS_LOG_FILE"
fi
if [ ! -e "$REDIS_PID_FILE" ]; then
  touch "$REDIS_PID_FILE"
  echo "Created pid file at $REDIS_PID_FILE"
fi
if [ ! -e "$REDIS_IPC_SOCKET" ]; then
  touch "$REDIS_IPC_SOCKET"
  echo "Created temporary IPC socket file at $REDIS_IPC_SOCKET"
fi
chmod -R 777 "$REDIS_PATH"
echo "Redis directories, files and IPC socket are created, and permissions are set."
"""

    # Write files to the generated directory
    with open(os.path.join(generated_dir, "redis.conf"), "w") as f:
        f.write(redis_conf_content)

    with open(os.path.join(generated_dir, "start.sh"), "w") as f:
        f.write(start_sh_content)

    with open(os.path.join(generated_dir, "stop.sh"), "w") as f:
        f.write(stop_sh_content)

    with open(os.path.join(generated_dir, "setup.sh"), "w") as f:
        f.write(setup_sh_content)

    # Set permissions
    for script in ["start.sh", "stop.sh", "setup.sh"]:
        script_path = os.path.join(generated_dir, script)
        os.chmod(script_path, 0o755)
        logger.log("Changed permission to executable:", "755", "Script:", script_path,
                   colors=["GRAY", "DEBUG"])

    logger.log("Files generated in", generated_dir,
               colors=["WHITE", "BRIGHT_SUCCESS"])


def execute_scripts(generated_dir):
    try:
        cwd = os.path.realpath(generated_dir)
        if os.path.exists(f"{generated_dir}/setup.sh") and os.path.exists(f"{generated_dir}/start.sh"):
            subprocess.run([f"./setup.sh"], cwd=cwd, check=True)
            subprocess.run([f"./start.sh"], cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Script call error: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error executing script: {e}")


if __name__ == "__main__":
    # Example usage
    generated_dir = "generated"
    redis_data_dir = "scraped_urls"
    generate_redis_files(
        redis_path=f"/Users/jethroestrada/redis/{redis_data_dir}",
        redis_port=6379,
        generated_dir=generated_dir
    )
    execute_scripts(generated_dir)
