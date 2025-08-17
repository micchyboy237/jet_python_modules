import os
from jet.cache.redis import generate_redis_files, execute_scripts
from jet.executor import run_commands


def setup_and_start_scraped_urls_cache(port=6379):
    redis_base_dir = "/Users/jethroestrada/redis"
    redis_data_dir = os.path.join(redis_base_dir, "scraped_urls")
    generated_dir = os.path.join(redis_data_dir, "src")
    generate_redis_files(
        redis_path=redis_data_dir,
        redis_port=port,
        generated_dir=generated_dir
    )
    # execute_scripts(generated_dir)
    main_run_commands([
        "pwd",
        "sh ./setup.sh",
        "sh ./start.sh",
        f"lsof -i:{port}",
    ], work_dir=generated_dir)


def setup_and_start_scraped_ollama_models_cache(port=3103):
    redis_base_dir = "/Users/jethroestrada/redis"
    redis_data_dir = os.path.join(redis_base_dir, "ollama_models")
    generated_dir = os.path.join(redis_data_dir, "src")
    generate_redis_files(
        redis_path=redis_data_dir,
        redis_port=port,
        generated_dir=generated_dir
    )
    # execute_scripts(generated_dir)
    main_run_commands([
        "pwd",
        "sh ./setup.sh",
        "sh ./start.sh",
        f"lsof -i:{port}",
    ], work_dir=generated_dir)


def main_run_commands(commands: list[str], work_dir=None):
    """Test the run_commands function."""
    from jet.logger import logger

    command_results = {}

    current_command = ""
    current_command_result = ""
    for output in run_commands(commands, work_dir=work_dir):
        if output.startswith("command:"):
            command = output.split("command:")[1].strip()

            if current_command != command:
                # Store current command result
                if current_command_result:
                    command_results[current_command] = current_command_result.strip(
                    )
                    logger.log("Command:", current_command,
                               colors=["GRAY", "INFO"])
                    logger.log("Result:", current_command_result,
                               colors=["GRAY", "SUCCESS"])

                # Reset for next command
                current_command = command
                current_command_result = ""

        elif output.startswith("data:"):
            result_line = output.split("data:")[1].strip()
            current_command_result += result_line + "\n"

    # Store remaining command result
    if current_command_result:
        command_results[current_command] = current_command_result.strip(
        )
        logger.log("Command:", current_command,
                   colors=["GRAY", "INFO"])
        logger.log("Result:", current_command_result,
                   colors=["GRAY", "SUCCESS"])

    return command_results


if __name__ == "__main__":
    # setup_and_start_scraped_urls_cache()
    setup_and_start_scraped_ollama_models_cache()
