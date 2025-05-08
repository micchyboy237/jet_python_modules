from jet.llm.mlx.base import MLX
from jet.logger import logger
import uuid
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def parallel_stream_generate(model_name, prompts, max_tokens, temp, verbose, worker_verbose, task_id):
    if rank != 0:  # Only worker processes (non-rank 0) execute this
        try:
            mlx = MLX(model_name)
        except Exception as e:
            logger.error(
                f"Rank {rank}: Failed to load model {model_name}: {str(e)}")
            comm.send({"type": "error", "message": str(e),
                      "prompt_id": None, "task_id": task_id}, dest=0)
            return

        for prompt in prompts:
            prompt_id = str(uuid.uuid4())
            try:
                if worker_verbose:
                    logger.info(
                        f"Rank {rank}: Generating for prompt: {prompt}")
                tokens_generated = 0
                for chunk in mlx.stream_chat(prompt, model=model_name, temperature=temp, max_tokens=max_tokens):
                    content = chunk["choices"][0]["message"]["content"]
                    # Approximate token count
                    tokens_generated += len(content.split())
                    # Send each chunk to the main process
                    comm.send({
                        "type": "chunk",
                        "prompt": prompt,
                        "content": content,
                        "prompt_id": prompt_id,
                        "task_id": task_id
                    }, dest=0)
                    if worker_verbose:
                        logger.info(
                            f"Rank {rank}: Prompt ID {prompt_id}: {content}")
                # Send a result marker
                comm.send({
                    "type": "result",
                    "prompt": prompt,
                    "content": "",
                    "prompt_id": prompt_id,
                    "task_id": task_id,
                    "truncated": tokens_generated >= max_tokens  # Indicate if max_tokens was reached
                }, dest=0)
            except Exception as e:
                if worker_verbose:
                    logger.error(
                        f"Rank {rank}: Prompt ID {prompt_id} failed: {str(e)}")
                comm.send({
                    "type": "error",
                    "message": str(e),
                    "prompt_id": prompt_id,
                    "task_id": task_id
                }, dest=0)
    else:
        # Main process (rank 0) should not process prompts
        pass


if __name__ == "__main__":
    import json
    import sys

    logger.debug(json.dumps(sys.argv))
    if len(sys.argv) < 2:
        print("error: Usage: mpirun -np 5 python parallel_stream_script.py <input_json>", flush=True)
        sys.exit(1)
    try:
        input_json = ' '.join(sys.argv[1:]).strip("'")
        input_data = json.loads(input_json)
    except json.JSONDecodeError as e:
        print(f"error: Invalid JSON input: {str(e)}", flush=True)
        sys.exit(1)

    parallel_stream_generate(
        model_name=input_data["model"],
        prompts=input_data["prompts"],
        max_tokens=input_data["max_tokens"],
        temp=input_data["temp"],
        verbose=input_data["verbose"],
        task_id=input_data.get("task_id")
    )
    MPI.Finalize()
