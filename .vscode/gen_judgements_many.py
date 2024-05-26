import subprocess
import concurrent.futures
from dotenv import load_dotenv

# Hardcoded list of models
model_list = [f"stablelm-2-brief-1_6b_v8_r51_epoch-{i:02}" for i in range(1, 13)]

# Load environment variables from .env file
load_dotenv()

# Maximum number of workers (processes) to run simultaneously
MAX_WORKERS = 3


def run_operation(model_id, azure_deployment_name):
    """
    Run the operation with the provided model-id and Azure deployment name.
    """
    command = f"python3.11 gen_judgment.py -y --model-list {model_id} --azure-deployment-name {azure_deployment_name} --parallel 5"

    # Running the constructed command
    subprocess.run(
        command, shell=True, cwd="/private/var/user/src/FastChat/fastchat/llm_judge"
    )


def main():
    azure_deployment_name = "gpt-4-0125-preview"

    # Using ProcessPoolExecutor to manage the concurrency of subprocesses
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_operation = {
            executor.submit(run_operation, model_id, azure_deployment_name): model_id
            for model_id in model_list
        }

        # As each future completes, print its result
        for future in concurrent.futures.as_completed(future_to_operation):
            model_id = future_to_operation[future]
            try:
                future.result()  # We are not expecting any return value here, just catching exceptions
            except Exception as exc:
                print(f"\033[91m{model_id} generated an exception: {exc}\033[0m")  # Red text for errors
            else:
                print(f"\033[92m{model_id} has completed.\033[0m")  # Green text for successful completion

    print("Script execution started. Check the terminal for progress.")


if __name__ == "__main__":
    main()
