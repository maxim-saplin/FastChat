import subprocess
import concurrent.futures
from dotenv import load_dotenv

# Hardcoded list of models
model_list = [
    "stablelm-2-brief-1_6b_v8_r50_epoch-1",
    "stablelm-2-brief-1_6b_v8_r50_epoch-2",
    "stablelm-2-brief-1_6b_v8_r50_epoch-3",
    "stablelm-2-brief-1_6b_v8_r50_epoch-4",
    "stablelm-2-brief-1_6b_v8_r50_epoch-5",
]

# Load environment variables from .env file
load_dotenv()

# Maximum number of workers (processes) to run simultaneously
MAX_WORKERS = 3


def run_operation(model_id, azure_deployment_name):
    """
    Run the operation with the provided model-id and Azure deployment name.
    """
    # Construct command to execute in the new terminal
    command = f"python gen_judgment.py --model-list {model_id} --azure-deployment-name {azure_deployment_name} --parallel 5"

    # Use `cmd /c` to run the command and then close the command prompt
    start_command = f"cmd /c {command}"

    # Running the constructed command
    subprocess.run(
        start_command, shell=True, cwd="${workspaceFolder}/fastchat/llm_judge"
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
                print(f"{model_id} generated an exception: {exc}")
            else:
                print(f"{model_id} has completed.")

    print(
        "Script execution started. Check the new Command Prompt windows for progress."
    )


if __name__ == "__main__":
    main()
