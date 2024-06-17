import subprocess
import concurrent.futures


def run_operation(epoch, model_path, model_id):
    """
    Run the operation with the provided epoch, model-path, and model-id in a new Command Prompt window.
    """
    # Construct command to execute in the new terminal
    model_answer_command = f"python gen_model_answer.py --model-path \"{model_path}\" --model-id \"{model_id}\""

    # Use `cmd /c` to run the command and then close the command prompt
    start_command = f"cmd /c {model_answer_command}"

    # Running the constructed command
    subprocess.run(start_command, shell=True,
                   cwd="F:\\src\\FastChat\\fastchat\\llm_judge")


# Hardcoded paths for the model checkpoints
model_paths = [
    "F:\\src\\finetuning\\qlora\\out_qlora-20240611151306\\checkpoint-10191",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240611151306\\checkpoint-20382",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240611151306\\checkpoint-30573",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240611151306\\checkpoint-40765",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240611151306\\checkpoint-50955",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240611151306\\checkpoint-61146",
]

# Maximum number of workers (processes) to run simultaneously
MAX_WORKERS = 4
MODEL_NAME = "stablelm-2-brief-1_6b_v8_r55"
SKIP_N = 4


def main():
    global SKIP_N
    # Using ProcessPoolExecutor to manage the concurrency of subprocesses
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        future_to_operation = {
            executor.submit(run_operation, epoch, model_path, f"{MODEL_NAME}_epoch-{epoch:02}"): model_path
            for epoch, model_path in enumerate(model_paths[SKIP_N:], SKIP_N + 1)
        }

        # As each future completes, print its result
        for future in concurrent.futures.as_completed(future_to_operation):
            model_path = future_to_operation[future]
            try:
                future.result()  # We are not expecting any return value here, just catching exceptions
            except Exception as exc:
                print(f"{model_path} generated an exception: {exc}")
            else:
                print(f"{model_path} has completed.")

    print("Script execution started. Check the new Command Prompt windows for progress.")


if __name__ == "__main__":
    main()
