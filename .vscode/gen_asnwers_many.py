import subprocess

def run_operation(epoch, model_path, model_id):
    """
    Run the operation with the provided epoch, model-path, and model-id in a new Command Prompt window.
    """
    # Construct command to execute in the new terminal
    model_answer_command = f"python gen_model_answer.py --model-path \"{model_path}\" --model-id \"{model_id}\""
    
    # Opening a new Command Prompt window for this command
    start_command = f"start cmd /k {model_answer_command}"
    
    # Running the constructed command
    subprocess.run(start_command, shell=True, cwd="F:\\src\\FastChat\\fastchat\\llm_judge")


# Hardcoded paths for the model checkpoints
model_paths = [
    "F:\\src\\finetuning\\qlora\\out_qlora-20240523143404\\checkpoint-173",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240523143404\\checkpoint-347",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240523143404\\checkpoint-521",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240523143404\\checkpoint-694",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240523143404\\checkpoint-868",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240523143404\\checkpoint-1042",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240523143404\\checkpoint-1215",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240523143404\\checkpoint-1389",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240523143404\\checkpoint-1563",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240523143404\\checkpoint-1736",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240523143404\\checkpoint-1910",
    "F:\\src\\finetuning\\qlora\\out_qlora-20240523143404\\checkpoint-2076"
]

# Iterate through each checkpoint and run the operation in a new Command Prompt
for epoch, model_path in enumerate(model_paths, 1):
    model_id = f"stablelm-2-brief-1_6b_v8_r50_epoch-{epoch:02}"
    
    run_operation(epoch, model_path, model_id)

print("Script execution started. Check the new Command Prompt windows for progress.")
