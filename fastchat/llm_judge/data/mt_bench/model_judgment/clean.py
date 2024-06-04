import sys
import re
import os

def clear_line():
    sys.stdout.write('\r')
    sys.stdout.flush()

def search_and_delete(file_path, pattern):
    lines_scanned = 0
    lines_matched = 0
    buffer_size = 1000  # Adjust buffer size as needed
    temp_file_path = file_path + '.tmp'

    with open(file_path, 'r') as file, open(temp_file_path, 'w') as temp_file:
        for line in file:
            lines_scanned += 1
            if re.search(pattern, line):
                lines_matched += 1
                
            else:
                temp_file.write(line)

            # Buffer stats updates
            if lines_scanned % buffer_size == 0:
                clear_line()
                sys.stdout.write(f"Lines scanned: {lines_scanned} | Lines matched: {lines_matched}")
                sys.stdout.flush()

        # Final update
        clear_line()
        sys.stdout.write(f"Lines scanned: {lines_scanned} | Lines matched: {lines_matched}")
        sys.stdout.flush()

    print()  # Move to the next line after loop finishes

    # Replace the original file with the temporary file
    os.replace(temp_file_path, file_path)

def main():
    file_path = 'fastchat/llm_judge/data/mt_bench/model_judgment/gpt-4_single.jsonl'  # Replace with your file path
    pattern = r'.*"model":\s*"Meta-Llama-3-8B-Instruct-hf".*'
    search_and_delete(file_path, pattern)

if __name__ == "__main__":
    main()
