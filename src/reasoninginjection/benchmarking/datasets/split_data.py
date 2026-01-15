# splits data into dev, test, and final sets

# for jsonl files, one entry per line
import random
import time

INPUT_FILE = "ficticious_nq_dataset.jsonl"

def split_data(input_file: str = INPUT_FILE, dev_ratio: float = 0.1, test_ratio: float = 0.5):
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    lines = [line for line in lines if line.strip() != ""]
    
    
    random.seed(int(time.time()))
    random.shuffle(lines)
    
    total = len(lines)
    dev_size = int(total * dev_ratio)
    test_size = int(total * test_ratio)
    
    dev_set = lines[:dev_size]
    test_set = lines[dev_size:dev_size + test_size]
    final_set = lines[dev_size + test_size:]
    
    filename_no_ext = input_file.rsplit('.', 1)[0]
    dev_file = f"{filename_no_ext}_dev.jsonl"
    test_file = f"{filename_no_ext}_test.jsonl"
    final_file = f"{filename_no_ext}_final.jsonl"
    
    with open(dev_file, 'w') as f:
        for line in dev_set:
            f.write(line)
    
    with open(test_file, 'w') as f:
        for line in test_set:
            f.write(line)
    
    with open(final_file, 'w') as f:
        for line in final_set:
            f.write(line)
    
    print(f"Total entries: {total}")
    print(f"Dev set size: {len(dev_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Final set size: {len(final_set)}")
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        INPUT_FILE = sys.argv[1]
    split_data()