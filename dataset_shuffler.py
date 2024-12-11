import os
import random
import shutil
from pathlib import Path

def shuffle_dataset(base_dir, splits=('train', 'val', 'test')):
    """
    Shuffles files between train/val/test splits while maintaining class structure.
    
    Args:
        base_dir (str): Path to the base directory containing cancer/non_cancer folders
        splits (tuple): Names of the split directories (default: train, val, test)
    """
    # Get the class directories (cancer/non_cancer)
    class_dirs = [d for d in os.listdir(base_dir) 
                 if os.path.isdir(os.path.join(base_dir, d))]
    
    for class_name in class_dirs:
        class_path = os.path.join(base_dir, class_name)
        
        # Collect all files from all splits
        all_files = []
        for split in splits:
            split_path = os.path.join(class_path, split)
            if os.path.exists(split_path):
                files = [os.path.join(split_path, f) for f in os.listdir(split_path)
                        if os.path.isfile(os.path.join(split_path, f))]
                all_files.extend(files)
        
        # Shuffle all files
        random.shuffle(all_files)
        
        # Calculate split sizes (maintaining original proportions)
        total_files = len(all_files)
        split_sizes = {}
        start_idx = 0
        
        for split in splits:
            split_path = os.path.join(class_path, split)
            if os.path.exists(split_path):
                original_size = len(os.listdir(split_path))
                split_sizes[split] = original_size
        
        # Create temporary directories for the new splits
        temp_base = os.path.join(base_dir, f"temp_{random.randint(1000, 9999)}")
        os.makedirs(temp_base, exist_ok=True)
        
        # Move files to temporary splits
        start_idx = 0
        for split, size in split_sizes.items():
            temp_split_path = os.path.join(temp_base, split)
            os.makedirs(temp_split_path, exist_ok=True)
            
            for file_path in all_files[start_idx:start_idx + size]:
                file_name = os.path.basename(file_path)
                shutil.copy2(file_path, os.path.join(temp_split_path, file_name))
            
            start_idx += size
        
        # Replace original splits with shuffled ones
        for split in splits:
            original_split_path = os.path.join(class_path, split)
            temp_split_path = os.path.join(temp_base, split)
            
            if os.path.exists(original_split_path):
                shutil.rmtree(original_split_path)
            shutil.copytree(temp_split_path, original_split_path)
        
        # Clean up temporary directory
        shutil.rmtree(temp_base)

if __name__ == "__main__":
    # Example usage
    dataset_path = "/home/arklin/work/code/clinical-trial-classifier/protocol_documents" 
    shuffle_dataset(dataset_path)
