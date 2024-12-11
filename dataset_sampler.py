import os
import shutil
import random
from pathlib import Path

def create_smaller_dataset(
    source_root: str,
    target_root: str,
    samples_per_class: int = 100,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    # test_ratio will be 1 - train_ratio - val_ratio
):
    # Create target directory structure
    for class_name in ['cancer', 'non_cancer']:
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(target_root, class_name, split), exist_ok=True)
    
    # Calculate split sizes
    train_size = int(samples_per_class * train_ratio)
    val_size = int(samples_per_class * val_ratio)
    test_size = samples_per_class - train_size - val_size
    
    # Process each class
    for class_name in ['cancer', 'non_cancer']:
        # Get all files from all splits
        all_files = []
        source_class_dir = os.path.join(source_root, class_name)
        
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(source_class_dir, split)
            files = [os.path.join(split_path, f) for f in os.listdir(split_path) 
                    if os.path.isfile(os.path.join(split_path, f))]
            all_files.extend(files)
        
        # Randomly sample files
        if len(all_files) < samples_per_class:
            raise ValueError(f"Not enough files in {class_name} directory. "
                           f"Found {len(all_files)}, needed {samples_per_class}")
        
        sampled_files = random.sample(all_files, samples_per_class)
        
        # Split into train/val/test
        train_files = sampled_files[:train_size]
        val_files = sampled_files[train_size:train_size + val_size]
        test_files = sampled_files[train_size + val_size:]
        
        # Copy files to new locations
        for files, split_name in [(train_files, 'train'), 
                                (val_files, 'val'), 
                                (test_files, 'test')]:
            target_dir = os.path.join(target_root, class_name, split_name)
            for file_path in files:
                shutil.copy2(file_path, target_dir)
        
        print(f"{class_name} split sizes:")
        print(f"  Train: {len(train_files)}")
        print(f"  Val: {len(val_files)}")
        print(f"  Test: {len(test_files)}")

if __name__ == "__main__":
    # Example usage
    source_root = "protocol_documents"
    target_root = "protocol_documents_small"
    
    create_smaller_dataset(
        source_root=source_root,
        target_root=target_root,
        samples_per_class=100,
        train_ratio=0.7,
        val_ratio=0.15
    )
