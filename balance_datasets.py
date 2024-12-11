import os
import random
import shutil

def count_files(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def balance_datasets(cancer_dir, non_cancer_dir):
    # Count files in both directories
    cancer_count = count_files(cancer_dir)
    non_cancer_count = count_files(non_cancer_dir)
    
    print(f"Current counts:")
    print(f"Cancer protocols: {cancer_count}")
    print(f"Non-cancer protocols: {non_cancer_count}")
    
    if cancer_count >= non_cancer_count:
        print("Datasets are already balanced or non-cancer has fewer files.")
        return
    
    # Calculate how many files to remove
    files_to_remove = non_cancer_count - cancer_count
    print(f"\nNeed to remove {files_to_remove} files from non-cancer directory")
    
    # Get list of all files in non-cancer directory
    non_cancer_files = [f for f in os.listdir(non_cancer_dir) 
                       if os.path.isfile(os.path.join(non_cancer_dir, f))]
    
    # Randomly select files to remove
    files_to_delete = random.sample(non_cancer_files, files_to_remove)
    
    # Create a backup directory
    backup_dir = os.path.join(os.path.dirname(non_cancer_dir), "non_cancer_backup")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Move files to backup instead of deleting
    print("\nMoving excess files to backup directory...")
    for file in files_to_delete:
        src = os.path.join(non_cancer_dir, file)
        dst = os.path.join(backup_dir, file)
        shutil.move(src, dst)
    
    print(f"\nMoved {len(files_to_delete)} files to {backup_dir}")
    print(f"New counts:")
    print(f"Cancer protocols: {count_files(cancer_dir)}")
    print(f"Non-cancer protocols: {count_files(non_cancer_dir)}")

if __name__ == "__main__":
    cancer_dir = "protocol_documents/cancer"
    non_cancer_dir = "protocol_documents/non_cancer"
    balance_datasets(cancer_dir, non_cancer_dir)
