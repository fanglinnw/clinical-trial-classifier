import os
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

def get_file_hash(filepath: str, block_size: int = 65536) -> str:
    """
    Calculate SHA-256 hash of a file's contents.
    
    Args:
        filepath: Path to the file
        block_size: Size of chunks to read (default: 64KB)
    
    Returns:
        str: Hexadecimal digest of the file hash
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()

def count_unique_files(directory: str) -> Dict[str, Set[str]]:
    """
    Count unique files in a directory and its subdirectories based on content.
    
    Args:
        directory: Path to the root directory
    
    Returns:
        Dict mapping file hashes to sets of paths with that hash
    """
    hash_map = defaultdict(set)
    
    # Convert to Path object for better cross-platform compatibility
    root_path = Path(directory)
    
    # Walk through all files in directory and subdirectories
    for filepath in root_path.rglob('*'):
        if filepath.is_file():
            try:
                file_hash = get_file_hash(str(filepath))
                hash_map[file_hash].add(str(filepath))
            except (PermissionError, FileNotFoundError) as e:
                print(f"Error processing {filepath}: {e}")
                
    return hash_map

def print_analysis(hash_map: Dict[str, Set[str]]) -> None:
    """
    Print analysis of unique and duplicate files.
    
    Args:
        hash_map: Dict mapping file hashes to sets of paths
    """
    total_files = sum(len(paths) for paths in hash_map.values())
    unique_files = len(hash_map)
    duplicate_sets = sum(1 for paths in hash_map.values() if len(paths) > 1)
    
    print(f"\nAnalysis Results:")
    print(f"Total files scanned: {total_files}")
    print(f"Unique files (by content): {unique_files}")
    print(f"Sets of duplicate files: {duplicate_sets}")
    
    # Print details of duplicate files
    if duplicate_sets > 0:
        print("\nDuplicate Files:")
        for file_hash, paths in hash_map.items():
            if len(paths) > 1:
                print(f"\nDuplicate set (hash: {file_hash[:8]}...):")
                for path in paths:
                    print(f"  - {path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory")
        sys.exit(1)
    
    print(f"Scanning directory: {directory}")
    hash_map = count_unique_files(directory)
    print_analysis(hash_map)
