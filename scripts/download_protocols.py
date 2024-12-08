import requests
import os
import time
import argparse
import json
import random
from tqdm import tqdm


class ClinicalTrialsAPI:
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/v2"
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 Clinical Trials Protocol Downloader"
        }

    def search_studies(self, max_studies=1000, is_cancer=True, existing_nct_ids=None):
        """
        Search studies using REST API 2.0.3 with protocol documents
        """
        endpoint = f"{self.base_url}/studies"
        params = {
            "format": "json",
            'query.cond': 'cancer' if is_cancer else 'NOT cancer',
            "pageSize": min(max_studies, 100),  # API limits to 100 per page
            "aggFilters": 'docs:prot'
        }

        studies = []
        next_page_token = None
        existing_nct_ids = set(existing_nct_ids or [])

        while len(studies) < max_studies:
            if next_page_token:
                params['pageToken'] = next_page_token

            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            valid_studies = [
                study for study in data['studies']
                if (study.get('documentSection', {}).get('largeDocumentModule', {}).get('largeDocs') and
                    get_nct_id(study) not in existing_nct_ids)
            ]

            studies.extend(valid_studies)
            print(f"Found {len(valid_studies)} new studies in this batch (Total: {len(studies)})")

            next_page_token = data.get('nextPageToken')
            if not next_page_token or len(valid_studies) == 0:
                break

            time.sleep(1)  # Be nice to the server

        return studies[:max_studies]


def get_nct_id(study_data):
    try:
        return study_data['protocolSection']['identificationModule']['nctId']
    except KeyError:
        return None


def download_protocol(study_data, download_dir, force_download=False):
    try:
        docs = study_data['documentSection']['largeDocumentModule']['largeDocs']
        protocol_doc = next(
            (doc for doc in docs if doc.get('hasProtocol')),
            None
        )

        if not protocol_doc:
            return False

        nct_id = None
        nct_id = get_nct_id(study_data)
        if not nct_id:
            return False

        filename = protocol_doc['filename']
        sub_dir = nct_id[-2:]
        url = f"https://cdn.clinicaltrials.gov/large-docs/{sub_dir}/{nct_id}/{filename}"
        save_filename = f"{nct_id}_protocol.pdf"
        full_path = os.path.join(download_dir, save_filename)

        if os.path.exists(full_path) and not force_download:
            print(f"Skipping {nct_id} - file already exists")
            return True

        response = requests.get(url)
        response.raise_for_status()

        with open(full_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded {nct_id}")
        return True

    except Exception as e:
        print(f"Error downloading {nct_id if nct_id is not None else 'unknown'}: {str(e)}")
        return False


def get_existing_nct_ids(base_dir, split_dir):
    existing_ids = set()
    if os.path.exists(split_dir):
        for filename in os.listdir(split_dir):
            if filename.endswith('_protocol.pdf'):
                nct_id = filename.split('_')[0]
                existing_ids.add(nct_id)
    return existing_ids


def get_all_existing_nct_ids(directories):
    """Get NCT IDs from multiple directories."""
    all_nct_ids = set()
    for directory in directories:
        if os.path.exists(directory):
            for study_type in ['cancer', 'non_cancer']:
                type_dir = os.path.join(directory, study_type)
                if os.path.exists(type_dir):
                    # Check both split directories (train/val/test) and flat directories
                    for split in ['train', 'val', 'test', '']:
                        split_dir = os.path.join(type_dir, split)
                        if os.path.exists(split_dir):
                            all_nct_ids.update(get_existing_nct_ids(directory, split_dir))
    return all_nct_ids


def process_studies(api, study_type, target_size, base_dir, split_ratios=(0.7, 0.15, 0.15), force_download=False, existing_nct_ids=None, eval_mode=False):
    """Process and download studies with train/val/test split"""
    if not eval_mode and sum(split_ratios) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Create directories
    if eval_mode:
        # In eval mode, just create the study type directory
        study_dir = os.path.join(base_dir, study_type)
        os.makedirs(study_dir, exist_ok=True)
        split_dirs = {'eval': study_dir}
        splits = ['eval']
    else:
        # Create train/val/test splits
        splits = ['train', 'val', 'test']
        split_dirs = {}
        for split in splits:
            split_dirs[split] = os.path.join(base_dir, study_type, split)
            os.makedirs(split_dirs[split], exist_ok=True)

    # Get existing NCT IDs
    existing_ids = {split: get_existing_nct_ids(base_dir, split_dirs[split]) for split in splits}
    all_existing = set().union(*existing_ids.values())
    
    print(f"Found existing {study_type} protocols:")
    for split in splits:
        print(f"- {split}: {len(existing_ids[split])} protocols")

    # Calculate how many more studies we need
    total_existing = len(all_existing)
    needed_studies = max(0, target_size - total_existing)

    if needed_studies > 0:
        print(f"\nNeed {needed_studies} more {study_type} studies to reach target of {target_size}")
        studies = api.search_studies(needed_studies, study_type == "cancer", existing_nct_ids or all_existing)
        
        if eval_mode:
            # In eval mode, put all studies in the study type directory
            split_studies = {'eval': studies}
        else:
            # Split new studies for train/val/test
            random.seed(42)
            random.shuffle(studies)
            
            # Calculate split sizes for new studies
            train_size = int(len(studies) * split_ratios[0])
            val_size = int(len(studies) * split_ratios[1])
            
            split_studies = {
                'train': studies[:train_size],
                'val': studies[train_size:train_size + val_size],
                'test': studies[train_size + val_size:]
            }

        # Download protocols for each split
        success_counts = {split: 0 for split in splits}
        for split, studies_subset in split_studies.items():
            print(f"\nDownloading {len(studies_subset)} protocols for {split} split...")
            for study in tqdm(studies_subset):
                if download_protocol(study, split_dirs[split], force_download):
                    success_counts[split] += 1

        return success_counts
    else:
        print("No additional studies needed")
        return {split: 0 for split in splits}


def main():
    parser = argparse.ArgumentParser(description='Download ClinicalTrials.gov protocol documents')
    parser.add_argument('--target-size', type=int, default=1500,
                      help='Target number of protocols per type (cancer/non-cancer) (default: 1500)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                      help='Ratio of studies to use for training (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                      help='Ratio of studies to use for validation (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                      help='Ratio of studies to use for testing (default: 0.15)')
    parser.add_argument('--force-download', action='store_true',
                      help='Force re-download of existing files')
    parser.add_argument('--output-dir', type=str, default="protocol_documents",
                      help='Output directory for protocol documents (default: protocol_documents)')
    parser.add_argument('--eval-mode', action='store_true',
                      help='Download evaluation dataset with no overlap with other datasets')
    parser.add_argument('--exclude-dirs', nargs='+', default=[],
                      help='Additional directories to check for existing NCT IDs to exclude')
    args = parser.parse_args()

    split_ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if not args.eval_mode and sum(split_ratios) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)
    api = ClinicalTrialsAPI()

    # Get all existing NCT IDs to exclude
    exclude_dirs = ["protocol_documents"] + args.exclude_dirs
    if args.output_dir not in exclude_dirs:
        exclude_dirs.append(args.output_dir)
    existing_nct_ids = get_all_existing_nct_ids(exclude_dirs)
    if existing_nct_ids:
        print(f"Found {len(existing_nct_ids)} existing NCT IDs to exclude")

    try:
        # Process both types of studies
        for study_type in ['cancer', 'non_cancer']:
            print(f"\nProcessing {study_type} studies (target: {args.target_size})...")
            
            if args.eval_mode:
                # In eval mode, don't create train/val/test splits
                success_counts = process_studies(
                    api, study_type, args.target_size, base_dir, 
                    force_download=args.force_download,
                    existing_nct_ids=existing_nct_ids,
                    eval_mode=True
                )
            else:
                success_counts = process_studies(
                    api, study_type, args.target_size, base_dir, 
                    split_ratios=split_ratios,
                    force_download=args.force_download,
                    existing_nct_ids=existing_nct_ids,
                    eval_mode=False
                )

            if sum(success_counts.values()) > 0:
                print(f"\nNewly downloaded {study_type} protocols:")
                for split, count in success_counts.items():
                    print(f"- {split}: {count} protocols")

        # Print final statistics
        print("\nFinal dataset statistics:")
        for study_type in ['cancer', 'non_cancer']:
            print(f"\n{study_type.title()} protocols:")
            total = 0
            if args.eval_mode:
                splits = ['']  # Just the base directory in eval mode
            else:
                splits = ['train', 'val', 'test']
            
            for split in splits:
                split_dir = os.path.join(base_dir, study_type, split)
                if os.path.exists(split_dir):
                    count = len(get_existing_nct_ids(base_dir, split_dir))
                    total += count
                    split_name = split if split else 'eval'
                    print(f"- {split_name}: {count} protocols")
            print(f"Total: {total} protocols")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()