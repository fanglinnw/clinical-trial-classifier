import requests
import os
import time
import argparse
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
            if not next_page_token:
                break

            time.sleep(1)  # Be nice to the server

        return studies[:max_studies]


def get_nct_id(study_data):
    try:
        return study_data['protocolSection']['identificationModule']['nctId']
    except KeyError:
        return None


def download_protocol(study_data, download_dir, force_download=False):
    nct_id = get_nct_id(study_data)
    if not nct_id:
        tqdm.write(f"Invalid or missing NCT ID")
        return False

    try:
        docs = study_data['documentSection']['largeDocumentModule']['largeDocs']
        protocol_doc = next(
            (doc for doc in docs if doc.get('hasProtocol')),
            None
        )

        if not protocol_doc:
            return False

        filename = protocol_doc['filename']
        sub_dir = nct_id[-2:]
        url = f"https://cdn.clinicaltrials.gov/large-docs/{sub_dir}/{nct_id}/{filename}"
        save_filename = f"{nct_id}_protocol.pdf"
        full_path = os.path.join(download_dir, save_filename)

        if os.path.exists(full_path) and not force_download:
            tqdm.write(f"Skipping {nct_id} - file already exists")
            return True

        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        
        with open(full_path, 'wb') as f, tqdm(
            desc=f"Downloading {nct_id}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            leave=False
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
                
        tqdm.write(f"Successfully downloaded {nct_id}")
        return True

    except requests.exceptions.RequestException as e:
        tqdm.write(f"Error downloading {nct_id}: {str(e)}")
        return False
    except Exception as e:
        tqdm.write(f"An unexpected error occurred while downloading {nct_id}: {str(e)}")
        return False


def get_existing_nct_ids(split_dir):
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
                    all_nct_ids.update(get_existing_nct_ids(type_dir))
    return all_nct_ids


def process_studies(api, study_type, target_size, base_dir, force_download=False, existing_nct_ids=None):
    """Process and download studies"""
    # Create directory
    study_dir = os.path.join(base_dir, study_type)
    os.makedirs(study_dir, exist_ok=True)

    # Get existing NCT IDs from this directory
    current_dir_ids = get_existing_nct_ids(study_dir)
    tqdm.write(f"Found {len(current_dir_ids)} existing {study_type} protocols in {study_dir}")
    
    # Combine with other existing IDs
    all_existing_ids = set(existing_nct_ids or set()) | current_dir_ids
    if existing_nct_ids:
        tqdm.write(f"Total {len(existing_nct_ids)} NCT IDs found across all directories")

    # Calculate how many more studies we need
    needed_studies = max(0, target_size - len(current_dir_ids))

    if needed_studies > 0:
        tqdm.write(f"\nNeed {needed_studies} more {study_type} studies to reach target of {target_size}")
        # Pass all_existing_ids to prevent duplicates across all directories
        studies = api.search_studies(needed_studies, study_type == "cancer", all_existing_ids)
        
        # Download protocols
        success_count = 0
        tqdm.write(f"\nDownloading {len(studies)} protocols...")
        with tqdm(
            studies,
            desc=f"Processing protocols",
            unit="protocol"
        ) as pbar:
            for study in pbar:
                if download_protocol(study, study_dir, force_download):
                    success_count += 1
                pbar.set_postfix({"success": success_count})

        return success_count
    else:
        tqdm.write("No additional studies needed")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Download ClinicalTrials.gov protocol documents')
    parser.add_argument('--train-size', type=int, default=750,
                      help='Target number of protocols per type (cancer/non-cancer) for training (default: 750)')
    parser.add_argument('--test-size', type=int, default=100,
                      help='Target number of protocols per type (cancer/non-cancer) for testing (default: 100)')
    parser.add_argument('--no-test', action='store_true',
                      help='Skip downloading test set')
    parser.add_argument('--force-download', action='store_true',
                      help='Force re-download of existing files')
    parser.add_argument('--train-dir', type=str, default="protocol_documents",
                      help='Output directory for training data (default: protocol_documents)')
    parser.add_argument('--test-dir', type=str, default="protocol_documents_test",
                      help='Output directory for test data (default: protocol_documents_test)')
    parser.add_argument('--exclude-dirs', nargs='+', default=[],
                      help='Additional directories to check for existing NCT IDs to exclude')
    args = parser.parse_args()

    base_dir = args.train_dir
    test_dir = args.test_dir if not args.no_test else None
    
    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    if test_dir:
        os.makedirs(test_dir, exist_ok=True)
    
    api = ClinicalTrialsAPI()

    # Get all existing NCT IDs to exclude from both train and test directories
    exclude_dirs = args.exclude_dirs.copy()
    if args.train_dir not in exclude_dirs:
        exclude_dirs.append(args.train_dir)
    if test_dir and test_dir not in exclude_dirs:
        exclude_dirs.append(test_dir)
    
    print("\nGathering existing NCT IDs...")
    existing_nct_ids = get_all_existing_nct_ids(exclude_dirs)
    if existing_nct_ids:
        print(f"Found {len(existing_nct_ids)} existing NCT IDs to exclude")

    try:
        # Process both types of studies
        total_downloads = 0
        total_test_downloads = 0
        
        # First download training data
        for study_type in ['cancer', 'non_cancer']:
            print(f"\nProcessing {study_type} training studies (target: {args.train_size})...")
            success_count = process_studies(
                api, study_type, args.train_size, base_dir, 
                force_download=args.force_download,
                existing_nct_ids=existing_nct_ids
            )
            total_downloads += success_count
            
            if success_count > 0:
                print(f"\nNewly downloaded {study_type} protocols: {success_count}")
                # Update existing_nct_ids with newly downloaded protocols
                existing_nct_ids.update(get_existing_nct_ids(os.path.join(base_dir, study_type)))
        
        # Then download test data
        if test_dir:
            for study_type in ['cancer', 'non_cancer']:
                print(f"\nProcessing {study_type} test studies (target: {args.test_size})...")
                test_success_count = process_studies(
                    api, study_type, args.test_size, test_dir,
                    force_download=args.force_download,
                    existing_nct_ids=existing_nct_ids
                )
                total_test_downloads += test_success_count
                
                if test_success_count > 0:
                    print(f"\nNewly downloaded {study_type} test protocols: {test_success_count}")
                    # Update existing_nct_ids with newly downloaded test protocols
                    existing_nct_ids.update(get_existing_nct_ids(os.path.join(test_dir, study_type)))

        # Print final statistics
        print("\nFinal dataset statistics:")
        print("\nTraining dataset:")
        for study_type in ['cancer', 'non_cancer']:
            study_dir = os.path.join(base_dir, study_type)
            if os.path.exists(study_dir):
                count = len(os.listdir(study_dir))
                print(f"- {study_type}: {count} protocols")
        
        if test_dir:
            print("\nTest dataset:")
            for study_type in ['cancer', 'non_cancer']:
                study_dir = os.path.join(test_dir, study_type)
                if os.path.exists(study_dir):
                    count = len(os.listdir(study_dir))
                    print(f"- {study_type}: {count} protocols")
        
        if total_downloads > 0 or total_test_downloads > 0:
            print(f"\nTotal new downloads:")
            print(f"- Training: {total_downloads}")
            if test_dir:
                print(f"- Test: {total_test_downloads}")

    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    main()