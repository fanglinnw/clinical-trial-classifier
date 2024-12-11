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
                    # Check both split directories (train/val/test) and flat directories
                    for split in ['train', 'val', 'test', '']:
                        split_dir = os.path.join(type_dir, split)
                        if os.path.exists(split_dir):
                            all_nct_ids.update(get_existing_nct_ids(split_dir))
    return all_nct_ids


def process_studies(api, study_type, target_size, base_dir, force_download=False, existing_nct_ids=None):
    """Process and download studies"""
    # Create directory
    study_dir = os.path.join(base_dir, study_type)
    os.makedirs(study_dir, exist_ok=True)

    # Get existing NCT IDs
    existing_ids = get_existing_nct_ids(study_dir)
    tqdm.write(f"Found {len(existing_ids)} existing {study_type} protocols")

    # Calculate how many more studies we need
    needed_studies = max(0, target_size - len(existing_ids))

    if needed_studies > 0:
        tqdm.write(f"\nNeed {needed_studies} more {study_type} studies to reach target of {target_size}")
        studies = api.search_studies(needed_studies, study_type == "cancer", existing_nct_ids or existing_ids)
        
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
    parser.add_argument('--target-size', type=int, default=750,
                      help='Target number of protocols per type (cancer/non-cancer) (default: 750)')
    parser.add_argument('--test-size', type=int, default=None,
                      help='If specified, download this many protocols for each type into a separate test directory')
    parser.add_argument('--force-download', action='store_true',
                      help='Force re-download of existing files')
    parser.add_argument('--output-dir', type=str, default="protocol_documents",
                      help='Output directory for protocol documents (default: protocol_documents)')
    parser.add_argument('--test-dir', type=str, default="protocol_documents_test",
                      help='Output directory for test protocols (default: protocol_documents_test)')
    parser.add_argument('--exclude-dirs', nargs='+', default=[],
                      help='Additional directories to check for existing NCT IDs to exclude')
    args = parser.parse_args()

    base_dir = args.output_dir
    test_dir = args.test_dir if args.test_size else None
    os.makedirs(base_dir, exist_ok=True)
    if test_dir:
        os.makedirs(test_dir, exist_ok=True)
    api = ClinicalTrialsAPI()

    # Get all existing NCT IDs to exclude
    exclude_dirs = ["protocol_documents", "protocol_documents_test"] + args.exclude_dirs
    if args.output_dir not in exclude_dirs:
        exclude_dirs.append(args.output_dir)
    if test_dir and test_dir not in exclude_dirs:
        exclude_dirs.append(test_dir)
    existing_nct_ids = get_all_existing_nct_ids(exclude_dirs)
    if existing_nct_ids:
        print(f"Found {len(existing_nct_ids)} existing NCT IDs to exclude")

    try:
        # Process both types of studies
        total_downloads = 0
        total_test_downloads = 0
        
        for study_type in ['cancer', 'non_cancer']:
            print(f"\nProcessing {study_type} studies (target: {args.target_size})...")
            
            # Download main dataset
            success_count = process_studies(
                api, study_type, args.target_size, base_dir, 
                force_download=args.force_download,
                existing_nct_ids=existing_nct_ids
            )
            total_downloads += success_count

            if success_count > 0:
                print(f"\nNewly downloaded {study_type} protocols: {success_count}")
            
            # Download test dataset if requested
            if args.test_size:
                print(f"\nProcessing {study_type} test studies (target: {args.test_size})...")
                test_success_count = process_studies(
                    api, study_type, args.test_size, test_dir,
                    force_download=args.force_download,
                    existing_nct_ids=existing_nct_ids
                )
                total_test_downloads += test_success_count
                
                if test_success_count > 0:
                    print(f"\nNewly downloaded {study_type} test protocols: {test_success_count}")

        # Print final statistics
        print("\nFinal dataset statistics:")
        print("\nMain dataset:")
        for study_type in ['cancer', 'non_cancer']:
            study_dir = os.path.join(base_dir, study_type)
            count = len(get_existing_nct_ids(study_dir)) if os.path.exists(study_dir) else 0
            print(f"{study_type.title()}: {count} protocols")
        
        if test_dir:
            print("\nTest dataset:")
            for study_type in ['cancer', 'non_cancer']:
                study_dir = os.path.join(test_dir, study_type)
                count = len(get_existing_nct_ids(study_dir)) if os.path.exists(study_dir) else 0
                print(f"{study_type.title()}: {count} protocols")
        
        print(f"\nTotal new downloads: {total_downloads} (main) + {total_test_downloads} (test)")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()