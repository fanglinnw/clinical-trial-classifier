import os
from pathlib import Path
import PyPDF2
from datetime import datetime

def clean_pdf_directory(input_dir, min_chars=8000):
    """
    Delete PDFs that cannot be parsed or don't meet minimum character requirements.
    
    Args:
        input_dir (str): Input directory containing PDFs
        min_chars (int): Minimum number of characters required
    """
    # Create a log file
    log_file = os.path.join(input_dir, f'deletion_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    results = {
        'processed': 0,
        'deleted': 0,
        'valid': 0
    }
    
    def log_message(message):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    # Recursively find all PDF files
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                relative_path = os.path.relpath(pdf_path, input_dir)
                results['processed'] += 1
                
                try:
                    # Try to open and read the PDF
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        text = ""
                        
                        # Extract text from each page until we reach min_chars
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                            if len(text) >= min_chars:
                                break
                        
                        # Delete file if we don't get enough text
                        if len(text) < min_chars:
                            os.remove(pdf_path)
                            results['deleted'] += 1
                            log_message(f"Deleted (too short - {len(text)} chars): {relative_path}")
                        else:
                            results['valid'] += 1
                            log_message(f"Valid: {relative_path}")
                            
                except Exception as e:
                    # Delete file if it can't be parsed
                    os.remove(pdf_path)
                    results['deleted'] += 1
                    log_message(f"Deleted (parsing error): {relative_path} - Error: {str(e)}")
    
    # Write summary to log
    summary = f"""
Cleaning Summary:
---------------
Total processed: {results['processed']}
Valid files: {results['valid']}
Deleted files: {results['deleted']}
    """
    log_message(summary)
    
    return results

# Example usage
if __name__ == "__main__":
    input_directory = "protocol_documents"
    
    results = clean_pdf_directory(
        input_dir=input_directory,
        min_chars=8000
    )
    
    print("Cleaning complete!")
    print(f"Processed {results['processed']} files:")
    print(f"- Valid: {results['valid']}")
    print(f"- Deleted: {results['deleted']}")
