import argparse
import os
import boto3
import time
import csv
import logging
from PyPDF2 import PdfReader, PdfWriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration is supplied via environment variables so this script can be
# released publicly without leaking the original private S3 bucket name.
AWS_REGION = os.environ.get('PRAP_TABLE_EXTRACTION_AWS_REGION', 'us-west-2')
bucket_name = os.environ.get('PRAP_TABLE_EXTRACTION_S3_BUCKET', '')

textract = boto3.client('textract', region_name=AWS_REGION)
s3 = boto3.client('s3', region_name=AWS_REGION)

def upload_pdf_to_s3(pdf_file_path, bucket_name):
    file_name = os.path.basename(pdf_file_path)
    s3.upload_file(pdf_file_path, bucket_name, file_name)
    logger.info(f"Uploaded {file_name} to S3 bucket {bucket_name}")
    return file_name

def extract_tables_from_page(pdf_page_path, output_dir, page_num):
    # Upload the PDF page to S3 and get the file name
    file_name = upload_pdf_to_s3(pdf_page_path, bucket_name)

    # Start the Textract job
    response = textract.start_document_analysis(
        DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': file_name}},
        FeatureTypes=['TABLES']
    )

    # Get the Job ID to track the progress
    job_id = response['JobId']
    logger.info(f"Started processing page {page_num}, Job ID: {job_id}")

    status = "IN_PROGRESS"
    while status == "IN_PROGRESS":
        time.sleep(5)  # Adjust the sleep time if necessary
        response = textract.get_document_analysis(JobId=job_id)
        status = response['JobStatus']
        logger.info(f"Job status for page {page_num}: {status}")

    if status == "SUCCEEDED":
        logger.info(f"Processing completed for page {page_num}")
        tables = []

        # Iterate over the blocks to extract tables
        for block in response['Blocks']:
            if block['BlockType'] == 'TABLE':
                table_data = extract_table_data(block, response['Blocks'])
                tables.append(table_data)

        output_file_path = os.path.join(output_dir, f"page_{page_num}_table.csv")
        save_table_to_csv(tables, output_file_path)
        logger.info(f"Saved extracted tables for page {page_num} to: {output_file_path}")
    else:
        logger.error(f"Failed to process page {page_num}, status: {status}")

def extract_table_data(table_block, blocks):
    # Extract table data from a table block
    table = {}
    for relationship in table_block.get('Relationships', []):
        if relationship['Type'] == 'CHILD':
            for id in relationship['Ids']:
                try:
                    cell = next(block for block in blocks if block['Id'] == id)
                    if cell['BlockType'] == 'CELL':
                        row_index = cell.get('RowIndex', 0)
                        col_index = cell.get('ColumnIndex', 0)
                        text = extract_text_from_cell(cell, blocks)
                        if row_index not in table:
                            table[row_index] = {}
                        table[row_index][col_index] = text
                except StopIteration:
                    # If we can't find the block, just continue to the next one
                    logger.warning(f"Could not find block with ID {id}")
                    continue
    return table

def extract_text_from_cell(cell_block, blocks):
    # Extract the text from a table cell block
    text = ''
    for relationship in cell_block.get('Relationships', []):
        if relationship['Type'] == 'CHILD':
            for id in relationship['Ids']:
                try:
                    word = next(block for block in blocks if block['Id'] == id)
                    if word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
                except StopIteration:
                    logger.warning(f"Could not find word block with ID {id}")
                    continue
    return text.strip()

def save_table_to_csv(tables, output_file_path):
    with open(output_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        for table in tables:
            # Sort the rows by their index
            for row_index in sorted(table.keys()):
                row = table[row_index]
                # Sort columns by their index and write the row to the CSV
                csv_writer.writerow([row.get(col_index, '') for col_index in sorted(row.keys())])

def split_pdf_by_page(pdf_file_path, temp_dir):
    reader = PdfReader(pdf_file_path)
    page_paths = []

    for page_num in range(len(reader.pages)):
        writer = PdfWriter()
        writer.add_page(reader.pages[page_num])
        page_path = os.path.join(temp_dir, f"page_{page_num + 1}.pdf")
        with open(page_path, 'wb') as page_file:
            writer.write(page_file)
        page_paths.append(page_path)

    logger.info(f"Split PDF into {len(page_paths)} pages")
    return page_paths

def process_all_pdfs(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Create a temporary directory for storing individual PDF pages
    temp_dir = os.path.join(output_dir, 'temp_pages')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        logger.info(f"Created temporary directory: {temp_dir}")

    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            pdf_file_path = os.path.join(input_dir, filename)
            pdf_name = os.path.splitext(filename)[0]
            
            logger.info(f"Processing PDF: {filename}")
            
            pdf_output_dir = os.path.join(output_dir, pdf_name)
            if not os.path.exists(pdf_output_dir):
                os.makedirs(pdf_output_dir)
                logger.info(f"Created output directory for PDF: {pdf_output_dir}")
            
            page_paths = split_pdf_by_page(pdf_file_path, temp_dir)

            # Process each page individually
            for page_num, page_path in enumerate(page_paths, start=1):
                extract_tables_from_page(page_path, pdf_output_dir, page_num)

            for page_path in page_paths:
                os.remove(page_path)
            logger.info(f"Cleaned up temporary page files for {filename}")

    os.rmdir(temp_dir)
    logger.info(f"Removed temporary directory: {temp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract tables from PDFs using AWS Textract."
    )
    parser.add_argument(
        "--input-dir",
        default="../data/input",
        help="Directory containing input PDF files (default: ../data/input).",
    )
    parser.add_argument(
        "--output-dir",
        default="../data/output",
        help="Directory where extracted-table CSVs are written (default: ../data/output).",
    )
    args = parser.parse_args()

    if not bucket_name:
        raise RuntimeError(
            "Environment variable PRAP_TABLE_EXTRACTION_S3_BUCKET must be set "
            "to the S3 bucket name Textract should use as a staging area."
        )

    logger.info("Starting PDF table extraction process")

    process_all_pdfs(args.input_dir, args.output_dir)

    logger.info("PDF table extraction process completed")