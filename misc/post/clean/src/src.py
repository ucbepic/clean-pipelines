import pandas as pd
import json
import re
import hashlib
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Process officer data from JSON and CSV sources')
    parser.add_argument('--json-input', required=True, help='Path to input JSON file containing Fresno data')
    parser.add_argument('--csv-input', required=True, help='Path to input CSV file containing POST data')
    parser.add_argument('--output-dir', required=True, help='Directory for output files')
    return parser.parse_args()


def extract_json(incidents):
    all_officers = []
    
    for incident in incidents:
        case_name = incident['intake_case_name']
        incident_date = incident['INCIDENT_DATE_STR']
        
        try:
            deduped_officers = incident['extracted_officers']['deduped']
            
            for officer in deduped_officers:
                officer_record = {
                    'case_name': case_name,
                    'incident_date': incident_date,
                    'name': officer['name'],
                    'rank': officer['rank'],
                    'agency': officer['agency'],
                    'role': officer['role'],
                }
                all_officers.append(officer_record)
                
        except KeyError as e:
            print(f"Error processing incident {case_name}: {e}")
            continue
    
    df = pd.DataFrame(all_officers)

    df = df[~((df.incident_date.str.contains("None")))]
    return df

def clean_officer_names(df):
    # First, filter out "not provided" entries
    df = df[~df['name'].str.contains('not provided', case=False, na=False)]
    
    # Create a list to store the expanded records
    expanded_records = []
    
    for _, row in df.iterrows():
        names = []
        name_str = str(row['name']).strip()
        
        # Handle different separation cases
        if ',' in name_str:
            # Handle comma-separated lists
            names.extend([name.strip() for name in name_str.split(',')])
        elif ' and ' in name_str.lower():
            # Handle 'and' clauses
            names.extend([name.strip() for name in name_str.lower().split(' and ')])
        elif '/' in name_str:
            # Handle forward slash separation
            names.extend([name.strip() for name in name_str.split('/')])
        else:
            # Single name
            names.append(name_str)
        
        # Create a new record for each name while keeping other fields
        for name in names:
            if name.strip():  # Only include non-empty names
                new_record = row.to_dict()
                new_record['name'] = name.strip()
                expanded_records.append(new_record)
    
    # Create new DataFrame from expanded records
    expanded_df = pd.DataFrame(expanded_records)
    
    # Reset index for clean numbering
    expanded_df = expanded_df.reset_index(drop=True)
    
    return expanded_df

def clean_name_col(df):
    df.loc[:, "name"] = (df.name
                         .str.lower()
                         .str.strip()
                         .fillna("")
                         .str.replace(r"the officer's full name is caudle\.", "caudle", regex=True)
                         .str.replace(r"(.+)?sorry(.+)?", "", regex=True)
                         .str.replace(r" p935", "", regex=False)
    )
    return df[~((df.name == ""))]

def parse_name_components(df):
    def process_single_name(name):
        # Initialize components
        first = middle = last = suffix = ''
        
        # Remove any extra whitespace
        name = ' '.join(name.split())
        
        # Extract suffix if present
        suffix_pattern = r'\s+(Jr\.?|Sr\.?|III|II|IV)$'
        suffix_match = re.search(suffix_pattern, name, re.IGNORECASE)
        if suffix_match:
            suffix = suffix_match.group(0).strip()
            # Standardize suffix format
            suffix = suffix.replace('.', '')  # Remove periods
            name = name[:suffix_match.start()]
        
        # Split the remaining name
        parts = name.split()
        
        if len(parts) == 1:
            # Single name (treat as last name)
            last = parts[0]
            
        elif len(parts) == 2:
            # First and last name only
            first, last = parts
            
        else:
            # Handle cases with middle names or initials
            first = parts[0]
            last = parts[-1]
            middle = ' '.join(parts[1:-1])
        
        # Handle special cases
        
        # Case 1: Initial with period (e.g., 'F. Lopez')
        if len(first) == 1 or (len(first) == 2 and first.endswith('.')):
            first = first.rstrip('.')
            
        # Case 2: Compound last names with hyphens or apostrophes
        if '-' in last or "'" in last or last.startswith('O'):
            last = last
            
        # Case 3: Single letter first name with no last name (e.g., 'Rudy C')
        if len(parts) == 2 and len(parts[1]) == 1:
            first = parts[0]
            middle = parts[1].rstrip('.')
            last = ''
            
        return pd.Series({
            'first_name': first.strip(),
            'middle_name': middle.strip(),
            'last_name': last.strip(),
            'suffix': suffix.strip()
        })
    
    # Create new columns with the parsed components
    name_components = df['name'].apply(process_single_name)
    result_df = pd.concat([df, name_components], axis=1)
    return result_df

def clean_name_components(df):
    df.loc[:, "first_name"] = df.first_name.str.replace(r"\.", "", regex=True)
    df.loc[:, "middle_name"] = df.middle_name.str.replace(r"\.", "", regex=True)
    df.loc[:, "last_name"] = df.last_name.str.replace(r"\.", "", regex=True)
    df.loc[:, "suffix"] = df.suffix.str.replace(r"\.", "", regex=True)
    return df 

def drop_bad_rows(df):
    df = df.fillna("")
    
    # Create mask for rows to keep:
    # - Either first_name is not empty, OR
    # - last_name has more than 3 words
    mask = (df['first_name'] != "") | (df['last_name'].str.count(' ') >= 3)
    
    # Return filtered dataframe
    return df[mask]

def generate_uid(df):
    # Fill NaN values with empty strings to avoid issues with concatenation
    columns_for_uid = ['first_name', 'middle_name', 'last_name', 'suffix', 'case_name']
    for col in columns_for_uid:
        df[col] = df[col].fillna('')
    
    # Create a combined string of all fields
    combined_values = df[columns_for_uid].apply(lambda x: ''.join(x).lower(), axis=1)
    
    # Generate SHA-256 hash for each combined string
    df['uid'] = combined_values.apply(lambda x: hashlib.sha1(x.encode()).hexdigest())
    return df

def read_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    df = extract_json(data)

    df = (df
          .pipe(clean_officer_names)
          .pipe(clean_name_col)
          .pipe(parse_name_components)
          .pipe(clean_name_components)
          .pipe(drop_bad_rows)
          .pipe(generate_uid)
    )

    df = df.rename(columns={"agency": "agency_name"})

    df = df[["first_name", "middle_name", "last_name", "suffix", "agency_name", "incident_date", "uid"]]

    df = df.fillna("")
    return df

def filter_for_fresno_agencies(df):
    # List of target agencies
    target_agencies = [
        "Fresno", "Selma", "Parlier", "Clovis", "Reedly", "Sanger", "Kerman",
        "Kingsburg", "Coalinga", "Firebaugh", "Calwa", "Friant", "Mendota",
        "Fowler", "Shaver Lake", "San Joaquin", "Orange Cove", "Del Ray",
        "Yokuts Valley", "Auberry", "Huron", "Caruthers", "Riverdale", "Laton",
        "Big Creek", "Tranquility", "Biola", "Raisin City", "Easton", "Three Rocks",
        "Cantou Creek", "Lanare", "Minkler", "Mayfair", "Malaga", "Bowles",
        "Monmouth", "West Park"
    ]
    
    # Create pattern for matching any of the target agencies
    pattern = '|'.join(target_agencies)
    
    # Find person_nbr values who have worked at any target agency
    target_persons = df[df['agency_name'].str.contains(pattern, case=False, na=False)]['person_nbr'].unique()
    df = df[df['person_nbr'].isin(target_persons)]

    return df


def clean_post(df):
    df = df[["first_name", "middle_name", "last_name", "suffix", "agency_name", "start_date", "end_date", "person_nbr"]]
    
    string_columns = ["first_name", "middle_name", "last_name", "suffix", "agency_name"]
    for col in string_columns:
        df[col] = df[col].str.lower()
    
    return df

def read_post(csv_path):
    df = pd.read_csv(csv_path)

    df = (df
          .pipe(clean_post)
          .pipe(filter_for_fresno_agencies)
          )

    df = df[["first_name", "middle_name", "last_name", "suffix", "agency_name", "start_date", "end_date", "person_nbr"]]
    df = df.fillna("")
    return df 

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process POST data
    df_post = read_post(args.csv_input)
    post_output_path = output_dir / "df_post.csv"
    df_post.to_csv(post_output_path, index=False)
    
    # Process JSON data
    df_json = read_json(args.json_input)
    json_output_path = output_dir / "df_cpdp.csv"
    df_json.to_csv(json_output_path, index=False)

if __name__ == "__main__":
    main()