## Dataset 1: Targeted and Random Sampling (N=995)

- **Three-tier stratified sampling design (N=995)**: We employed a three-tier strategy to balance targeted coverage with population representativeness. Tier 1 oversampled 11 agencies identified as problematic based on data quality issues, high case volume, or unique institutional characteristics (n=570). Tier 2 extended sampling to medical examiner and district attorney agencies (excluding Tier 1 overlaps) to assess whether identified issues were agency-specific or systemic within these institutional types, rather than across other strata such as police departments, sheriff's offices, corrections facilities, or regional agencies (n=106). Tier 3 randomly sampled from all other agencies to maintain representativeness (n=319).

- **Tier 1 targeted agencies**: Los Angeles County Sheriff's Department, Los Angeles Police Department, California Department of Corrections & Rehabilitation, Oakland Police Department, Riverside County District Attorney, California Highway Patrol, California State Personnel Board, Riverside County Sheriff's Department, Fresno County District Attorney, Los Angeles Medical Examiner-Coroner, and Los Angeles Civil Service Commission.

- **Sampling rates varied by agency size**: Tier 1 agencies were sampled at rates ranging from 6.7% (LA County Sheriff, 100 of 1,501 cases) to 78.9% (LA Civil Service Commission, 15 of 19 cases), with larger agencies receiving more cases but lower sampling rates to balance representation and efficiency.

- **Agency coverage limitations**: The final sample drew from 147 of 438 total agencies (34%), with Tier 3's random sampling covering 115 of 381 agencies (30%), reflecting the concentration of cases among higher-volume agencies.

## Dataset 2: Random Sampling (N=500)

- Randomly sampled from all cases. 

## Pipeline
1. If there are multiple dates for a given case, choose the date with more citations, if possible, otherwise concatenate dates into a list. 
2. If there are multiple dates for a UOF case, should we drop? Given that this an indicator of a multicase file. This signal does not apply to misconduct. If this  is TRUE, manual review / HIL? 
3. If the date is not in the first 1/4th of the document, flag? 

