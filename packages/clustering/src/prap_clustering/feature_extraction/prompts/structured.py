"""
Structured output prompts for converting extraction results to JSON.

These prompts take the human-readable string outputs from the extraction pipeline
and convert them to structured JSON format for downstream clustering.
"""

# Dates Structuring Prompt
DATES_STRUCTURING = """You are a data structuring assistant. Your task is to convert the incident date extraction result into a structured JSON format.

<input>
{{ source_text }}
</input>

<instructions>
1. Read the extraction result above
2. Identify the INCIDENT DATE (not report date, not investigation date)
3. Return a JSON object with the following structure:
   {"incident_date": "YYYY-MM-DD"}
4. If no incident date is found or the extraction says "None", return:
   {"incident_date": null}
5. Use ISO-8601 date format (YYYY-MM-DD)
6. Return ONLY the JSON object, no additional text or explanation
</instructions>

<examples>
Example 1:
Input: "INCIDENT DATE: January 26, 2014"
Output: {"incident_date": "2014-01-26"}

Example 2:
Input: "No incident date found in document"
Output: {"incident_date": null}

Example 3:
Input: "VERIFICATION RESULT: CONFIRMED. Incident occurred on 2024-03-15"
Output: {"incident_date": "2024-03-15"}
</examples>

JSON Output:"""


# Case IDs Structuring Prompt
CASE_IDS_STRUCTURING = """You are a data structuring assistant. Your task is to convert the case ID extraction result into a structured JSON format.

<input>
{{ source_text }}
</input>

<instructions>
1. Read the extraction result above
2. Identify ALL case IDs, incident numbers, IA numbers, OIA numbers, or similar identifiers
3. Return a JSON array of objects with the following structure:
   [{"id": "IA2018-0167"}, {"id": "2018-0167"}]
4. Each unique identifier should be a separate object in the array
5. If no case IDs are found, return an empty array: []
6. Preserve the exact format of each ID (don't modify or standardize)
7. Return ONLY the JSON array, no additional text or explanation
</instructions>

<examples>
Example 1:
Input: "PRIMARY CASE ID: IA2018-0167. Also referenced as Case #2018-0167"
Output: [{"id": "IA2018-0167"}, {"id": "2018-0167"}]

Example 2:
Input: "No case identifiers found in document"
Output: []

Example 3:
Input: "VERIFICATION RESULT: CONFIRMED. Case number is OIA-2023-145"
Output: [{"id": "OIA-2023-145"}]

Example 4:
Input: "Multiple references: IA2019-0234, Case 2019-0234, File #19-234"
Output: [{"id": "IA2019-0234"}, {"id": "2019-0234"}, {"id": "19-234"}]
</examples>

JSON Output:"""


# Subject Names Structuring Prompt
SUBJECT_NAMES_STRUCTURING = """You are a data structuring assistant. Your task is to convert the subject/civilian name extraction result into a structured JSON format.

<input>
{{ source_text }}
</input>

<instructions>
1. Read the extraction result above
2. Identify ALL subjects/civilians (NOT law enforcement personnel)
3. For each person, extract:
   - name: Full name of the person
   - subject_type: One of "suspect", "victim", "witness", "complainant", "other"
4. Return a JSON array of objects with this structure:
   [{"name": "Kevin Bushnell", "subject_type": "suspect"}, {"name": "John Doe", "subject_type": "witness"}]
5. If a person has multiple roles (e.g., suspect AND victim), include them once with the most relevant type
6. If no subjects/civilians are found, return an empty array: []
7. Return ONLY the JSON array, no additional text or explanation
</instructions>

<examples>
Example 1:
Input: "SUBJECTS: Kevin Bushnell - suspect involved in shooting; Kevin Banks - suspect who fled scene"
Output: [{"name": "Kevin Bushnell", "subject_type": "suspect"}, {"name": "Kevin Banks", "subject_type": "suspect"}]

Example 2:
Input: "WITNESSES: John Smith - witness interviewed at scene. VICTIMS: Jane Doe - assault victim"
Output: [{"name": "John Smith", "subject_type": "witness"}, {"name": "Jane Doe", "subject_type": "victim"}]

Example 3:
Input: "No civilian subjects identified in document"
Output: []

Example 4:
Input: "COMPLAINANT: Mary Johnson filed complaint. SUSPECT: Robert Wilson - subject of investigation"
Output: [{"name": "Mary Johnson", "subject_type": "complainant"}, {"name": "Robert Wilson", "subject_type": "suspect"}]
</examples>

JSON Output:"""


# Officer Names Structuring Prompt
OFFICER_NAMES_STRUCTURING = """You are a data structuring assistant. Your task is to convert the officer name extraction result into a structured JSON format.

<input>
{{ source_text }}
</input>

<instructions>
1. Read the extraction result above
2. Identify ALL law enforcement personnel (officers, detectives, sergeants, etc.)
3. For each officer, extract:
   - name: Full name including rank/title if present (e.g., "Officer Butera", "Detective Smith")
   - context: Brief description of their role or actions in the incident (e.g., "responded to scene", "conducted interview", "discharged firearm")
4. Return a JSON array of objects with this structure:
   [{"name": "Officer Butera", "context": "responded to scene"}, {"name": "Sgt. Rodriguez", "context": "supervised investigation"}]
5. Keep context concise (5-10 words) but informative about the officer's role
6. If no context is available for an officer, use an empty string: ""
7. If no officers are found, return an empty array: []
8. Return ONLY the JSON array, no additional text or explanation
</instructions>

<examples>
Example 1:
Input: "OFFICERS: Officer Butera, Badge #1234, responded to scene. Sgt. Rodriguez supervised investigation"
Output: [{"name": "Officer Butera", "context": "responded to scene"}, {"name": "Sgt. Rodriguez", "context": "supervised investigation"}]

Example 2:
Input: "No law enforcement personnel identified in document"
Output: []

Example 3:
Input: "VERIFICATION RESULT: CONFIRMED. Detective James Wilson conducted interview. Officer Sarah Chen provided backup"
Output: [{"name": "Detective James Wilson", "context": "conducted interview"}, {"name": "Officer Sarah Chen", "context": "provided backup"}]

Example 4:
Input: "Responding officers: M. Johnson #5678 discharged firearm during confrontation, Det. K. Brown #9012 arrived as backup and secured scene"
Output: [{"name": "M. Johnson", "context": "discharged firearm during confrontation"}, {"name": "Det. K. Brown", "context": "arrived as backup and secured scene"}]

Example 5:
Input: "Officer-involved shooting. Officer T. Martinez #4321 discharged weapon. Sgt. L. Kim reviewed use of force. Det. R. Patel investigated incident"
Output: [{"name": "Officer T. Martinez", "context": "discharged weapon"}, {"name": "Sgt. L. Kim", "context": "reviewed use of force"}, {"name": "Det. R. Patel", "context": "investigated incident"}]
</examples>

JSON Output:"""


# Export all prompts as a dictionary
PROMPTS = {
    "dates": DATES_STRUCTURING,
    "case_ids": CASE_IDS_STRUCTURING,
    "subject_names": SUBJECT_NAMES_STRUCTURING,
    "officer_names": OFFICER_NAMES_STRUCTURING,
}
