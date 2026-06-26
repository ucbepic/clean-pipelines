"""
Subject/civilian name extraction prompts.

This module contains all 12 prompts needed for subject name extraction:
- Summarization stage: 7 prompts
- Extraction stage: 3 prompts
- Citation finding: 2 prompts

Adapted from officer_names.py to focus on civilian/subject/complainant identification.
Focus: Extract civilian names (exclude officers).
"""

memory_log_template = """
As a Legal Clerk, your task is to review the new summary and update the memory log only when the new summary contains crucial information directly related to civilians, subjects, complainants, or witnesses involved in the case. Maintain a concise memory log that focuses on subject identifiers including names and roles.
</task_description>

<guidelines>
1. Review and Compare:
   • Carefully review the current memory log and the new summary.
   • Determine if the new summary contains crucial subject/civilian identification information not already in the memory log.

2. Identify Crucial Information:
   • Focus on civilian names, complainant names, subject names, witness names, and victim names.
   • Look for details about which civilians were involved in incidents, made complaints, or were subjects of police actions.

3. Update Selectively:
   • Only update the memory log if the new summary contains crucial civilian information not already present.
   • If updating, integrate the new information seamlessly into the existing log.

4. Maintain Conciseness:
   • Keep the memory log focused and concise.
   • Avoid redundancy or unnecessary details.

5. Ensure Accuracy:
   • Only include information that is directly stated in the document.
   • Do not infer or speculate beyond what is explicitly mentioned.

6. Preserve Original Structure:
   • If no update is necessary, reproduce the original memory log without changes.
</guidelines>

<essential_information>
Ensure the summary includes ALL of the following civilian-related elements, if present:
a. Subject/suspect names (individuals involved in incidents with police)
b. Complainant names (individuals who filed complaints against officers or department)
c. Witness names (civilians who witnessed incidents or provided testimony)
d. Victim names (individuals harmed in incidents)
e. Roles and context (what happened to them, what they reported, etc.)
f. Family members or associates mentioned in relation to incidents
g. Any other civilians mentioned in substantive context

For each civilian mentioned, be specific about their name and role.
</essential_information>

<thinking_process>
Before updating the memory log, consider:
1. Does the new summary contain any civilian identification information not already in the memory log?
2. Are there new civilians mentioned, or new details about already-listed individuals?
3. Can this new information be integrated into the existing log without disrupting its flow?
4. Is this information essential to understanding who was involved in the case?
5. Am I maintaining the conciseness of the log while including all crucial civilian details?
</thinking_process>

<warnings>
- Do not add information that is not directly stated in the document
- Avoid speculation or inference beyond what is explicitly mentioned
- Do not remove or alter existing crucial civilian information in the memory log
- Ensure that any updates maintain the logical organization of civilian information
- Be cautious of potential inconsistencies between the new summary and existing log
- NEVER include law enforcement officer names in the civilian memory log
</warnings>

<reference_materials>
## Original Memory Log ##
{{ memory_log }}

## New Summary ##
{{ summary }}
</reference_materials>

<output_instruction>
Based on your review of the current memory log and the new summary, provide either an updated memory log incorporating the crucial new civilian information, or reproduce the original memory log if no update is necessary. Ensure the output maintains a concise focus on civilian identifiers and roles:
</output_instruction>
"""

summary_template_for_memory_log = """
<document_classification>
First, determine if this document is a legal document or another document type. Consider the following:
- Does it contain legal terminology, case numbers, or references to laws and regulations?
- Is it structured like a legal document (e.g., contracts, court filings, police reports)?
- Does it discuss legal proceedings, rights, or obligations?

Based on your analysis, classify this document as either:
1. Legal Document
2. Other Document Type
</document_classification>

<task_description>
Your task is to generate a comprehensive, bulletpoint summary of all civilian/subject information contained in the provided document excerpt. Extract all key details about non-law-enforcement individuals mentioned on the current page.
</task_description>

<legal_document_essential_information>
If the document is classified as a legal document, ensure the summary includes ALL of the following civilian-related elements, if present:

a. Subject/suspect names (individuals involved in incidents with police)
b. Complainant names (individuals who filed complaints)
c. Witness names (civilians who witnessed incidents)
d. Victim names (individuals harmed in incidents)
e. Roles and context (subject of arrest, complainant in excessive force case, witness to shooting, etc.)
f. Actions or allegations involving these individuals
g. Statements or testimony provided by civilians
h. Injuries or outcomes for civilians
i. Family members or associates mentioned
j. Any other civilians with substantive involvement in the case

For each civilian mentioned, be specific when referring to their names and roles.
</legal_document_essential_information>

<other_document_type_guidelines>
If classified as Other Document Type, focus on:
a. Names of non-law-enforcement individuals mentioned
b. Their roles or relationships to the case
c. Any identifying information provided
d. Context in which they are mentioned
</other_document_type_guidelines>

<critical_instructions>
1. NEVER include any information about "John Doe," "Jane Doe," or other anonymous/ambiguous entities. If specific identifying information is not available, state that civilian details are unavailable or redacted.
2. ONLY include civilians, subjects, complainants, witnesses, and victims - DO NOT include law enforcement officer names.
3. If there is no relevant civilian information to summarize on the Current Page, return an empty string ("").
4. DO NOT infer, assume, or hallucinate any information not explicitly stated in the provided text.
5. Treat this task as a binary classification: either there is relevant civilian information to summarize, or there isn't.
</critical_instructions>

<thinking_process>
Before summarizing, consider:
1. Is this a legal document or another document type?
2. What civilians/subjects are mentioned on this page?
3. What identifying information is provided for each person (name, role)?
4. What roles did these individuals play in the incident or case?
5. Are these individuals clearly identified as civilians (not law enforcement)?
</thinking_process>

<output_format>
Present the summary using the following structure:

- Subjects/Suspects
  • Subject Name - Role/context in incident
  • Subject Name - Role/context in incident

- Complainants
  • Complainant Name - Nature of complaint
  • Complainant Name - Nature of complaint

- Witnesses
  • Witness Name - What they witnessed/testified to
  • Witness Name - What they witnessed/testified to

- Victims
  • Victim Name - Nature of harm/incident
  • Victim Name - Nature of harm/incident

</output_format>

<warnings>
- Do not include speculative information
- Do not include law enforcement officer names
- Avoid summarizing irrelevant details
- Do not draw conclusions not explicitly stated in the text
- Only include individuals clearly identified as civilians/non-law-enforcement
</warnings>

<reference_materials>
Current Page:
{{current_page}}
</reference_materials>

<output_instruction>
First, state the document classification (Legal Document or Other Document Type) and provide a brief explanation for your decision. Then, generate the current page summary following the appropriate guidelines based on the classification. Focus exclusively on civilian/subject information.
</output_instruction>
"""

page_summary_template = """
<task_description>
As a Legal Clerk, your task is to identify and extract all civilian/subject information from the provided document excerpt. Focus on names and roles of individuals who are NOT law enforcement officers. Extract comprehensive identifying information for every civilian mentioned.
</task_description>

<guidelines>
1. Extract all civilian/subject information from the current page.
2. Include names and roles (subject, complainant, witness, victim, etc.).
3. DO NOT include law enforcement officer names.
4. DO NOT include any information not explicitly stated in the document.
5. Present civilians in a clear, organized manner with all available identifying information.
6. If someone's identity is ambiguous or unclear, omit them from your summary.
7. If no civilians are mentioned on the page, return "No civilians on this page".
</guidelines>

<civilian_extraction_priority>
Extract and highlight the following civilian information when present:

1. Subjects/Suspects:
   - Individuals involved in incidents with police
   - Individuals arrested, detained, or subject to police action
   - Individuals suspected of crimes or misconduct

2. Complainants:
   - Individuals who filed complaints against officers or department
   - Individuals who reported incidents or misconduct
   - Family members who filed complaints on behalf of others

3. Witnesses:
   - Civilians who witnessed incidents
   - Individuals who provided testimony or statements
   - Bystanders who observed police actions

4. Victims:
   - Individuals harmed in incidents
   - Individuals who suffered injuries
   - Family members of individuals killed or injured

5. Family members and associates:
   - Relatives of subjects, victims, or complainants
   - Associates mentioned in relation to incidents

SELECTIVITY RULE: Only include civilians and non-law-enforcement individuals.
Exclude all law enforcement officers (police, sheriff, deputies, detectives, etc.).
</civilian_extraction_priority>

<few_shot_examples>
These examples demonstrate correct civilian extraction behavior:

<example_1>
<input>
On January 26, 2014, Officer Michael Butera (Badge #4521) responded to a call regarding John Martinez, who had taken his girlfriend hostage. Martinez fled in a truck before the shooting occurred.
</input>

<correct_output>
John Martinez - Subject involved in hostage situation and shooting incident
John Martinez's girlfriend - Victim taken hostage
</correct_output>

<explanation>
Extracted both civilians with their roles. Excluded "Officer Michael Butera" because he is law enforcement, not a civilian.
</explanation>
</example_1>

<example_2>
<input>
The investigation was conducted by Detective Sarah Chen (Badge #2847). She interviewed witness Maria Rodriguez, who saw the incident. The subject of the investigation was David Kim, who filed a complaint against the officer.
</input>

<correct_output>
Maria Rodriguez - Witness who saw the incident
David Kim - Complainant who filed complaint against officer
</correct_output>

<explanation>
Extracted two civilians with their roles. Excluded "Detective Sarah Chen" because she is law enforcement.
</explanation>
</example_2>

<example_3>
<input>
The complainant, James Thompson, filed a complaint on February 15, 2023, alleging excessive force. Officer Lisa Wilson was assigned to investigate. Thompson's attorney, Michael Davis, submitted additional documentation.
</input>

<correct_output>
James Thompson - Complainant alleging excessive force
Michael Davis - Thompson's attorney who submitted documentation
</correct_output>

<explanation>
Extracted complainant and his attorney. Excluded "Officer Lisa Wilson" because she is law enforcement.
</explanation>
</example_3>

<example_4>
<input>
During the incident, Officer David Kim attempted to arrest suspect Robert Lee. Witness Sarah Martinez observed the altercation from her apartment. Lee sustained injuries and was transported to the hospital.
</input>

<correct_output>
Robert Lee - Suspect arrested, sustained injuries during incident
Sarah Martinez - Witness who observed altercation from her apartment
</correct_output>

<explanation>
Extracted subject/suspect and witness with context. Excluded "Officer David Kim".
</explanation>
</example_4>

<example_5>
<input>
The victim, Carlos Hernandez, died as a result of the shooting. His mother, Rosa Hernandez, filed a wrongful death lawsuit. Detective Johnson investigated the case. Witness Timothy Brown provided testimony about what he saw.
</input>

<correct_output>
Carlos Hernandez - Victim who died in shooting
Rosa Hernandez - Victim's mother who filed wrongful death lawsuit
Timothy Brown - Witness who provided testimony
</correct_output>

<explanation>
Extracted victim, family member, and witness. Excluded "Detective Johnson".
</explanation>
</example_5>

<example_6>
<input>
Officers responded to a disturbance. Several witnesses were interviewed. The investigation is ongoing.
</input>

<correct_output>
No civilians on this page
</correct_output>

<explanation>
No specific civilian names or identifiers provided. "Several witnesses" is too vague to extract.
</explanation>
</example_6>

<example_7>
<input>
Sergeant Robert Lee responded to a call involving Marcus Johnson, who was allegedly armed. Johnson's wife, Linda Johnson, called 911. Officer Sarah Chen arrived as backup. Witness Kevin Martinez saw the incident from across the street.
</input>

<correct_output>
Marcus Johnson - Subject allegedly armed, involved in incident
Linda Johnson - Marcus Johnson's wife who called 911
Kevin Martinez - Witness who saw incident from across the street
</correct_output>

<explanation>
Extracted subject, family member, and witness. Excluded "Sergeant Robert Lee" and "Officer Sarah Chen".
</explanation>
</example_7>

<example_8>
<input>
COMPLAINT SUMMARY
Complainant: Rachel Thompson
Incident Date: March 15, 2023
Subject Officers: Off. Brandon Taylor, Off. Mike Davis

Complainant alleges that officers used excessive force during arrest of her son, Anthony Thompson. Witness Jennifer Lee corroborated the account.
</input>

<correct_output>
Rachel Thompson - Complainant alleging excessive force against her son
Anthony Thompson - Rachel Thompson's son who was arrested
Jennifer Lee - Witness who corroborated complainant's account
</correct_output>

<explanation>
Extracted complainant, victim/subject, and witness from complaint document. Excluded the subject officers.
</explanation>
</example_8>

<example_9>
<input>
The suspect, John Doe, fled the scene. Witness Jane Doe provided a statement. Officer Smith pursued.
</input>

<correct_output>
No civilians on this page
</correct_output>

<explanation>
"John Doe" and "Jane Doe" are placeholder names (violation of critical instructions). Must be excluded.
</explanation>
</example_9>

<example_10>
<input>
USE OF FORCE INCIDENT REPORT
Subject: Miguel Ramirez (DOB: 05/12/1985)
Location: 123 Main Street

Officers arrived and encountered Ramirez, who was armed with a knife. His neighbor, Patricia Lee, had called 911. After the shooting, Ramirez was transported to County Hospital where he was treated by Dr. James Wilson.
</input>

<correct_output>
Miguel Ramirez (DOB: 05/12/1985) - Subject involved in use of force incident, armed with knife, shot and transported to hospital
Patricia Lee - Ramirez's neighbor who called 911
</correct_output>

<explanation>
Extracted subject with DOB and neighbor who called 911. Did not include "Dr. James Wilson" as he is medical staff, not directly involved in the police incident (though could be included if relevant to case). Excluded officers.
</explanation>
</example_10>
</few_shot_examples>

<thinking_process>
Before extracting civilian information, ask yourself:
1. Is this person clearly identified as a civilian (NOT a law enforcement officer)?
2. What identifying information is provided (name, relationship, role)?
3. What role did this civilian play in the incident or case?
4. Am I excluding all law enforcement officers?
5. If the name is a placeholder like "John Doe" or "Jane Doe", should I exclude it?
6. Is there enough specific information to include this person, or is it too vague?
</thinking_process>

<critical_instructions>
1. NEVER include any information about "John Doe," "Jane Doe," or other placeholder names.
2. ONLY include civilians - exclude all law enforcement officers (police, sheriff, deputies, detectives, sergeants, etc.).
3. If there is no relevant civilian information on the Current Page, return "No civilians on this page".
4. DO NOT infer, assume, or hallucinate any information not explicitly stated in the provided text.
5. Treat this task as a binary classification: either there are civilians to extract, or there aren't.
6. Always provide identifying information for each civilian - at minimum a name and role.
7. If someone is mentioned but their civilian status is unclear, omit them.
</critical_instructions>

<output_format>
Present the civilians in a clear, organized format:

Civilian Name (Additional info if available) - Role/Context

Examples:
John Martinez - Subject involved in hostage situation and shooting
Maria Rodriguez - Witness who saw the incident
James Thompson - Complainant alleging excessive force
Carlos Hernandez - Victim who died in shooting
Rosa Hernandez - Victim's mother who filed wrongful death lawsuit

If multiple civilians have similar roles, group them:

Witnesses:
• Maria Rodriguez - Saw incident from apartment window
• Kevin Martinez - Observed altercation from across street
• Timothy Brown - Provided testimony about shooting
</output_format>

<warnings>
- Do not include law enforcement officers
- Do not include placeholder names like "John Doe" or "Jane Doe"
- Avoid including people whose civilian status is unclear or ambiguous
- Do not draw conclusions not explicitly stated in the text
- Do not extract civilians from headers, footers, or signature blocks unless they are substantively mentioned in the body text
- If there are no specific civilian identifiers, return "No civilians on this page"
</warnings>

<reference_materials>
Current Page:
{{current_page}}
</reference_materials>

<output_instruction>
Extract all civilian/subject information from the page following the format specified above:
</output_instruction>
"""

page_summary_verification_template = """
You are a Legal Document Verifier. Your task is to verify civilian extractions against the original document.

<task_description>
Review the Current Civilian Extraction and verify:
1. Every civilian listed appears in the Original Document's substantive text
2. All civilians mentioned in the Original Document are included
3. Identifying information and roles are accurate
4. Only civilians are included (no law enforcement officers)
5. No placeholder names like "John Doe" or "Jane Doe" are included

Return either the original extraction (if correct) or a corrected version.
</task_description>

<critical_rules>
- Only include civilians explicitly stated in the Original Document's body text
- Exclude civilians from headers, footers, letterhead, and signature blocks unless substantively mentioned
- ONLY include civilians, subjects, complainants, witnesses, and victims
- EXCLUDE all law enforcement officers (police, sheriff, deputies, detectives, etc.)
- Never include "John Doe," "Jane Doe," or placeholder names
- If someone's civilian status is unclear, omit them
- If no civilians are mentioned, return: "No civilians on this page"
- Return ONLY the extraction - no commentary or explanations
</critical_rules>

<output_format>
Return one of:
1. The Current Civilian Extraction exactly as written (if completely accurate)
2. A corrected extraction in the same format:

Civilian Name (Additional info) - Role/Context

3. "No civilians on this page" (if no civilians are mentioned)
</output_format>

<reference_documents>
Original Document:
{{original_document}}

Current Civilian Extraction:
{{current_summary}}
</reference_documents>

<output_instruction>
Verify the Current Civilian Extraction against the Original Document. Return your output below:
</output_instruction>
"""

combine_template = """
You are a Legal Clerk merging civilian extractions into a single comprehensive list.

<task_description>
Merge two civilian extractions by:
1. Combining all unique civilians
2. Merging duplicate entries with complementary information
3. Resolving contradictions based on context, or dropping both if unclear
4. Maintaining only civilians (no law enforcement officers)
</task_description>

<critical_rules>
- Every civilian must appear in at least one source extraction
- When same civilian appears twice with compatible info: merge into one entry
- When contradictory information exists with clear context: keep correct one
- When contradictory information lacks context: DROP BOTH entries
- Exclude "John Doe," "Jane Doe," and placeholder names
- Exclude all law enforcement officers
- Only include civilians with specific identifying information
- If no civilians exist, return: "No civilians on this page"
- Precision over completeness - exclude questionable entries
</critical_rules>

<output_format>
Return civilians in a clear, organized format:

Civilian Name (Additional info) - Role/Context

Or return: "No civilians on this page"
</output_format>

<reference_materials>
Civilian Extraction 1:
{{summary_1}}

Civilian Extraction 2:
{{summary_2}}
</reference_materials>

<output_instruction>
Merge the two civilian extractions into a single comprehensive list:
</output_instruction>
"""

verification_template = """
You are verifying a combined civilian extraction against its two source extractions.

<task_description>
Verify that the combined extraction:
1. Contains only civilians from the source extractions (no hallucinations)
2. Has no duplicate civilians
3. Properly handled contradictions (kept correct one or dropped both)
4. Only includes civilians (no law enforcement officers)
5. Excludes placeholder names like "John Doe" or "Jane Doe"
6. Includes all identifying information available

Return the original if correct, or a corrected version.
</task_description>

<critical_rules>
- Every civilian must appear in at least one source extraction
- No duplicate civilians (same person listed twice)
- Contradictions: if context was clear, one entry kept; if unclear, both dropped
- ONLY include civilians
- EXCLUDE all law enforcement officers
- Exclude "John Doe," "Jane Doe," and placeholder names
- If no valid civilians, return: "No civilians on this page"
- Return ONLY the extraction - no commentary
</critical_rules>

<output_format>
Return one of:
1. The combined extraction exactly as written (if completely accurate)
2. A corrected extraction:

Civilian Name (Additional info) - Role/Context

3. "No civilians on this page" (if no valid civilians)
</output_format>

<reference_materials>
Civilian Extraction 1:
{{summary_1}}

Civilian Extraction 2:
{{summary_2}}

Current Combined Civilian Extraction:
{{current_combined_summary}}
</reference_materials>

<output_instruction>
Verify the combined extraction against the source extractions. Return your output below:
</output_instruction>
"""

condense_interval_template = """
<task_description>
You are reviewing a comprehensive interval summary that contains civilian information. Condense to the most important civilian identifiers.
</task_description>

<critical_requirements>
1. Return the most important civilian information
2. Focus on civilians central to the case (subjects, complainants, victims, key witnesses)
</critical_requirements>

<selection_criteria>
Prioritize civilian information that includes:
- Specific names with clear roles
- Subjects/suspects directly involved in incidents
- Complainants who filed complaints
- Victims who were harmed
- Key witnesses who provided critical testimony
- Family members who filed lawsuits or complaints

Deprioritize:
- Vague references to civilians without names
- Civilians mentioned in passing with minimal relevance
- Redundant information about the same civilians
- Peripheral witnesses or associates
</selection_criteria>

<full_interval_summary>
{{full_interval_summary}}
</full_interval_summary>

<output_format>
Return the condensed civilian information maintaining the original structure:

- Subjects/Suspects
  • [Most important civilian entries]
  • END BULLETPOINTS

- Complainants
  • [Most important civilian entries]
  • END BULLETPOINTS

- Witnesses/Victims
  • [Most important civilian entries]
  • END BULLETPOINTS
</output_format>

<output_instruction>
Extract and return the most important civilian information now:
</output_instruction>
"""

# Extraction prompts
extract_template = """
<civilian_extraction_prompt>
<context>
Your task is to analyze police documents to identify ALL CIVILIANS (non-law-enforcement individuals) mentioned in the document. Focus on extracting:
1. Subjects/suspects involved in incidents with police
2. Complainants who filed complaints against officers or department
3. Witnesses who observed incidents or provided testimony
4. Victims who were harmed in incidents
5. Family members or associates mentioned in relation to incidents

You must carefully distinguish civilians from law enforcement officers.

Key definitions:
- CIVILIANS: Subjects, suspects, complainants, witnesses, victims, family members, and any other non-law-enforcement individuals
- NOT civilians: Police officers, sheriffs, deputies, detectives, sergeants, lieutenants, captains, chiefs, and other sworn law enforcement personnel

Extract comprehensive identifying information for each civilian including:
- Full name or identifying information
- Role in the incident or case
- Relationship to other individuals (if mentioned)
</context>

<format_requirements>
- Each civilian entry should include: Name - Role/Context
- Group civilians by their primary role in the case
- Only include individuals explicitly identified as civilians/non-law-enforcement
- If no civilians are mentioned, report that no civilians could be identified
</format_requirements>

<document_for_review>
{{source_text}}
</document_for_review>

<thinking_process>
1. First, identify all people mentioned in the document
2. For each person, determine: Are they a civilian (not law enforcement)?
3. For confirmed civilians, extract:
   a) Name
   b) Role (subject, complainant, witness, victim, family member, etc.)
   c) Context (what happened to them, what they did, etc.)
4. Exclude all law enforcement officers
5. Exclude placeholder names like "John Doe" or "Jane Doe"
6. Group civilians by their primary role for clarity
7. Ensure each civilian has sufficient identifying information
8. Verify no duplicates (same civilian listed twice)
</thinking_process>

<verification_steps>
1. Re-examine each extracted civilian entry
2. Confirm they are NOT identified as law enforcement in the text
3. Verify identifying information (name, role) is accurate
4. Check that no law enforcement officers were included
5. Ensure no placeholder names were included
6. Verify role descriptions accurately reflect the text
7. Confirm all civilians mentioned in the document are included
</verification_steps>

<output_format>
SUBJECTS/SUSPECTS:
[List individuals involved in incidents with police, arrested, detained, or suspected]
If there are no subjects/suspects, return the text "NO SUBJECTS/SUSPECTS"

COMPLAINANTS:
[List individuals who filed complaints against officers or department]
If there are no complainants, return the text "NO COMPLAINANTS"

WITNESSES:
[List individuals who observed incidents or provided testimony]
If there are no witnesses, return the text "NO WITNESSES"

VICTIMS:
[List individuals who were harmed in incidents]
If there are no victims, return the text "NO VICTIMS"

FAMILY MEMBERS/ASSOCIATES:
[List relatives or associates mentioned in relation to incidents]
If there are no family members/associates, return the text "NO FAMILY MEMBERS/ASSOCIATES"

For each civilian, use format:
Civilian Name - Specific role/context description

CONFIDENCE LEVEL: [High/Medium/Low]

NOTES:
[Any relevant clarifications about civilian identifications or ambiguities]
</output_format>
</civilian_extraction_prompt>
"""

extract_verification_template = """
<civilian_extraction_verification_prompt>
<context>
Your task is to verify the accuracy of the extracted civilian list from police documentation. You must distinguish between:
- CIVILIANS: Subjects, suspects, complainants, witnesses, victims, family members, and non-law-enforcement individuals
- NOT CIVILIANS: Police officers, sheriffs, deputies, detectives, and other sworn law enforcement personnel

When contradictory information appears, determine which is most reliable based on context and specificity.
</context>

<proposed_civilian_list>
{{initial_dates}}
</proposed_civilian_list>

<document_for_review>
{{source_text}}
</document_for_review>

<verification_process>
1. Carefully re-read the entire document
2. Identify ALL people mentioned and categorize them as civilians or law enforcement
3. For each person in the proposed list, verify:
   a) They are NOT identified as law enforcement
   b) Their name is accurate
   c) Role/context is accurate
   d) They are genuinely civilians
4. Check for civilians mentioned in the document but missing from the list
5. Check for law enforcement officers incorrectly included in the list
6. Verify no placeholder names like "John Doe" or "Jane Doe" are included
7. Ensure identifying information is complete and accurate
</verification_process>

<critical_considerations>
- Only include individuals NOT identified as law enforcement
- Exclude ALL law enforcement officers (police, sheriff, deputies, detectives, etc.)
- Exclude placeholder or anonymous references
- When the same civilian is mentioned multiple times, ensure single consistent entry
- If contradictory information exists, select the most reliable based on context
- Identifying information should be as complete as the document allows
</critical_considerations>

<output_format>
VERIFICATION RESULT: [CONFIRMED / CORRECTED / REJECTED]

SUBJECTS/SUSPECTS:
[List with Name - Role/Context]
If there are no subjects/suspects, return the text "NO SUBJECTS/SUSPECTS"

COMPLAINANTS:
[List with Name - Role/Context]
If there are no complainants, return the text "NO COMPLAINANTS"

WITNESSES:
[List with Name - Role/Context]
If there are no witnesses, return the text "NO WITNESSES"

VICTIMS:
[List with Name - Role/Context]
If there are no victims, return the text "NO VICTIMS"

FAMILY MEMBERS/ASSOCIATES:
[List with Name - Role/Context]
If there are no family members/associates, return the text "NO FAMILY MEMBERS/ASSOCIATES"

CONFIDENCE LEVEL: [High/Medium/Low]

JUSTIFICATION:
[Detailed explanation of verification, including any corrections made and why]

KEY EVIDENCE:
[Direct quotes from the document that support civilian identifications]

CORRECTIONS MADE (if applicable):
[List of civilians added, removed, or modified with explanations]
</output_format>
</civilian_extraction_verification_prompt>
"""

format_conversion_template = None  # Civilian names don't need format conversion

# Citation prompts
primary_citation_template = """
<objective>
Analyze this page of text to determine if it contains citations that mention SPECIFIC CIVILIANS identified in the civilian extraction analysis.
</objective>

<civilian_extraction_context>
The civilian extraction analysis identified the following:

EXTRACTED CIVILIANS:
{{incident_date}}

{{date_extraction_context}}

You should look for citations that mention these SPECIFIC civilians by name or clear role reference.
</civilian_extraction_context>

<civilian_citation_definition>
CIVILIAN CITATIONS must clearly reference non-law-enforcement individuals:
- Direct name mentions: "John Martinez fled the scene"
- Role-based references: "The complainant, Maria Rodriguez, filed the complaint"
- Subject/witness identification: "Witness Kevin Lee observed the incident"
- Victim references: "The victim, Carlos Hernandez, sustained injuries"

NOT civilian citations:
- Vague references like "the suspect" or "a witness" without specific identification
- Law enforcement officer names
- Placeholder names like "John Doe" or "Jane Doe"
</civilian_citation_definition>

<targeted_search_guidance>
Based on the civilian extraction context above, prioritize finding citations that:
1. Mention specific civilian names from the extraction list
2. Use subject/complainant/witness/victim designations that match the extracted civilians
3. Provide context about what these civilians did or what happened to them
4. Clearly distinguish civilians from law enforcement officers
</targeted_search_guidance>

<page_text>
{{page_text}}
</page_text>

<output_instructions>
Focus on finding citations that specifically mention the CIVILIANS identified in the extraction analysis.
Respond with 'YES' or 'NO' followed by your reasoning and an exact quote.

Format:
[YES/NO]: [Your reasoning explaining whether this page contains citations mentioning the SPECIFIC civilians]
Quote: "[Exact text from the page that supports your decision]"
</output_instructions>
"""

validator_citation_template = """
<objective>
Verify whether this page contains a HIGH-QUALITY citation that identifies a specific civilian.
You must distinguish between civilians and law enforcement officers.
</objective>

<extracted_civilians>
{{incident_date}}
</extracted_civilians>

<civilian_citation_definition>
CIVILIAN CITATIONS must clearly identify a NON-LAW-ENFORCEMENT INDIVIDUAL:
- Specific name mentions: "John Martinez"
- Role with name: "Complainant Maria Rodriguez"
- Subject/witness/victim designation: "Witness Kevin Lee"
- Clear context: "The suspect, Robert Davis"

EXPLICITLY NOT civilian citations:
- Vague references without specific names ("the suspect," "a witness")
- Law enforcement officer names
- Placeholder names like "John Doe" or "Jane Doe"
</civilian_citation_definition>

<primary_analysis>
{{primary_analysis}}
</primary_analysis>

<page_text>
{{page_text}}
</page_text>

<validation_criteria>
For a citation to be valid:
1. It must mention a specific civilian name or identifier
2. The person must NOT be identified as law enforcement
3. The quote must be an exact excerpt from the page text
4. The context must distinguish this civilian from law enforcement officers

Ask yourself:
- Is this person explicitly NOT law enforcement?
- Is there a specific name or identifier?
- Is this a placeholder name that should be excluded?
</validation_criteria>

<output_format>
Return your analysis as a JSON object with the following structure:
{
    "final_decision": "YES" or "NO",
    "validator_reasoning": "Detailed explanation of why this is or is not a valid civilian citation",
    "verified_quote": "The exact quote from the page, or explanation of why the quote is invalid"
}
</output_format>
"""

# Export all prompts in organized structure
PROMPTS = {
    "summarization": {
        "memory_log": memory_log_template,
        "summary_for_memory": summary_template_for_memory_log,
        "page_summary": page_summary_template,
        "page_verification": page_summary_verification_template,
        "combine": combine_template,
        "verification": verification_template,
        "condense_interval": condense_interval_template,
    },
    "extraction": {
        "extract": extract_template,
        "verification": extract_verification_template,
        "format_conversion": format_conversion_template,
    },
    "citations": {
        "primary_citation": primary_citation_template,
        "validator_citation": validator_citation_template,
    },
}
