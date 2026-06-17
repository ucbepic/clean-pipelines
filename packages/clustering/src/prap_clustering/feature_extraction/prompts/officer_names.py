"""
Officer name extraction prompts.

This module contains all 12 prompts needed for officer name extraction:
- Summarization stage: 7 prompts
- Extraction stage: 3 prompts
- Citation finding: 2 prompts

Adapted from dates.py to focus on law enforcement officer identification.
"""

memory_log_template = """
As a Legal Clerk, your task is to review the new summary and update the memory log only when the new summary contains crucial information directly related to law enforcement officers involved in the case. Maintain a concise memory log that focuses on officer identifiers including names, badge numbers, ranks, and roles.
</task_description>

<guidelines>
1. Review and Compare:
   • Carefully review the current memory log and the new summary.
   • Determine if the new summary contains crucial officer identification information not already in the memory log.

2. Identify Crucial Information:
   • Focus on officer names, badge numbers, ranks, and roles.
   • Look for details about which officers were involved in incidents, investigations, or disciplinary proceedings.

3. Update Selectively:
   • Only update the memory log if the new summary contains crucial officer information not already present.
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
Ensure the summary includes ALL of the following officer-related elements, if present:
a. Officer names (full names when available)
b. Badge numbers or employee IDs
c. Ranks and titles (Officer, Detective, Sergeant, Lieutenant, etc.)
d. Roles in the incident (responding officer, supervisor, investigator, etc.)
e. Department or unit assignments
f. Officers involved in use of force, misconduct allegations, or investigations
g. Officers who provided testimony or statements
h. Supervisors or command staff mentioned

For each officer mentioned, be specific about their identifier information.
</essential_information>

<thinking_process>
Before updating the memory log, consider:
1. Does the new summary contain any officer identification information not already in the memory log?
2. Are there new officers mentioned, or new details about already-listed officers?
3. Can this new information be integrated into the existing log without disrupting its flow?
4. Is this information essential to understanding who was involved in the case?
5. Am I maintaining the conciseness of the log while including all crucial officer details?
</thinking_process>

<warnings>
- Do not add information that is not directly stated in the document
- Avoid speculation or inference beyond what is explicitly mentioned
- Do not remove or alter existing crucial officer information in the memory log
- Ensure that any updates maintain the logical organization of officer information
- Be cautious of potential inconsistencies between the new summary and existing log
- NEVER include civilian names, complainant names, or subject names in the officer memory log
</warnings>

<reference_materials>
## Original Memory Log ##
{{ memory_log }}

## New Summary ##
{{ summary }}
</reference_materials>

<output_instruction>
Based on your review of the current memory log and the new summary, provide either an updated memory log incorporating the crucial new officer information, or reproduce the original memory log if no update is necessary. Ensure the output maintains a concise focus on officer identifiers and roles:
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
Your task is to generate a comprehensive, bulletpoint summary of all law enforcement officer information contained in the provided document excerpt. Extract all key details about officers mentioned on the current page.
</task_description>

<legal_document_essential_information>
If the document is classified as a legal document, ensure the summary includes ALL of the following officer-related elements, if present:

a. Officer names (full names, or last name with first initial)
b. Badge numbers, employee IDs, or other unique identifiers
c. Ranks and titles (Officer, Detective, Sergeant, Lieutenant, Captain, Chief, etc.)
d. Department or unit assignments (patrol, investigations, internal affairs, etc.)
e. Roles in incidents (responding officer, backup, supervisor on scene, etc.)
f. Officers who used force or were involved in misconduct allegations
g. Officers who conducted investigations or interviews
h. Officers who provided testimony, statements, or reports
i. Supervisors or command staff involved in reviews or decisions
j. Officers who received discipline or were subjects of investigation

For each officer mentioned, be specific when referring to their names, identifiers, and roles.
</legal_document_essential_information>

<other_document_type_guidelines>
If classified as Other Document Type, focus on:
a. Names of law enforcement personnel mentioned
b. Their roles or positions
c. Any identifying information provided
d. Context in which they are mentioned
</other_document_type_guidelines>

<critical_instructions>
1. NEVER include any information about "John Doe," "Jane Doe," or other anonymous/ambiguous entities. If specific identifying information is not available, state that officer details are unavailable or redacted.
2. ONLY include law enforcement officers (police, sheriff, deputies, etc.) - DO NOT include civilian names, complainant names, witness names, or subject names.
3. If there is no relevant officer information to summarize on the Current Page, return an empty string ("").
4. DO NOT infer, assume, or hallucinate any information not explicitly stated in the provided text.
5. Treat this task as a binary classification: either there is relevant officer information to summarize, or there isn't.
</critical_instructions>

<thinking_process>
Before summarizing, consider:
1. Is this a legal document or another document type?
2. What officers are mentioned on this page?
3. What identifying information is provided for each officer (name, badge, rank)?
4. What roles did these officers play in the incident or case?
</thinking_process>

<output_format>
Present the summary using the following structure:

- Officers Involved in Incident
  • Officer Name (Badge #, Rank) - Role in incident
  • Officer Name (Badge #, Rank) - Role in incident

- Investigating Officers
  • Officer Name (Badge #, Rank) - Investigation role
  • Officer Name (Badge #, Rank) - Investigation role

- Supervisory Officers
  • Officer Name (Badge #, Rank) - Supervisory role
  • Officer Name (Badge #, Rank) - Supervisory role

- Officers Providing Testimony/Statements
  • Officer Name (Badge #, Rank) - Statement details
  • Officer Name (Badge #, Rank) - Statement details

</output_format>

<warnings>
- Do not include speculative information
- Do not include civilian names, witness names, or subject names
- Avoid summarizing irrelevant details
- Do not draw conclusions not explicitly stated in the text
- Only include individuals clearly identified as law enforcement officers
</warnings>

<reference_materials>
Current Page:
{{current_page}}
</reference_materials>

<output_instruction>
First, state the document classification (Legal Document or Other Document Type) and provide a brief explanation for your decision. Then, generate the current page summary following the appropriate guidelines based on the classification. Focus exclusively on law enforcement officer information.
</output_instruction>
"""

page_summary_template = """
<task_description>
As a Legal Clerk, your task is to identify and extract all law enforcement officer information from the provided document excerpt. Focus on officer names, badge numbers, ranks, and roles in the case. Extract comprehensive identifying information for every officer mentioned.
</task_description>

<guidelines>
1. Extract all law enforcement officer information from the current page.
2. Include names (full or last name with first initial), badge numbers, ranks, titles, and roles.
3. DO NOT include civilian names, complainant names, witness names, or subject/suspect names.
4. DO NOT include any information not explicitly stated in the document.
5. Present officers in a clear, organized manner with all available identifying information.
6. If someone's identity is ambiguous or unclear, omit them from your summary.
7. If no law enforcement officers are mentioned on the page, return "No officers on this page".
</guidelines>

<officer_extraction_priority>
Extract and highlight the following officer information when present:

1. Officers involved in incidents:
   - Officers who used force or were present during use of force
   - Officers who responded to calls or scenes
   - Officers involved in arrests or detentions
   - Officers involved in misconduct allegations

2. Investigating officers:
   - Internal affairs investigators
   - Detectives or investigators assigned to cases
   - Officers who conducted interviews or collected evidence
   - Supervisors overseeing investigations

3. Supervisory and command staff:
   - Sergeants, lieutenants, captains supervising incidents
   - Chiefs, assistant chiefs, commanders involved in reviews
   - Watch commanders or shift supervisors

4. Officers providing statements or testimony:
   - Officers who submitted reports
   - Officers who provided sworn statements
   - Officers who testified at hearings or trials

5. Officers subject to discipline:
   - Officers facing misconduct allegations
   - Officers who received discipline or sanctions
   - Officers involved in grievance or appeal processes

SELECTIVITY RULE: Only include individuals clearly identified as law enforcement officers.
Exclude civilians, complainants, witnesses, suspects, and subjects even if they interact with officers.
</officer_extraction_priority>

<few_shot_examples>
These examples demonstrate correct officer extraction behavior:

<example_1>
<input>
On January 26, 2014, Officer Michael Butera (Badge #4521) responded to a call regarding a traffic violation. Sergeant James Wilson supervised the incident. The suspect, John Martinez, fled the scene.
</input>

<correct_output>
Officer Michael Butera (Badge #4521) - Responding officer involved in incident
Sergeant James Wilson - Supervising officer on scene
</correct_output>

<explanation>
Extracted both officers with available identifiers and roles. Excluded "John Martinez" because he is the suspect, not an officer.
</explanation>
</example_1>

<example_2>
<input>
The investigation was conducted by Detective Sarah Chen (Badge #2847) and Detective Robert Park (Badge #2901). They interviewed three civilian witnesses. The case was supervised by Lieutenant Mark Davidson.
</input>

<correct_output>
Detective Sarah Chen (Badge #2847) - Investigating officer
Detective Robert Park (Badge #2901) - Investigating officer
Lieutenant Mark Davidson - Supervising officer overseeing investigation
</correct_output>

<explanation>
Extracted three officers with their ranks and roles. Did not include civilian witnesses.
</explanation>
</example_2>

<example_3>
<input>
The complainant, Maria Rodriguez, filed a complaint against the department. Officer James Thompson was assigned to review the complaint. Deputy Chief Karen Williams made the final determination.
</input>

<correct_output>
Officer James Thompson - Assigned to review complaint
Deputy Chief Karen Williams - Made final determination
</correct_output>

<explanation>
Extracted officers involved in the complaint process. Excluded "Maria Rodriguez" because she is the complainant, not an officer.
</explanation>
</example_3>

<example_4>
<input>
This memo is from Captain Richard Hayes (Employee ID: 10254) to Chief of Police Sandra Martinez regarding the incident. Officer David Kim (#3421) and Officer Lisa Chen (#3422) were present at the scene. The witness, Tom Johnson, provided a statement.
</input>

<correct_output>
Captain Richard Hayes (Employee ID: 10254) - Author of memo
Chief of Police Sandra Martinez - Recipient of memo
Officer David Kim (Badge #3421) - Present at scene
Officer Lisa Chen (Badge #3422) - Present at scene
</correct_output>

<explanation>
Extracted all four officers with their identifiers and roles. Excluded witness "Tom Johnson".
</explanation>
</example_4>

<example_5>
<input>
Detective John Anderson conducted the internal affairs investigation. He interviewed Officer Brandon Miller (Badge #5501), who was the subject of the complaint. Sergeant Emily Turner provided a witness statement. The complainant was a civilian resident.
</input>

<correct_output>
Detective John Anderson - Conducted internal affairs investigation
Officer Brandon Miller (Badge #5501) - Subject of complaint
Sergeant Emily Turner - Provided witness statement
</correct_output>

<explanation>
Extracted all three officers. Miller is included even though he's the subject of the complaint because he is still an officer. Excluded the unnamed civilian complainant.
</explanation>
</example_5>

<example_6>
<input>
The shift supervisor was notified at 3:00 PM. Several officers responded to the scene. The investigation is ongoing.
</input>

<correct_output>
No officers on this page
</correct_output>

<explanation>
No specific officer names or identifiers provided. "Shift supervisor" and "several officers" are too vague to extract.
</explanation>
</example_6>

<example_7>
<input>
Officer Michael Davis (Badge #4192) discharged his firearm. His partner, Officer Jennifer White (Badge #4201), was present but did not discharge her weapon. Sergeant Robert Lee (#5012) responded as supervisor. The suspect sustained injuries and was transported to the hospital.
</input>

<correct_output>
Officer Michael Davis (Badge #4192) - Discharged firearm during incident
Officer Jennifer White (Badge #4201) - Partner present at scene, did not discharge weapon
Sergeant Robert Lee (Badge #5012) - Supervisor responded to scene
</correct_output>

<explanation>
Extracted all three officers with detailed role information. Excluded the suspect.
</explanation>
</example_7>

<example_8>
<input>
INTERNAL AFFAIRS BUREAU
Investigator: Det. Sarah Martinez #2765
Reviewed By: Lt. Thomas Anderson
Date: March 15, 2023

Subject Officer: Off. Brandon Taylor #3891
Allegation: Excessive force

The investigation found that Officer Taylor used force during the arrest of suspect James Wilson on February 10, 2023.
</input>

<correct_output>
Detective Sarah Martinez (Badge #2765) - Internal Affairs investigator
Lieutenant Thomas Anderson - Reviewed investigation
Officer Brandon Taylor (Badge #3891) - Subject officer, allegation of excessive force
</correct_output>

<explanation>
Extracted all three officers from the IA report header and body. Excluded suspect "James Wilson". Included subject officer even though he faces allegations.
</explanation>
</example_8>

<example_9>
<input>
Officers John Doe and Jane Doe responded to the call. An unidentified sergeant supervised the scene. The complainant wishes to remain anonymous.
</input>

<correct_output>
No officers on this page
</correct_output>

<explanation>
"John Doe" and "Jane Doe" are placeholder names (violation of critical instructions). "Unidentified sergeant" lacks specific identifying information. Anonymous complainant is not an officer.
</explanation>
</example_9>

<example_10>
<input>
USE OF FORCE REPORT
Reporting Officer: Sgt. Michael Brown, Badge #5234
Involved Officers:
- Off. Kevin Zhang, Badge #4567 (Primary)
- Off. Lisa Rodriguez, Badge #4571 (Backup)
- Off. Marcus Johnson, Badge #4582 (Backup)

Watch Commander: Capt. Rebecca Foster #6012
</input>

<correct_output>
Sergeant Michael Brown (Badge #5234) - Reporting officer
Officer Kevin Zhang (Badge #4567) - Primary officer involved in use of force
Officer Lisa Rodriguez (Badge #4571) - Backup officer present
Officer Marcus Johnson (Badge #4582) - Backup officer present
Captain Rebecca Foster (Badge #6012) - Watch commander
</correct_output>

<explanation>
Extracted all five officers from the use of force report with their roles clearly identified.
</explanation>
</example_10>
</few_shot_examples>

<thinking_process>
Before extracting officer information, ask yourself:
1. Is this person explicitly identified as a law enforcement officer (police, sheriff, deputy, detective, etc.)?
2. What identifying information is provided (name, badge number, rank, title)?
3. What role did this officer play in the incident or case?
4. Am I excluding civilians, complainants, witnesses, and subjects?
5. If the name is a placeholder like "John Doe" or "Jane Doe", should I exclude it?
6. Is there enough specific information to include this officer, or is it too vague?
</thinking_process>

<critical_instructions>
1. NEVER include any information about "John Doe," "Jane Doe," or other placeholder names.
2. ONLY include law enforcement officers - exclude all civilians, complainants, witnesses, subjects, and suspects.
3. If there is no relevant officer information on the Current Page, return "No officers on this page".
4. DO NOT infer, assume, or hallucinate any information not explicitly stated in the provided text.
5. Treat this task as a binary classification: either there are officers to extract, or there aren't.
6. Always provide identifying information for each officer - at minimum a name and rank/role.
7. If someone is mentioned but their law enforcement status is unclear, omit them.
</critical_instructions>

<output_format>
Present the officers in a clear, organized format:

Officer Name (Badge #/ID, Rank/Title) - Role/Context

Examples:
Officer Michael Butera (Badge #4521) - Responding officer involved in shooting
Detective Sarah Chen (Badge #2847) - Lead investigator on internal affairs case
Sergeant James Wilson - Supervising officer on scene
Captain Rebecca Foster (Badge #6012) - Watch commander overseeing incident response
Lieutenant Thomas Anderson - Reviewed use of force investigation

If multiple officers have similar roles, group them:

Officers involved in incident:
• Officer Kevin Zhang (Badge #4567) - Primary officer
• Officer Lisa Rodriguez (Badge #4571) - Backup officer
• Officer Marcus Johnson (Badge #4582) - Backup officer
</output_format>

<warnings>
- Do not include civilians, complainants, witnesses, subjects, or suspects
- Do not include placeholder names like "John Doe" or "Jane Doe"
- Avoid including people whose law enforcement status is unclear or ambiguous
- Do not draw conclusions not explicitly stated in the text
- Do not extract officers from headers, footers, or signature blocks unless they are substantively mentioned in the body text
- If there are no specific officer identifiers, return "No officers on this page"
</warnings>

<reference_materials>
Current Page:
{{current_page}}
</reference_materials>

<output_instruction>
Extract all law enforcement officer information from the page following the format specified above:
</output_instruction>
"""

page_summary_verification_template = """
You are a Legal Document Verifier. Your task is to verify officer extractions against the original document.

<task_description>
Review the Current Officer Extraction and verify:
1. Every officer listed appears in the Original Document's substantive text
2. All officers mentioned in the Original Document are included
3. Identifying information (badges, ranks) is accurate
4. Only law enforcement officers are included (no civilians, complainants, witnesses, or subjects)
5. No placeholder names like "John Doe" or "Jane Doe" are included

Return either the original extraction (if correct) or a corrected version.
</task_description>

<critical_rules>
- Only include officers explicitly stated in the Original Document's body text
- Exclude officers from headers, footers, letterhead, and signature blocks unless substantively mentioned
- ONLY include law enforcement officers (police, sheriff, deputies, detectives, etc.)
- EXCLUDE all civilians, complainants, witnesses, subjects, and suspects
- Never include "John Doe," "Jane Doe," or placeholder names
- Use "unidentified officer" if an officer is mentioned but no specific name is given
- If no officers are mentioned, return: "No officers on this page"
- Return ONLY the extraction - no commentary or explanations
</critical_rules>

<output_format>
Return one of:
1. The Current Officer Extraction exactly as written (if completely accurate)
2. A corrected extraction in the same format:

Officer Name (Badge #/ID, Rank) - Role/Context

3. "No officers on this page" (if no officers are mentioned)
</output_format>

<reference_documents>
Original Document:
{{original_document}}

Current Officer Extraction:
{{current_summary}}
</reference_documents>

<output_instruction>
Verify the Current Officer Extraction against the Original Document. Return your output below:
</output_instruction>
"""

combine_template = """
You are a Legal Clerk merging officer extractions into a single comprehensive list.

<task_description>
Merge two officer extractions by:
1. Combining all unique officers
2. Merging duplicate officers with complementary information
3. Resolving contradictions based on context, or dropping both if unclear
4. Maintaining only law enforcement officers (no civilians, witnesses, or subjects)
</task_description>

<critical_rules>
- Every officer must appear in at least one source extraction
- When same officer appears twice with compatible info: merge into one entry
- When contradictory information exists with clear context: keep correct one
- When contradictory information lacks context: DROP BOTH entries
- Exclude "John Doe," "Jane Doe," and placeholder names
- Exclude all civilians, complainants, witnesses, subjects, and suspects
- Only include law enforcement officers with specific identifying information
- If no officers exist, return: "No officers on this page"
- Precision over completeness - exclude questionable entries
</critical_rules>

<output_format>
Return officers in a clear, organized format:

Officer Name (Badge #/ID, Rank) - Role/Context

Or return: "No officers on this page"
</output_format>

<reference_materials>
Officer Extraction 1:
{{summary_1}}

Officer Extraction 2:
{{summary_2}}
</reference_materials>

<output_instruction>
Merge the two officer extractions into a single comprehensive list:
</output_instruction>
"""

verification_template = """
You are verifying a combined officer extraction against its two source extractions.

<task_description>
Verify that the combined extraction:
1. Contains only officers from the source extractions (no hallucinations)
2. Has no duplicate officers
3. Properly handled contradictions (kept correct one or dropped both)
4. Only includes law enforcement officers (no civilians, witnesses, or subjects)
5. Excludes placeholder names like "John Doe" or "Jane Doe"
6. Includes all identifying information available

Return the original if correct, or a corrected version.
</task_description>

<critical_rules>
- Every officer must appear in at least one source extraction
- No duplicate officers (same person listed twice)
- Contradictions: if context was clear, one entry kept; if unclear, both dropped
- ONLY include law enforcement officers
- EXCLUDE all civilians, complainants, witnesses, subjects, and suspects
- Exclude "John Doe," "Jane Doe," and placeholder names
- If no valid officers, return: "No officers on this page"
- Return ONLY the extraction - no commentary
</critical_rules>

<output_format>
Return one of:
1. The combined extraction exactly as written (if completely accurate)
2. A corrected extraction:

Officer Name (Badge #/ID, Rank) - Role/Context

3. "No officers on this page" (if no valid officers)
</output_format>

<reference_materials>
Officer Extraction 1:
{{summary_1}}

Officer Extraction 2:
{{summary_2}}

Current Combined Officer Extraction:
{{current_combined_summary}}
</reference_materials>

<output_instruction>
Verify the combined extraction against the source extractions. Return your output below:
</output_instruction>
"""

condense_interval_template = """
<task_description>
You are reviewing a comprehensive interval summary that contains officer information. Condense to the most important officer identifiers.
</task_description>

<critical_requirements>
1. Return the most important officer information
2. Focus on officers central to the case (involved in incidents, investigations, or discipline)
</critical_requirements>

<selection_criteria>
Prioritize officer information that includes:
- Specific names with badge numbers and ranks
- Officers directly involved in use of force or misconduct incidents
- Investigating officers and internal affairs personnel
- Officers who received discipline or were subjects of complaints
- Supervisory officers who made key decisions
- Officers who provided critical testimony or statements

Deprioritize:
- Vague references to officers without names
- Officers mentioned in passing with minimal relevance
- Redundant information about the same officers
- Administrative details without substantive case involvement
</selection_criteria>

<full_interval_summary>
{{full_interval_summary}}
</full_interval_summary>

<output_format>
Return the condensed officer information maintaining the original structure:

- Officers Involved in Incidents
  • [Most important officer entries]
  • END BULLETPOINTS

- Investigating Officers
  • [Most important officer entries]
  • END BULLETPOINTS

- Supervisory Officers
  • [Most important officer entries]
  • END BULLETPOINTS
</output_format>

<output_instruction>
Extract and return the most important officer information now:
</output_instruction>
"""

# Extraction prompts
extract_template = """
<officer_extraction_prompt>
<context>
Your task is to analyze police documents to identify ALL LAW ENFORCEMENT OFFICERS mentioned in the document. Focus on extracting:
1. Officers involved in use of force or misconduct incidents
2. Investigating officers and internal affairs personnel
3. Supervisory officers and command staff
4. Officers who provided statements or testimony
5. Officers who were subjects of complaints or discipline

You must carefully distinguish law enforcement officers from civilians, complainants, witnesses, and subjects.

Key definitions:
- LAW ENFORCEMENT OFFICERS: Police officers, sheriffs, deputies, detectives, sergeants, lieutenants, captains, chiefs, and other sworn personnel
- NOT officers: Civilians, complainants, witnesses, subjects, suspects, victims, or administrative staff without law enforcement status

Extract comprehensive identifying information for each officer including:
- Full name or last name with first initial
- Badge number or employee ID
- Rank or title
- Role in the incident or case
</context>

<format_requirements>
- Each officer entry should include: Name (Badge #/ID, Rank) - Role
- If badge number is not available, include rank and role
- Group officers by their primary role in the case
- Only include individuals explicitly identified as law enforcement officers
- If no officers are mentioned, report that no officers could be identified
</format_requirements>

<document_for_review>
{{source_text}}
</document_for_review>

<thinking_process>
1. First, identify all people mentioned in the document
2. For each person, determine: Are they a law enforcement officer?
3. For confirmed officers, extract:
   a) Name (full or last name with first initial)
   b) Badge number or employee ID (if mentioned)
   c) Rank or title (Officer, Detective, Sergeant, etc.)
   d) Role in the case (responding officer, investigator, subject of complaint, etc.)
4. Exclude all civilians, complainants, witnesses, subjects, and suspects
5. Exclude placeholder names like "John Doe" or "Jane Doe"
6. Group officers by their primary role for clarity
7. Ensure each officer has sufficient identifying information
8. Verify no duplicates (same officer listed twice)
</thinking_process>

<verification_steps>
1. Re-examine each extracted officer entry
2. Confirm they are explicitly identified as law enforcement in the text
3. Verify identifying information (name, badge, rank) is accurate
4. Check that no civilians, witnesses, or subjects were included
5. Ensure no placeholder names were included
6. Verify role descriptions accurately reflect the text
7. Confirm all officers mentioned in the document are included
</verification_steps>

<output_format>
OFFICERS INVOLVED IN INCIDENT:
[List officers who were present at or involved in the primary incident]

INVESTIGATING OFFICERS:
[List officers who conducted investigations or interviews]

SUPERVISORY OFFICERS:
[List supervisors, command staff, and review personnel]

OFFICERS SUBJECT TO COMPLAINTS/DISCIPLINE:
[List officers who were subjects of complaints, allegations, or discipline]

OFFICERS PROVIDING STATEMENTS/TESTIMONY:
[List officers who provided statements, reports, or testimony]

For each officer, use format:
Officer Name (Badge #XXXX, Rank) - Specific role description

If no officers found in a category, state "None identified"

CONFIDENCE LEVEL: [High/Medium/Low]

NOTES:
[Any relevant clarifications about officer identifications or ambiguities]
</output_format>
</officer_extraction_prompt>
"""

extract_verification_template = """
<officer_extraction_verification_prompt>
<context>
Your task is to verify the accuracy of the extracted officer list from police documentation. You must distinguish between:
- LAW ENFORCEMENT OFFICERS: Police, sheriffs, deputies, detectives, and other sworn personnel
- NOT OFFICERS: Civilians, complainants, witnesses, subjects, suspects, and administrative staff

When contradictory information appears, determine which is most reliable based on context and specificity.
</context>

<proposed_officer_list>
{{initial_dates}}
</proposed_officer_list>

<document_for_review>
{{source_text}}
</document_for_review>

<verification_process>
1. Carefully re-read the entire document
2. Identify ALL people mentioned and categorize them as officers or non-officers
3. For each person in the proposed list, verify:
   a) They are explicitly identified as a law enforcement officer
   b) Their name is accurate
   c) Badge number/ID is correct (if provided)
   d) Rank/title is accurate
   e) Role description matches the document
4. Check for officers mentioned in the document but missing from the list
5. Check for non-officers incorrectly included in the list
6. Verify no placeholder names like "John Doe" or "Jane Doe" are included
7. Ensure identifying information is complete and accurate
</verification_process>

<critical_considerations>
- Only include individuals explicitly identified as law enforcement officers
- Exclude ALL civilians, complainants, witnesses, subjects, and suspects
- Exclude placeholder or anonymous references
- When the same officer is mentioned multiple times, ensure single consistent entry
- If contradictory information exists, select the most reliable based on context
- Identifying information should be as complete as the document allows
</critical_considerations>

<output_format>
VERIFICATION RESULT: [CONFIRMED / CORRECTED / REJECTED]

OFFICERS INVOLVED IN INCIDENT:
[List with Name (Badge #, Rank) - Role]

INVESTIGATING OFFICERS:
[List with Name (Badge #, Rank) - Role]

SUPERVISORY OFFICERS:
[List with Name (Badge #, Rank) - Role]

OFFICERS SUBJECT TO COMPLAINTS/DISCIPLINE:
[List with Name (Badge #, Rank) - Role]

OFFICERS PROVIDING STATEMENTS/TESTIMONY:
[List with Name (Badge #, Rank) - Role]

CONFIDENCE LEVEL: [High/Medium/Low]

JUSTIFICATION:
[Detailed explanation of verification, including any corrections made and why]

KEY EVIDENCE:
[Direct quotes from the document that support officer identifications]

CORRECTIONS MADE (if applicable):
[List of officers added, removed, or modified with explanations]
</output_format>
</officer_extraction_verification_prompt>
"""

format_conversion_template = None  # Officer names don't need format conversion

# Citation prompts
primary_citation_template = """
<objective>
Analyze this page of text to determine if it contains citations that mention SPECIFIC OFFICERS identified in the officer extraction analysis.
</objective>

<officer_extraction_context>
The officer extraction analysis identified the following:

EXTRACTED OFFICERS:
{{incident_date}}

{{date_extraction_context}}

You should look for citations that mention these SPECIFIC officers by name, badge number, or clear role reference.
</officer_extraction_context>

<officer_citation_definition>
OFFICER CITATIONS must clearly reference law enforcement officers:
- Direct name mentions: "Officer Michael Butera responded"
- Badge number references: "Badge #4521 was involved"
- Rank and name combinations: "Detective Sarah Chen conducted the investigation"
- Role-based references when clearly identifying a specific officer: "The responding officer, Butera, discharged his weapon"

NOT officer citations:
- Vague references like "officers responded" or "a sergeant was present" without specific identification
- Civilian names, complainant names, witness names, or subject names
- Placeholder names like "John Doe" or "Jane Doe"
- Administrative personnel without law enforcement status
</officer_citation_definition>

<targeted_search_guidance>
Based on the officer extraction context above, prioritize finding citations that:
1. Mention specific officer names from the extraction list
2. Reference badge numbers or employee IDs from the extraction list
3. Use rank-name combinations that match the extracted officers
4. Provide context about what these officers did or their roles
5. Clearly distinguish officers from civilians, witnesses, or subjects
</targeted_search_guidance>

<page_text>
{{page_text}}
</page_text>

<output_instructions>
Focus on finding citations that specifically mention the OFFICERS identified in the extraction analysis.
Respond with 'YES' or 'NO' followed by your reasoning and an exact quote.

Format:
[YES/NO]: [Your reasoning explaining whether this page contains citations mentioning the SPECIFIC officers]
Quote: "[Exact text from the page that supports your decision]"
</output_instructions>
"""

validator_citation_template = """
<objective>
Verify whether this page contains a HIGH-QUALITY citation that identifies a specific law enforcement officer.
You must distinguish between officers and non-officers (civilians, witnesses, subjects).
</objective>

<extracted_officers>
{{incident_date}}
</extracted_officers>

<officer_citation_definition>
OFFICER CITATIONS must clearly identify a LAW ENFORCEMENT OFFICER:
- Specific name mentions: "Officer Michael Butera"
- Badge number with name: "Det. Sarah Chen, Badge #2847"
- Rank and name: "Sergeant James Wilson"
- Clear role identification: "The responding officer, Butera"

EXPLICITLY NOT officer citations:
- Vague references without specific names ("officers responded," "a detective investigated")
- Civilian names, complainant names, witness names
- Subject names or suspect names
- Placeholder names like "John Doe" or "Jane Doe"
- Administrative staff without law enforcement status
</officer_citation_definition>

<primary_analysis>
{{primary_analysis}}
</primary_analysis>

<page_text>
{{page_text}}
</page_text>

<validation_criteria>
For a citation to be valid:
1. It must mention a specific officer name, badge number, or unique identifier
2. The person must be clearly identified as a law enforcement officer
3. The quote must be an exact excerpt from the page text
4. The context must distinguish this officer from civilians, witnesses, or subjects

Ask yourself:
- Is this person explicitly identified as a law enforcement officer (police, detective, sergeant, etc.)?
- Or is this a civilian, complainant, witness, or subject?
- Is there a specific name, badge number, or unique identifier?
- Is this a placeholder name that should be excluded?
</validation_criteria>

<output_format>
Return your analysis as a JSON object with the following structure:
{
    "final_decision": "YES" or "NO",
    "validator_reasoning": "Detailed explanation of why this is or is not a valid officer citation",
    "verified_quote": "The exact quote from the page, or explanation of why the quote is invalid"
}
</output_format>
"""

# Export all prompts in organized structure
PROMPTS = {
    'summarization': {
        'memory_log': memory_log_template,
        'summary_for_memory': summary_template_for_memory_log,
        'page_summary': page_summary_template,
        'page_verification': page_summary_verification_template,
        'combine': combine_template,
        'verification': verification_template,
        'condense_interval': condense_interval_template,
    },
    'extraction': {
        'extract': extract_template,
        'verification': extract_verification_template,
        'format_conversion': format_conversion_template,
    },
    'citations': {
        'primary_citation': primary_citation_template,
        'validator_citation': validator_citation_template,
    },
}
