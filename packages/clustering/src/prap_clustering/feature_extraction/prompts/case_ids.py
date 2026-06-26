"""
Case ID extraction prompts.

This module contains all 12 prompts needed for case ID extraction:
- Summarization stage: 7 prompts
- Extraction stage: 3 prompts
- Citation finding: 2 prompts

Adapted from dates.py to focus on case identifier extraction.
Focus: Extract case numbers, incident numbers, report numbers, IA numbers.
"""

memory_log_template = """
As a Legal Clerk, your task is to review the new summary and update the memory log only when the new summary contains crucial information directly related to case identifiers. Maintain a concise memory log that focuses on case numbers, incident IDs, report numbers, and investigation identifiers.
</task_description>

<guidelines>
1. Review and Compare:
   • Carefully review the current memory log and the new summary.
   • Determine if the new summary contains crucial case identifier information not already in the memory log.

2. Identify Crucial Information:
   • Focus on case numbers, incident numbers, report numbers, IA numbers, complaint IDs, and investigation identifiers.
   • Look for details about which identifiers are associated with incidents, investigations, or reports.

3. Update Selectively:
   • Only update the memory log if the new summary contains crucial case identifier information not already present.
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
Ensure the summary includes ALL of the following case identifier elements, if present:
a. Case numbers and formats
b. Incident numbers or incident IDs
c. Report numbers
d. Internal Affairs (IA) case numbers
e. Complaint numbers or complaint IDs
f. Investigation identifiers
g. File numbers or file IDs
h. Any other official case identifiers

For each identifier mentioned, be specific about the format and context.
</essential_information>

<thinking_process>
Before updating the memory log, consider:
1. Does the new summary contain any case identifier information not already in the memory log?
2. Are there new identifiers mentioned, or new details about already-listed IDs?
3. Can this new information be integrated into the existing log without disrupting its flow?
4. Is this information essential to understanding how the case is identified?
5. Am I maintaining the conciseness of the log while including all crucial identifier details?
</thinking_process>

<warnings>
- Do not add information that is not directly stated in the document
- Avoid speculation or inference beyond what is explicitly mentioned
- Do not remove or alter existing crucial case identifier information in the memory log
- Ensure that any updates maintain the logical organization of identifier information
- Be cautious of potential inconsistencies between the new summary and existing log
</warnings>

<reference_materials>
## Original Memory Log ##
{{ memory_log }}

## New Summary ##
{{ summary }}
</reference_materials>

<output_instruction>
Based on your review of the current memory log and the new summary, provide either an updated memory log incorporating the crucial new case identifier information, or reproduce the original memory log if no update is necessary. Ensure the output maintains a concise focus on case identifiers:
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
Your task is to generate a comprehensive, bulletpoint summary of all case identifier information contained in the provided document excerpt. Extract all key details about case numbers, incident IDs, report numbers, and other official identifiers mentioned on the current page.
</task_description>

<legal_document_essential_information>
If the document is classified as a legal document, ensure the summary includes ALL of the following case identifier elements, if present:

a. Case numbers (format and context)
b. Incident numbers or incident report IDs
c. Report numbers
d. Internal Affairs (IA) case numbers
e. Complaint numbers or complaint IDs
f. Investigation identifiers
g. File numbers or document control numbers
h. Court case numbers or docket numbers
i. Any cross-referenced case identifiers

For each identifier mentioned, be specific about format and what it identifies.
</legal_document_essential_information>

<other_document_type_guidelines>
If classified as Other Document Type, focus on:
a. Any reference numbers or identifiers mentioned
b. Document tracking numbers
c. Context in which identifiers are used
</other_document_type_guidelines>

<critical_instructions>
1. DO NOT include placeholder identifiers like "Case #XXXXX" or generic examples without specific values.
2. If there is no relevant case identifier information to summarize on the Current Page, return an empty string ("").
3. DO NOT infer, assume, or hallucinate any identifiers not explicitly stated in the provided text.
4. Treat this task as a binary classification: either there are relevant case identifiers to summarize, or there aren't.
</critical_instructions>

<thinking_process>
Before summarizing, consider:
1. Is this a legal document or another document type?
2. What case identifiers are mentioned on this page?
3. What format do these identifiers use (IA-2020-045, Case #12345, etc.)?
4. What does each identifier reference (incident, complaint, investigation, report)?
</thinking_process>

<output_format>
Present the summary using the following structure:

- Case/Incident Numbers
  • Identifier format - Context/what it identifies
  • Identifier format - Context/what it identifies

- Report Numbers
  • Identifier format - Context/report type
  • Identifier format - Context/report type

- Investigation/IA Numbers
  • Identifier format - Context/investigation type
  • Identifier format - Context/investigation type

- Complaint Numbers
  • Identifier format - Context/complaint details
  • Identifier format - Context/complaint details

</output_format>

<warnings>
- Do not include speculative information
- Avoid summarizing placeholder or example identifiers
- Do not draw conclusions not explicitly stated in the text
- Only include identifiers clearly stated in the document
</warnings>

<reference_materials>
Current Page:
{{current_page}}
</reference_materials>

<output_instruction>
First, state the document classification (Legal Document or Other Document Type) and provide a brief explanation for your decision. Then, generate the current page summary following the appropriate guidelines based on the classification. Focus exclusively on case identifier information.
</output_instruction>
"""

page_summary_template = """
<task_description>
As a Legal Clerk, your task is to identify and extract all case identifiers from the provided document excerpt. Focus on case numbers, incident IDs, report numbers, IA numbers, and any other official tracking identifiers. Extract comprehensive information for every identifier mentioned.
</task_description>

<guidelines>
1. Extract all case identifier information from the current page.
2. Include the complete identifier format and context (what it identifies).
3. DO NOT include placeholder identifiers or generic examples.
4. DO NOT include any information not explicitly stated in the document.
5. Present identifiers in a clear, organized manner with format and context.
6. If an identifier format is ambiguous or unclear, include it with a note.
7. If no case identifiers are mentioned on the page, return "No case identifiers on this page".
</guidelines>

<case_id_extraction_priority>
Extract and highlight the following case identifier types when present:

1. Case numbers:
   - Primary case numbers for incidents
   - Cross-referenced or related case numbers
   - Court case numbers or docket numbers

2. Incident numbers:
   - Incident report numbers
   - Event numbers or call numbers
   - CAD (Computer Aided Dispatch) numbers

3. Internal Affairs identifiers:
   - IA case numbers (IA-2020-045, IAD552, etc.)
   - OIA numbers (Office of Internal Affairs)
   - Investigation tracking numbers

4. Complaint identifiers:
   - Complaint numbers or complaint IDs
   - Allegation tracking numbers
   - Grievance numbers

5. Report numbers:
   - Police report numbers
   - Supplemental report numbers
   - Use of force report numbers

6. File/document identifiers:
   - File numbers
   - Document control numbers
   - Archive or storage identifiers

SELECTIVITY RULE: Only include identifiers explicitly stated in the document.
Exclude placeholder examples, template markers, or generic references without specific values.
</case_id_extraction_priority>

<few_shot_examples>
These examples demonstrate correct case identifier extraction behavior:

<example_1>
<input>
On January 26, 2014, Officer Butera responded to Incident #2014-0567. The subsequent investigation was assigned Case Number IA-2014-023 by Internal Affairs.
</input>

<correct_output>
Incident #2014-0567 - Initial incident involving Officer Butera
IA-2014-023 - Internal Affairs case number for investigation
</correct_output>

<explanation>
Extracted both identifiers with their formats and contexts.
</explanation>
</example_1>

<example_2>
<input>
INTERNAL AFFAIRS CASE SUMMARY
Case Number: H-OIA-097-20-A
Related Incident: #2020-1245
Complaint #: C-2020-0089

This investigation concerns the use of force incident documented in Report #2020-UOF-034.
</input>

<correct_output>
H-OIA-097-20-A - Internal Affairs case number
Incident #2020-1245 - Related incident number
C-2020-0089 - Complaint number
Report #2020-UOF-034 - Use of force report number
</correct_output>

<explanation>
Extracted all four case identifiers from the IA case summary header and body.
</explanation>
</example_2>

<example_3>
<input>
This report references previous investigations under Case #12345 and Case #12346. The current matter is filed as IAD552 with supplemental reports IAD552-A and IAD552-B.
</input>

<correct_output>
Case #12345 - Previous investigation reference
Case #12346 - Previous investigation reference
IAD552 - Current Internal Affairs Division case
IAD552-A - Supplemental report for IAD552
IAD552-B - Supplemental report for IAD552
</correct_output>

<explanation>
Extracted all five identifiers including the main case and its supplemental reports.
</explanation>
</example_3>

<example_4>
<input>
For reference, see Form #XXXXX or contact the records department with your case number.
</input>

<correct_output>
No case identifiers on this page
</correct_output>

<explanation>
"Form #XXXXX" is a placeholder, not an actual identifier. No specific case identifiers provided.
</explanation>
</example_4>

<example_5>
<input>
The complaint was filed under Complaint #2023-C-0445. The officer-involved shooting is documented in Incident Report #2023-IR-0891, with follow-up investigation IA-2023-067. Related court case is Docket #CV-2024-00123.
</input>

<correct_output>
Complaint #2023-C-0445 - Filed complaint
Incident Report #2023-IR-0891 - Officer-involved shooting report
IA-2023-067 - Follow-up investigation number
Docket #CV-2024-00123 - Related court case number
</correct_output>

<explanation>
Extracted all four identifiers with clear contexts showing relationships between the case numbers.
</explanation>
</example_5>

<example_6>
<input>
Several cases are pending review. Multiple incident reports have been filed. The investigation is ongoing.
</input>

<correct_output>
No case identifiers on this page
</correct_output>

<explanation>
References to cases and reports but no specific identifiers provided.
</explanation>
</example_6>

<example_7>
<input>
CASE FILE SUMMARY
Primary Case: IA2018-17
Cross-Reference: 2018-17, Case #2018-0017
Incident Date: March 15, 2018
Report Numbers: R-2018-0234, R-2018-0235
</input>

<correct_output>
IA2018-17 - Primary Internal Affairs case
2018-17 - Cross-reference identifier for same case
Case #2018-0017 - Alternative case number reference
R-2018-0234 - Report number
R-2018-0235 - Report number
</correct_output>

<explanation>
Extracted all five identifiers, noting that the first three refer to the same case with different formatting.
</explanation>
</example_7>

<example_8>
<input>
Use of Force Review Board
Date: June 10, 2023
Case: UOF-2023-045

The board reviewed the incident originally documented in CAD #230515-0123 and investigated under OIA-2023-089. The officer's previous disciplinary history includes Case IA-2021-034 and Case IA-2022-015.
</input>

<correct_output>
UOF-2023-045 - Use of force case under review
CAD #230515-0123 - Computer Aided Dispatch number for original incident
OIA-2023-089 - Office of Internal Affairs investigation number
IA-2021-034 - Prior disciplinary case
IA-2022-015 - Prior disciplinary case
</correct_output>

<explanation>
Extracted all five identifiers including current case and historical references.
</explanation>
</example_8>

<example_9>
<input>
Investigation Case Number: TBD
Temporary ID: PENDING
Complaint received but not yet assigned a case number.
</input>

<correct_output>
No case identifiers on this page
</correct_output>

<explanation>
"TBD" and "PENDING" are placeholders, not actual assigned identifiers.
</explanation>
</example_9>

<example_10>
<input>
DEPARTMENTAL NOTICE
Re: Investigation N-BPH-284-18-A

This notice pertains to the investigation filed as N-BPH-284-18-A, related to Incident #2018-3421 which occurred on October 15, 2018. Previous notices regarding this matter referenced File #BPH-284.
</input>

<correct_output>
N-BPH-284-18-A - Investigation case number
Incident #2018-3421 - Related incident number
File #BPH-284 - Previous file reference for same matter
</correct_output>

<explanation>
Extracted all three identifiers, noting the relationship between the full case number and abbreviated file reference.
</explanation>
</example_10>
</few_shot_examples>

<thinking_process>
Before extracting case identifiers, ask yourself:
1. Is this a specific identifier or a placeholder/template reference?
2. What is the complete format of the identifier (include all prefixes, numbers, suffixes)?
3. What does this identifier reference (incident, investigation, complaint, report)?
4. Are there multiple identifiers that reference the same case?
5. Is there enough specific information to extract this identifier?
</thinking_process>

<critical_instructions>
1. NEVER include placeholder identifiers like "Case #XXXXX", "TBD", "PENDING", or generic examples.
2. ONLY include specific, assigned case identifiers with actual values.
3. If there are no case identifiers on the Current Page, return "No case identifiers on this page".
4. DO NOT infer, assume, or hallucinate any identifiers not explicitly stated in the provided text.
5. Treat this task as a binary classification: either there are case identifiers to extract, or there aren't.
6. Always include the complete identifier format - don't abbreviate or truncate.
7. Provide context for what each identifier references.
</critical_instructions>

<output_format>
Present the case identifiers in a clear format:

Identifier Format - Context/what it identifies

Examples:
IA-2014-023 - Internal Affairs investigation case number
Incident #2014-0567 - Initial incident report number
H-OIA-097-20-A - Office of Internal Affairs case
Complaint #2023-C-0445 - Filed complaint number
Report #2020-UOF-034 - Use of force report
Case #12345 - Previous investigation reference
CAD #230515-0123 - Computer Aided Dispatch call number
Docket #CV-2024-00123 - Related court case

If multiple identifiers reference the same case, note the relationship:
IA2018-17 - Primary Internal Affairs case
2018-17 - Cross-reference for IA2018-17
Case #2018-0017 - Alternative reference for IA2018-17
</output_format>

<warnings>
- Do not include placeholder identifiers, templates, or examples
- Do not include references to identifiers without providing the actual identifier
- Avoid truncating identifier formats - include complete format as stated
- Do not draw conclusions not explicitly stated in the text
- If an identifier appears in headers, footers, or page numbers without substantive context, still extract it (unlike dates/names where we exclude headers)
- If there are no specific identifiers, return "No case identifiers on this page"
</warnings>

<reference_materials>
Current Page:
{{current_page}}
</reference_materials>

<output_instruction>
Extract all case identifier information from the page following the format specified above:
</output_instruction>
"""

page_summary_verification_template = """
You are a Legal Document Verifier. Your task is to verify case identifier extractions against the original document.

<task_description>
Review the Current Case Identifier Extraction and verify:
1. Every identifier listed appears in the Original Document exactly as stated
2. All case identifiers mentioned in the Original Document are included
3. Identifier formats are accurate and complete
4. No placeholder identifiers (XXXXX, TBD, PENDING) are included
5. Context for each identifier is accurate

Return either the original extraction (if correct) or a corrected version.
</task_description>

<critical_rules>
- Only include identifiers explicitly stated in the Original Document
- Identifier formats must match exactly (including prefixes, numbers, suffixes)
- Exclude placeholder identifiers, templates, or generic examples
- Include identifiers from any part of document (headers, body text, footers) if they are specific values
- If no identifiers are mentioned, return: "No case identifiers on this page"
- Return ONLY the extraction - no commentary or explanations
</critical_rules>

<output_format>
Return one of:
1. The Current Case Identifier Extraction exactly as written (if completely accurate)
2. A corrected extraction in the same format:

Identifier Format - Context/what it identifies

3. "No case identifiers on this page" (if no identifiers are mentioned)
</output_format>

<reference_documents>
Original Document:
{{original_document}}

Current Case Identifier Extraction:
{{current_summary}}
</reference_documents>

<output_instruction>
Verify the Current Case Identifier Extraction against the Original Document. Return your output below:
</output_instruction>
"""

combine_template = """
You are a Legal Clerk merging case identifier extractions into a single comprehensive list.

<task_description>
Merge two case identifier extractions by:
1. Combining all unique identifiers
2. Merging duplicate identifiers that reference the same case
3. Resolving contradictions based on context, or dropping both if unclear
4. Maintaining only specific, assigned identifiers (no placeholders)
</task_description>

<critical_rules>
- Every identifier must appear in at least one source extraction
- When same identifier appears twice: merge into one entry with combined context
- When multiple identifiers reference same case: keep all with note about relationship
- When contradictory information exists with clear context: keep correct one
- When contradictory information lacks context: DROP BOTH entries
- Exclude placeholder identifiers (XXXXX, TBD, PENDING)
- Only include specific, assigned case identifiers
- If no identifiers exist, return: "No case identifiers on this page"
- Precision over completeness - exclude questionable entries
</critical_rules>

<output_format>
Return identifiers in a clear format:

Identifier Format - Context/what it identifies

Or return: "No case identifiers on this page"
</output_format>

<reference_materials>
Case Identifier Extraction 1:
{{summary_1}}

Case Identifier Extraction 2:
{{summary_2}}
</reference_materials>

<output_instruction>
Merge the two case identifier extractions into a single comprehensive list:
</output_instruction>
"""

verification_template = """
You are verifying a combined case identifier extraction against its two source extractions.

<task_description>
Verify that the combined extraction:
1. Contains only identifiers from the source extractions (no hallucinations)
2. Has no unnecessary duplicate identifiers
3. Properly handled contradictions (kept correct one or dropped both)
4. Only includes specific, assigned identifiers (no placeholders)
5. Includes all identifiers from source extractions
6. Accurately preserved identifier formats

Return the original if correct, or a corrected version.
</task_description>

<critical_rules>
- Every identifier must appear in at least one source extraction
- Same identifier appearing twice should be merged unless referring to different contexts
- Contradictions: if context was clear, one entry kept; if unclear, both dropped
- Exclude placeholder identifiers (XXXXX, TBD, PENDING)
- Only include specific, assigned case identifiers
- If no valid identifiers, return: "No case identifiers on this page"
- Return ONLY the extraction - no commentary
</critical_rules>

<output_format>
Return one of:
1. The combined extraction exactly as written (if completely accurate)
2. A corrected extraction:

Identifier Format - Context/what it identifies

3. "No case identifiers on this page" (if no valid identifiers)
</output_format>

<reference_materials>
Case Identifier Extraction 1:
{{summary_1}}

Case Identifier Extraction 2:
{{summary_2}}

Current Combined Case Identifier Extraction:
{{current_combined_summary}}
</reference_materials>

<output_instruction>
Verify the combined extraction against the source extractions. Return your output below:
</output_instruction>
"""

condense_interval_template = """
<task_description>
You are reviewing a comprehensive interval summary that contains case identifier information. Condense to the most important identifiers.
</task_description>

<critical_requirements>
1. Return the most important case identifier information
2. Focus on primary case identifiers central to the case (main case numbers, investigation IDs)
</critical_requirements>

<selection_criteria>
Prioritize case identifier information that includes:
- Primary case numbers or incident numbers
- Internal Affairs investigation numbers
- Main complaint numbers
- Use of force report numbers for key incidents
- Court case numbers if applicable

Deprioritize:
- Cross-reference identifiers that duplicate primary IDs
- Supplemental report numbers
- Peripheral or tangentially related case numbers
- Redundant identifier references
</selection_criteria>

<full_interval_summary>
{{full_interval_summary}}
</full_interval_summary>

<output_format>
Return the condensed case identifier information maintaining the original structure:

- Primary Case/Investigation Numbers
  • [Most important identifiers]
  • END BULLETPOINTS

- Incident/Report Numbers
  • [Most important identifiers]
  • END BULLETPOINTS

- Complaint/IA Numbers
  • [Most important identifiers]
  • END BULLETPOINTS
</output_format>

<output_instruction>
Extract and return the most important case identifier information now:
</output_instruction>
"""

# Extraction prompts
extract_template = """
<case_id_extraction_prompt>
<context>
Your task is to analyze police documents to identify ALL CASE IDENTIFIERS mentioned in the document. Focus on extracting:
1. Case numbers and incident numbers
2. Internal Affairs (IA) case numbers
3. Report numbers (incident reports, use of force reports, etc.)
4. Complaint numbers or complaint IDs
5. Investigation tracking numbers
6. Court case numbers or docket numbers

You must carefully extract specific identifiers while excluding placeholders and generic references.

Key definitions:
- CASE IDENTIFIERS: Specific assigned numbers or alphanumeric codes that uniquely identify cases, incidents, investigations, complaints, or reports
- NOT identifiers: Placeholders (XXXXX, TBD, PENDING), generic references ("your case number"), template examples

Extract comprehensive information for each identifier including:
- Complete identifier format (include all prefixes, numbers, suffixes)
- Context (what it identifies - incident, investigation, complaint, report, etc.)
- Relationships to other identifiers if mentioned
</context>

<format_requirements>
- Each identifier entry should include: Complete Format - Context/what it identifies
- Group identifiers by type (case numbers, IA numbers, report numbers, etc.)
- Only include specific, assigned identifiers (no placeholders or templates)
- Note relationships between identifiers when mentioned (cross-references, related cases)
- If no identifiers are mentioned, report that no case identifiers could be identified
</format_requirements>

<document_for_review>
{{source_text}}
</document_for_review>

<thinking_process>
1. First, scan the document for all alphanumeric identifiers
2. For each potential identifier, determine:
   a) Is this a specific assigned identifier or a placeholder?
   b) What is the complete format?
   c) What does it identify (case, incident, investigation, complaint, report)?
3. Extract all specific identifiers with complete formats
4. Note any cross-references or relationships between identifiers
5. Exclude placeholders, templates, and generic references
6. Group identifiers by type for clarity
7. Ensure no duplicates unless they represent different contexts
8. Verify all identifiers are explicitly stated in the document
</thinking_process>

<verification_steps>
1. Re-examine each extracted identifier
2. Confirm the identifier format is complete and accurate
3. Verify it's a specific assigned identifier, not a placeholder
4. Check that the context/description is accurate
5. Ensure no identifiers were missed
6. Verify relationships between identifiers are correctly noted
7. Confirm all identifiers are explicitly stated in the document text
</verification_steps>

<output_format>
PRIMARY CASE/INCIDENT NUMBERS:
[List main case numbers and incident numbers with contexts]

INTERNAL AFFAIRS IDENTIFIERS:
[List IA case numbers, OIA numbers, investigation tracking numbers]

COMPLAINT NUMBERS:
[List complaint numbers and complaint IDs]

REPORT NUMBERS:
[List police reports, use of force reports, supplemental reports]

COURT CASE NUMBERS:
[List docket numbers, court case numbers if applicable]

OTHER IDENTIFIERS:
[List file numbers, CAD numbers, or other tracking identifiers]

For each identifier, use format:
Identifier Format - Specific context/what it identifies

If no identifiers found in a category, state "None identified"

CONFIDENCE LEVEL: [High/Medium/Low]

NOTES:
[Any relevant clarifications about identifier relationships, cross-references, or ambiguities]
</output_format>
</case_id_extraction_prompt>
"""

extract_verification_template = """
<case_id_extraction_verification_prompt>
<context>
Your task is to verify the accuracy of the extracted case identifier list from police documentation. You must distinguish between:
- SPECIFIC IDENTIFIERS: Assigned numbers or codes that uniquely identify cases, incidents, or reports
- PLACEHOLDERS: Generic references like XXXXX, TBD, PENDING, or template examples

When contradictory information appears, determine which is most reliable based on context and specificity.
</context>

<proposed_identifier_list>
{{initial_dates}}
</proposed_identifier_list>

<document_for_review>
{{source_text}}
</document_for_review>

<verification_process>
1. Carefully re-read the entire document
2. Identify ALL alphanumeric identifiers mentioned
3. For each identifier in the proposed list, verify:
   a) It appears in the document with the exact format shown
   b) It's a specific assigned identifier, not a placeholder
   c) The context/description is accurate
   d) The complete format is captured (no truncation)
4. Check for identifiers mentioned in the document but missing from the list
5. Check for placeholders or templates incorrectly included in the list
6. Verify relationships between identifiers are correctly noted
7. Ensure all identifier formats are complete and accurate
</verification_process>

<critical_considerations>
- Only include specific, assigned identifiers
- Exclude ALL placeholders (XXXXX, TBD, PENDING, etc.)
- Exclude generic references ("your case number") without specific values
- When the same case has multiple identifier formats, include all with relationship notes
- If contradictory information exists, select the most reliable based on context
- Identifier formats must be complete and accurate
</critical_considerations>

<output_format>
VERIFICATION RESULT: [CONFIRMED / CORRECTED / REJECTED]

PRIMARY CASE/INCIDENT NUMBERS:
[List with Complete Format - Context]

INTERNAL AFFAIRS IDENTIFIERS:
[List with Complete Format - Context]

COMPLAINT NUMBERS:
[List with Complete Format - Context]

REPORT NUMBERS:
[List with Complete Format - Context]

COURT CASE NUMBERS:
[List with Complete Format - Context]

OTHER IDENTIFIERS:
[List with Complete Format - Context]

CONFIDENCE LEVEL: [High/Medium/Low]

JUSTIFICATION:
[Detailed explanation of verification, including any corrections made and why]

KEY EVIDENCE:
[Direct quotes from the document that show the identifiers]

CORRECTIONS MADE (if applicable):
[List of identifiers added, removed, or modified with explanations]
</output_format>
</case_id_extraction_verification_prompt>
"""

format_conversion_template = """If case identifiers are stated, return them in a standardized format.

For each identifier, provide:
1. The identifier in its original format
2. A standardized version (if applicable) with consistent formatting

If there are multiple case identifiers, list them all separated by semicolons.
If no case identifiers are stated, return None.

Standardization rules:
- Remove extra spaces
- Preserve all alphanumeric characters, hyphens, and #symbols
- Keep prefixes (IA-, Case #, Report #, etc.)
- Do not alter the core identifier structure

Valid responses include:
IA-2020-045
Case #12345; Report #2020-0567
H-OIA-097-20-A
None
IA2018-17; 2018-17; Case #2018-0017
Incident #2014-0567; IA-2014-023
None

Below is the document that you are tasked with reviewing:

--------------------

{{ source_text }}

--------------------
"""

# Citation prompts
primary_citation_template = """
<objective>
Analyze this page of text to determine if it contains citations that mention SPECIFIC CASE IDENTIFIERS identified in the case ID extraction analysis.
</objective>

<case_id_extraction_context>
The case ID extraction analysis identified the following:

EXTRACTED CASE IDENTIFIERS:
{{incident_date}}

{{date_extraction_context}}

You should look for citations that mention these SPECIFIC case identifiers or related tracking numbers.
</case_id_extraction_context>

<case_id_citation_definition>
CASE ID CITATIONS must clearly reference specific case identifiers:
- Direct mentions: "Case IA-2014-023 was investigated"
- Identifier references: "Incident #2014-0567 occurred on January 26"
- Report number mentions: "Report #2020-UOF-034 documents the use of force"
- Cross-references: "See also Case #12345 for related matter"

NOT case ID citations:
- Vague references like "the case" or "this investigation" without specific identifiers
- Placeholder identifiers (XXXXX, TBD, PENDING)
- Generic references ("your case number") without specific values
</case_id_citation_definition>

<targeted_search_guidance>
Based on the case ID extraction context above, prioritize finding citations that:
1. Mention specific identifiers from the extraction list
2. Reference the exact identifier format (IA-2014-023, not just "2014-023")
3. Provide context about what the identifier references
4. Show relationships between different identifiers
</targeted_search_guidance>

<page_text>
{{page_text}}
</page_text>

<output_instructions>
Focus on finding citations that specifically mention the CASE IDENTIFIERS identified in the extraction analysis.
Respond with 'YES' or 'NO' followed by your reasoning and an exact quote.

Format:
[YES/NO]: [Your reasoning explaining whether this page contains citations mentioning the SPECIFIC case identifiers]
Quote: "[Exact text from the page that supports your decision]"
</output_instructions>
"""

validator_citation_template = """
<objective>
Verify whether this page contains a HIGH-QUALITY citation that identifies a specific case identifier.
You must distinguish between specific identifiers and placeholders/generic references.
</objective>

<extracted_case_identifiers>
{{incident_date}}
</extracted_case_identifiers>

<case_id_citation_definition>
CASE ID CITATIONS must clearly state a SPECIFIC IDENTIFIER:
- Specific mentions: "Case IA-2014-023"
- Identifier with context: "Incident #2014-0567 occurred on..."
- Report references: "Report #2020-UOF-034"
- Complete format citations: "Investigation H-OIA-097-20-A"

EXPLICITLY NOT case ID citations:
- Vague references ("the case," "this matter") without specific identifiers
- Placeholder identifiers (XXXXX, TBD, PENDING)
- Generic references ("your case number," "see case file")
- Partial identifiers without complete format
</case_id_citation_definition>

<primary_analysis>
{{primary_analysis}}
</primary_analysis>

<page_text>
{{page_text}}
</page_text>

<validation_criteria>
For a citation to be valid:
1. It must mention a specific case identifier with complete format
2. The identifier must be an assigned value, not a placeholder
3. The quote must be an exact excerpt from the page text
4. The identifier should match or relate to those in the extraction list

Ask yourself:
- Is this a specific assigned identifier or a placeholder/generic reference?
- Is the complete identifier format provided?
- Does this identifier appear in the extracted list or clearly relate to the case?
</validation_criteria>

<output_format>
Return your analysis as a JSON object with the following structure:
{
    "final_decision": "YES" or "NO",
    "validator_reasoning": "Detailed explanation of why this is or is not a valid case ID citation",
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
