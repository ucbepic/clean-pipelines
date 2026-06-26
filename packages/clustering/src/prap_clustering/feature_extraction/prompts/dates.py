"""
Date extraction prompts for incident date feature.

This module contains all 12 prompts needed for date extraction:
- Summarization stage: 7 prompts
- Extraction stage: 3 prompts
- Citation finding: 2 prompts
"""

# Copied from incident_date_extraction/summarize/src/prompts.py
memory_log_template = """
As a Legal Clerk, your task is to review the new summary and update the memory log only when the new summary contains crucial information directly related to the main subject of the document. Maintain a concise memory log that focuses on the key aspects of the events, allegations, investigations, and outcomes described in the document.
</task_description>

<guidelines>
1. Review and Compare:
   • Carefully review the current memory log and the new summary.
   • Determine if the new summary contains crucial information that is not already in the memory log.

2. Identify Crucial Information:
   • Focus on information specific to the main subject of the document.
   • Look for key details related to events, allegations, investigations, and outcomes.

3. Update Selectively:
   • Only update the memory log if the new summary contains crucial information not already present.
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
Ensure the summary includes ALL of the following elements, if present. First and foremost, your objective is to return a comprehensive summary that will provide the user with a thorough understanding of the contents of the summary.

Some essential information that will contribute to a comprehensive summary include but are not limited to:
a. Type and purpose of the legal document (e.g., police report, internal investigation)
b. Primary parties involved (full names, roles, badge numbers if applicable)
j. Allegations of misconduct and any associated information
c. Key legal issues, claims, or charges
k. Disciplinary outcomes or their current status
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements)

For each type of essential information classification, be specific when referring to people, places, and dates.
</essential_information>

<thinking_process>
Before updating the memory log, consider:
1. Does the new summary contain any crucial information not already in the memory log?
2. How does this new information relate to the main subject of the document?
3. Can this new information be integrated into the existing log without disrupting its flow?
4. Is this information essential to understanding the key aspects of the case?
5. Am I maintaining the conciseness of the log while including all crucial details?
</thinking_process>

<warnings>
- Do not add information that is not directly stated in the document
- Avoid speculation or inference beyond what is explicitly mentioned
- Do not remove or alter existing crucial information in the memory log
- Ensure that any updates maintain the chronological and logical flow of events
- Be cautious of potential inconsistencies between the new summary and existing log
</warnings>

<reference_materials>
## Original Memory Log ##
{{ memory_log }}

## New Summary ##
{{ summary }}
</reference_materials>

<output_instruction>
Based on your review of the current memory log and the new summary, provide either an updated memory log incorporating the crucial new information, or reproduce the original memory log if no update is necessary. Ensure the output maintains a concise focus on key aspects of events, allegations, investigations, and outcomes related to the main subject of the document:
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
Your task is to generate a comprehensive, bulletpoint summary of all the important information contained in the provided document excerpt. Extract all the key details presented in the current page.
</task_description>


<legal_document_essential_information>
If the document is classified as a legal document, ensure the summary includes ALL of the following elements, if present:

b. Primary parties involved (full names, roles, badge numbers if applicable)
c. Key legal issues, claims, or charges
d. Critical events or incidents (with specific dates, times and locations)
e. Main findings or decisions
f. Significant evidence or testimonies
g. Important outcomes or rulings
h. Current status of the matter
i. Any pending actions or future proceedings
j. Allegations of misconduct and any associated information
k. Disciplinary outcomes or their current status
l. Procedural events (e.g., filing of charges, hearings, notifications, motions, investigations, agreements, service of documents, compliance with legal requirements)
For each type of essential information, be specific when referring to people, places, and dates.
</legal_document_essential_information>


<other_document_type_guidelines>
If classified as Other Document Type, follow these guidelines:

Ensure the summary includes the following elements, if present:
b. Main topics or themes
c. Key individuals or organizations mentioned
d. Important dates, events, or milestones
e. Significant data or findings
f. Main arguments or conclusions
g. Any recommendations or future actions proposed
h. Relevant background information or context

</other_document_type_guidelines>

<critical_instructions>
1. NEVER include any information about "John Doe," "Jane Doe," or other anonymous/ambiguous entities (e.g., "ABC Corporation," "Company X," "Individual A") in your summary. If specific identifying information is not available in the document, acknowledge this limitation by stating that details are unavailable or redacted, rather than including placeholder names or making assumptions about identities.
2. If there is no relevant content to summarize in the Current Page, return an empty string ("").
3. DO NOT infer, assume, or hallucinate any information not explicitly stated in the provided text.
4. Treat this task as a binary classification: either there is relevant information to summarize, or there isn't.
</critical_instructions>

<thinking_process>
Before summarizing, consider:
1. Is this a legal document or another document type?
2. What are the main topics on this page?
3. What are the most important pieces of information to extract based on the document type?
</thinking_process>

<output_format>
Present the summary using the following structure:

- Main events
  • Sub-event 2.1
  • Sub-event 2.2

- Main actions
  • Sub-action 4.1
  • Sub-action 4.2

- Main legal issue
  • Sub-legal ossue 5.1
  • Sub-legal issue 5.2

- Main legal procedure
  • Sub-legal procedure 6.1
  • Sub-legal procedure 6.2

- Main allegations
  • Sub-allegation 7.1
  • Sub-allegation  7.2

- Main disiplinary outcomes
  • Sub-disciplinary outcome 8.1
  • Sub-disciplinary outcome 8.2

</output_format>

<warnings>
- Do not include speculative information
- Avoid summarizing irrelevant details
- Do not draw conclusions not explicitly stated in the text
</warnings>

<reference_materials>
Current Page:
{{current_page}}
</reference_materials>

<output_instruction>
First, state the document classification (Legal Document or Other Document Type) and provide a brief explanation for your decision. Then, generate the current page summary following the appropriate guidelines based on the classification.
</output_instruction>
"""

# This is a highly specialized prompt for date extraction from incident_date_extraction/summarize/src/prompts.py
page_summary_template = """
<task_description>
As a Legal Clerk, your task is to identify and extract all critical dates from the provided document excerpt. Focus on dates that are integral to understanding the timeline of the misconduct case, from incident to resolution. Use the context from the memory log and surrounding pages when necessary for clarity or relevance.
</task_description>

<guidelines>
1. Extract all critical dates from the current page that meet the selection criteria below.
2. Support the extracted dates with additional context from the memory log and surrounding pages to enhance understanding of their significance.
3. Use the memory log to help you understand which dates are relevant and which are merely administrative.
4. DO NOT include any dates not explicitly stated in any of the documents.
5. Present the dates in chronological order with clear descriptions of their significance.
6. If someone's identity is ambiguous, refer to them as "unidentified person".
7. If the significance of a date cannot be determined with confidence, omit it from your summary.
</guidelines>

<date_extraction_priority>
Extract and highlight the following critical dates when present:

1. Incident dates: When the alleged misconduct or use of force occurred
   - Date and time of the primary incident
   - Dates of related or follow-up incidents

2. Reporting dates: When complaints were filed or incidents were reported
   - When civilian complaints were filed
   - When internal reports were submitted
   - When supervisors were notified

3. Investigation milestones:
   - Investigation start date
   - Investigation completion date
   - Key interview dates (officers, witnesses, complainants)
   - Evidence collection dates (body camera footage reviewed, forensic analysis conducted)

4. Procedural dates:
   - Charging or disciplinary action filing dates
   - Notice of charges served
   - Hearing or trial dates
   - Motion filing dates
   - Decision or ruling dates
   - Service of documents dates

5. Outcome dates:
   - Final disposition dates
   - Discipline imposed dates
   - Appeal filing dates
   - Appeal deadlines
   - Case closure dates

SELECTIVITY RULE: Only include dates that materially advance understanding of the case timeline.
Omit routine administrative dates (e.g., document receipt acknowledgments, standard filing stamps, routine file transfers) unless they reveal procedural delays or compliance issues relevant to the misconduct allegation.
</date_extraction_priority>


<few_shot_examples>
These examples demonstrate correct date extraction behavior:

<example_1>
<input>
On January 26, 2014, at approximately 6:56 a.m., a San Diego police officer stopped Devenere's truck for a traffic violation. Between 6:56 a.m. and 7:37 a.m., Devenere took his girlfriend hostage and led police on a pursuit. At 7:37 a.m., the truck stopped and Officer Butera fired at Devenere.
</input>

<correct_output>
January 26, 2014, 6:56 a.m. - Incident Date: San Diego police officer stopped Devenere's truck for a traffic violation
January 26, 2014, 6:56 a.m. - 7:37 a.m. - Incident Period: Devenere took girlfriend hostage and led police on pursuit
January 26, 2014, 7:37 a.m. - Incident Date: Truck stopped and Officer Butera fired at Devenere
</correct_output>

<explanation>
Multiple related incident dates and times extracted with clear descriptions of what occurred at each point.
</explanation>
</example_1>

<example_2>
<input>
The investigation was initiated on February 12, 2023 following the incident. Investigator Martinez conducted interviews with three witnesses between February 15-17, 2023. Body camera footage was reviewed on February 20, 2023. The investigation concluded on March 5, 2023.
</input>

<correct_output>
February 12, 2023 - Investigation Milestone: Investigation initiated
February 15-17, 2023 - Investigation Milestone: Investigator Martinez conducted interviews with three witnesses
February 20, 2023 - Investigation Milestone: Body camera footage reviewed
March 5, 2023 - Investigation Milestone: Investigation concluded
</correct_output>

<explanation>
Complete investigation timeline extracted with all key milestones.
</explanation>
</example_2>

<example_3>
<input>
File reviewed on July 10, 2023 by records clerk. Document copied on July 12, 2023 for archives. Routine monthly audit completed August 1, 2023 with no findings. Standard compliance check performed August 15, 2023.
</input>

<correct_output>
No dates on this page
</correct_output>

<explanation>
All dates are purely administrative (file reviews, copying, routine audits) with no substantive case developments. None advance understanding of the misconduct investigation timeline.
</explanation>
</example_3>

<example_4>
<input>
The disciplinary hearing was scheduled for June 15, 2023 but was postponed. The hearing ultimately took place on July 10, 2023. The review board deliberated and issued its decision on July 18, 2023, suspending Officer Thompson for 30 days without pay effective August 1, 2023.
</input>

<correct_output>
July 10, 2023 - Procedural Date: Disciplinary hearing held (originally scheduled for June 15, 2023)
July 18, 2023 - Outcome Date: Review board issued decision, suspending Officer Thompson for 30 days without pay
August 1, 2023 - Outcome Date: Suspension effective date
</correct_output>

<explanation>
Procedural dates including rescheduling context, decision date, and when discipline takes effect.
</explanation>
</example_4>

<example_5>
<input>
Officer Martinez filed an appeal on September 5, 2023. The appeal deadline was September 30, 2023. The appeal hearing was held on October 20, 2023. The appeals board upheld the original decision on November 2, 2023, and the case was officially closed on November 15, 2023.
</input>

<correct_output>
September 5, 2023 - Outcome Date: Officer Martinez filed appeal
September 30, 2023 - Procedural Date: Appeal filing deadline
October 20, 2023 - Procedural Date: Appeal hearing held
November 2, 2023 - Outcome Date: Appeals board upheld original decision
November 15, 2023 - Outcome Date: Case officially closed
</correct_output>

<explanation>
Complete appeals process timeline from filing through final resolution.
</explanation>
</example_5>

<example_6>
<input>
The witness stated they saw the incident but could not recall the exact date. The officer's report mentioned "approximately mid-April" but no specific date was documented. The file was transferred between departments on May 5, 2023 for administrative processing. Weekly status meetings were held throughout June 2023.
</input>

<correct_output>
No dates on this page
</correct_output>

<explanation>
No specific dates are provided for the incident. "Mid-April" is too vague. The May 5 transfer date is purely administrative and doesn't advance case understanding. Weekly meetings without substantive developments are routine and should be excluded.
</explanation>
</example_6>

<example_7>
<input>
MEMORANDUM
To: Captain Johnson
From: Sergeant Williams
Date: March 15, 2023
Re: Use of Force Investigation

The incident occurred on February 10, 2023 at 2:30 PM when Officer Davis responded to a disturbance call. The investigation was initiated on February 12, 2023.
</input>

<correct_output>
February 10, 2023, 2:30 PM - Incident Date: Officer Davis responded to a disturbance call
February 12, 2023 - Investigation Milestone: Investigation initiated
</correct_output>

<explanation>
The date "March 15, 2023" in the memo header is document metadata, not a case event. Only dates from the body text (February 10 and February 12) are extracted.
</explanation>
</example_7>

<example_8>
<input>
Internal Affairs Report
Case #2023-045
Generated: April 30, 2023
Page 3 of 15

Body camera footage was reviewed on April 15, 2023 by Investigator Martinez. The footage showed the altercation that took place on March 8, 2023.
</input>

<correct_output>
March 8, 2023 - Incident Date: Altercation captured on body camera footage
April 15, 2023 - Investigation Milestone: Body camera footage reviewed by Investigator Martinez
</correct_output>

<explanation>
"April 30, 2023" is a report generation date (metadata) and "Page 3 of 15" is formatting. Only dates describing case events are extracted.
</explanation>
</example_8>

<example_9>
<input>
The complaint was filed on May 5, 2023. Officer Thompson was interviewed on May 20, 2023. The witness, John Doe, provided testimony on May 22, 2023.

Sincerely,
Chief of Police
June 1, 2023
</input>

<correct_output>
May 5, 2023 - Reporting Date: Complaint filed
May 20, 2023 - Investigation Milestone: Officer Thompson interviewed
</correct_output>

<explanation>
Three dates in body text, but "May 22, 2023" is excluded because it references "John Doe" (placeholder name). "June 1, 2023" in signature block is formatting, not extracted.
</explanation>
</example_9>

<example_10>
<input>
OFFICE OF THE DISTRICT ATTORNEY
January 15, 2024

The disciplinary hearing was held on December 10, 2023. Following deliberation, the review board issued its decision on December 15, 2023, suspending Officer Garcia for 15 days without pay.
</input>

<correct_output>
December 10, 2023 - Procedural Date: Disciplinary hearing held
December 15, 2023 - Outcome Date: Review board issued decision, suspending Officer Garcia for 15 days without pay
</correct_output>

<explanation>
"January 15, 2024" is letterhead dating. Only dates from the substantive text describing hearing and decision are extracted.
</explanation>
</example_10>
</few_shot_examples>

<thinking_process>
Before extracting a date, ask yourself:
1. Does this date appear in the substantive content of the document, or is it just part of the formatting/metadata?
2. Does this date mark a significant event in the case timeline?
3. Would removing this date cause the user to lose understanding of how the case progressed?
4. Does this date relate directly to the incident, investigation, or resolution?
5. Does this date reveal important timeline gaps, delays, or procedural milestones?
6. Is this date merely administrative, or does it have substantive significance?

CRITICAL: If a date appears in a header, footer, letterhead, or signature block, it is NOT part of the substantive content and should NOT be extracted.

When in doubt, ask: Does this date help establish what happened, when it happened, or how the case progressed through the system? If not, omit it.
</thinking_process>

<critical_instructions>
1. NEVER include any information about "John Doe," "Jane Doe," or other anonymous/ambiguous entities (e.g., "ABC Corporation," "Company X," "Individual A") in your summary. If specific identifying information is not available in the document, acknowledge this limitation by stating that details are unavailable or redacted, rather than including placeholder names or making assumptions about identities.
2. If there is no relevant date information to extract from the Current Page, return an empty string ("").
3. DO NOT infer, assume, or hallucinate any dates not explicitly stated in the provided text.
4. DO NOT extract dates from headers, footers, letterhead, page numbers, or signature blocks.
5. Treat this task as a binary classification: either there are relevant dates to extract, or there aren't.
6. Always provide context for each date—explain what happened on that date and why it's significant to the case.
7. A date must be described or referenced in the body text to be extracted - mere appearance in document formatting is insufficient.
</critical_instructions>

<output_format>
Present the dates in chronological order using the following structure:

[Date] - [Event Category]: [Clear description of what occurred and its significance]

Example:
March 15, 2023, 10:30 PM - Incident Date: Officer Smith allegedly used excessive force during arrest of suspect at 123 Main Street
March 18, 2023 - Reporting Date: Civilian witness filed formal complaint with Internal Affairs
April 2, 2023 - Investigation Milestone: Internal Affairs investigation formally opened, case assigned to Investigator Johnson
June 30, 2023 - Investigation Milestone: Investigation concluded, final report submitted to review board
July 15, 2023 - Procedural Date: Disciplinary hearing held before review board
August 1, 2023 - Outcome Date: Officer Smith suspended for 30 days without pay

If dates span a range, indicate this clearly:
April 2, 2023 - June 30, 2023 - Investigation Period: Internal Affairs conducted investigation including 12 witness interviews and review of body camera footage
</output_format>

<warnings>
- Do not include speculative dates
- Avoid extracting administrative dates that don't advance case understanding
- Do not draw conclusions about dates not explicitly stated in the text
- Do not include dates that merely document routine file management
- Do not extract dates from document headers, footers, letterhead, or signature blocks
- If there are no dates in the substantive content, return "No dates on this page"
</warnings>

<reference_materials>
Current Page:
{{current_page}}
</reference_materials>

<output_instruction>
Generate the chronological date extraction below:
</output_instruction>
"""

page_summary_verification_template = """
You are a Legal Document Verifier. Your task is to verify date extractions against the original document.

<task_description>
Review the Current Date Extraction and verify:
1. Every date appears in the Original Document's substantive text (not headers/footers)
2. All important dates from the Original Document are included
3. Dates are relevant to case progression (not purely administrative)
4. Descriptions are accurate

Return either the original extraction (if correct) or a corrected version.
</task_description>

<critical_rules>
- Only include dates explicitly stated in the Original Document's body text
- Exclude dates from headers, footers, letterhead, and signature blocks
- Exclude routine administrative dates (file transfers, photocopies, routine meetings)
- Use "unidentified person" for unclear identities
- Never include "John Doe," "Jane Doe," or placeholder names
- If no valid dates exist, return: "No dates on this page"
- Return ONLY the extraction - no commentary or explanations
</critical_rules>

<output_format>
Return one of:
1. The Current Date Extraction exactly as written (if completely accurate)
2. A corrected extraction in chronological order:

[Date] - [Event Category]: [Description]

3. "No dates on this page" (if no valid dates exist)
</output_format>

<reference_documents>
Original Document:
{{original_document}}

Current Date Extraction:
{{current_summary}}
</reference_documents>

<output_instruction>
Verify the Current Date Extraction against the Original Document. Return your output below:
</output_instruction>
"""

combine_template = """
You are a Legal Clerk merging date extractions into a single chronological list.

<task_description>
Merge two date extractions by:
1. Combining all unique dates in chronological order
2. Merging duplicate dates with complementary descriptions
3. Resolving contradictions based on context, or dropping both if unclear
4. Maintaining only case-relevant dates
</task_description>

<critical_rules>
- Every date must appear in at least one source extraction
- When same date appears twice with compatible info: merge into one entry
- When same event has contradictory dates with clear context: keep correct one
- When contradictory dates lack context: DROP BOTH dates
- Exclude "John Doe," "Jane Doe," and placeholder names
- Use "unidentified person" for unclear identities
- If no dates exist, return: "No dates on this page"
- Precision over completeness - exclude questionable dates
</critical_rules>

<output_format>
Return dates in chronological order:

[Date] - [Event Category]: [Description]

Or return: "No dates on this page"
</output_format>

<reference_materials>
Date Extraction 1:
{{summary_1}}

Date Extraction 2:
{{summary_2}}
</reference_materials>

<output_instruction>
Merge the two date extractions into a single chronological list:
</output_instruction>
"""

verification_template = """
You are verifying a combined date extraction against its two source extractions.

<task_description>
Verify that the combined extraction:
1. Contains only dates from the source extractions (no hallucinations)
2. Has no duplicate dates
3. Is in strict chronological order
4. Properly handled contradictions (kept correct one or dropped both)
5. Excludes administrative dates
6. Uses correct format

Return the original if correct, or a corrected version.
</task_description>

<critical_rules>
- Every date must appear in at least one source extraction
- No duplicate dates
- Must be chronologically ordered
- Contradictions: if context was clear, one date kept; if unclear, both dropped
- Exclude "John Doe," "Jane Doe," and placeholder names
- Use "unidentified person" for unclear identities
- Exclude administrative dates (file transfers, photocopies, routine meetings)
- If no valid dates, return: "No dates on this page"
- Return ONLY the extraction - no commentary
</critical_rules>

<output_format>
Return one of:
1. The combined extraction exactly as written (if completely accurate)
2. A corrected extraction in chronological order:

[Date] - [Event Category]: [Description]

3. "No dates on this page" (if no valid dates)
</output_format>

<reference_materials>
Date Extraction 1:
{{summary_1}}

Date Extraction 2:
{{summary_2}}

Current Combined Date Extraction:
{{current_combined_summary}}
</reference_materials>

<output_instruction>
Verify the combined extraction against the source extractions. Return your output below:
</output_instruction>
"""

condense_interval_template = """
<task_description>
You are reviewing a comprehensive interval summary that contains too many bulletpoints.
</task_description>

<critical_requirements>
1. Return EXACTLY 5 bulletpoints total
</critical_requirements>

<selection_criteria>
Prioritize bulletpoints that include:
- Specific names, badge numbers, and roles
- Specific dates, times, and locations
- Key allegations and charges
- Critical findings and decisions
- Important evidence and testimony
- Significant outcomes

Deprioritize:
- Vague or general statements
- Procedural details without substance
- Redundant information
- Minor administrative details
</selection_criteria>

<full_interval_summary>
{{full_interval_summary}}
</full_interval_summary>

<output_format>
Return the condensed summary in this EXACT format:

- Legal Issues, Claims, and Charges
  • [Most important bulletpoint 1]
  • [Most important bulletpoint 2]
  • [Most important bulletpoint 3]
  • END BULLETPOINTS

- Key Events and Incidents
  • [Most important bulletpoint 1]
  • [Most important bulletpoint 2]
  • [Most important bulletpoint 3]
  • [Most important bulletpoint 4]
  • END BULLETPOINTS


- Main Findings, Decisions and Actions
  • [Most important bulletpoint 1]
  • [Most important bulletpoint 2]
  • [Most important bulletpoint 3]
  • END BULLETPOINTS

</output_format>

<output_instruction>
Extract and return the 18 most important bulletpoints now:
</output_instruction>
"""

# Extraction prompts from incident_date_extraction/extract/src/extract.py
extract_template = """
<incident_date_extraction_prompt>
<context>
Your task is to analyze police documents to identify the MOST LIKELY INCIDENT DATE when either:
1. A use of force incident occurred - focusing on when the force was FIRST applied
2. A misconduct incident occurred - focusing on when the misconduct behavior took place

You must carefully distinguish incident dates from administrative dates in police documentation.

Key definitions:
- USE OF FORCE INCIDENT DATE: The specific date when force was first applied by the police officer
- MISCONDUCT INCIDENT DATE: The specific date when the misconduct behavior occurred by the police officer
- NOT incident dates: Report filing dates, investigation dates, witness interview dates, disciplinary hearing dates, or case closure dates

The document may contain conflicting dates or multiple references to the same date. Focus on identifying the SINGLE MOST RELIABLE date based on frequency of mention and strength of supporting context.
</context>

<format_requirements>
- The incident date must include month, day, and year (MM/DD/YYYY or equivalent format)
- For incidents spanning multiple days, identify the START date as the incident date
- For use of force incidents near midnight, the date when force began is the incident date
- Only return multiple dates when:
  a) There are clearly separate, distinct incidents described (not conflicting accounts of the same incident)
  b) Each incident has a different, clearly documented date
- If no clear incident date is present, report that no incident date could be extracted
</format_requirements>

<document_for_review>
{{source_text}}
</document_for_review>

<thinking_process>
1. First, identify all dates mentioned in the document and their frequency
2. For each date, determine its context: incident occurrence, report filing, investigation, etc.
3. Determine the incident type (use of force vs. misconduct) Your goal is to extract the date on which the shooting occurred, the date on which the use of force occurred or the date on which the misconduct occurred
4. For each potential incident date, evaluate:
   a) Frequency of mention in the document
   b) Strength of contextual evidence linking it to the incident
   c) Consistency with other details in the document
5. If there are conflicting dates for the same incident, select the date with the strongest supporting evidence
6. Only identify multiple dates if they clearly refer to separate, distinct incidents
7. For incidents spanning multiple days, identify the START date
8. Verify that the selected date(s) have sufficient context to confirm they are incident dates
9. Check that each date includes month, day, and year
10. If there are conflicting incident dates, I should choose the incident date that is referenced the most time as this is likely the correct date as opposed to noise.
</thinking_process>

<verification_steps>
1. Re-examine the text surrounding each identified incident date
2. For use of force incidents:
   - Confirm the date is connected to phrases like "force was used," "officer applied force," etc.
   - If the incident spans midnight, verify you've identified when force BEGAN
3. For misconduct incidents:
   - Confirm the date is connected to the actual misconduct behavior
4. If there are conflicting dates, evaluate which has:
   - The most frequent mentions
   - The strongest contextual support
   - The most reliable source within the document
5. Verify these are not report completion dates, witness interview dates, or other administrative dates
6. Ensure the dates are not referring to different incidents mentioned in passing
7. Check for conflicting information that might contradict these dates
</verification_steps>

<output_format>
INCIDENT TYPE: [Use of Force / Misconduct / Both / Unclear]

PRIMARY INCIDENT DATE: [MM/DD/YYYY or "No clear incident date found"]

CONFIDENCE LEVEL: [High/Medium/Low]

SUPPORTING EVIDENCE:
[Direct quotes from document that identify this as the incident date]

REASONING:
[Explanation of why this date represents the incident occurrence, including frequency analysis and why it was chosen over any conflicting dates]

ADDITIONAL INCIDENTS (only if clearly separate incidents):
- Incident: [Brief description]
  Date: [MM/DD/YYYY]
  Evidence: [Direct quote supporting this as a separate incident date]

CONFLICTING DATES CONSIDERED:
[List any conflicting dates mentioned for the same incident and explain why they were rejected in favor of the primary date]

OTHER ADMINISTRATIVE DATES:
[List any administrative dates (report filing, etc.) mentioned in the document]
</output_format>
</incident_date_extraction_prompt>
"""

extract_verification_template = """
<incident_date_verification_prompt>
<context>
Your task is to verify the accuracy of the extracted incident date from police documentation, focusing on identifying the SINGLE MOST RELIABLE date. You must distinguish between:
- USE OF FORCE INCIDENT DATE: The specific date when force was first applied by the police officer
- MISCONDUCT INCIDENT DATE: The specific date when the misconduct behavior occurred by the police officer

When conflicting dates appear, your verification must determine which date has the strongest supporting evidence.
</context>

<proposed_incident_date>
{{initial_dates}}
</proposed_incident_date>

<document_for_review>
{{source_text}}
</document_for_review>

<verification_process>
1. Carefully re-read the entire document with focus on all mentioned dates
2. Determine the incident type (use of force vs. misconduct)Your goal is to extract the date on which the shooting occurred, the date on which the use of force occurred or the date on which the misconduct occurred
3. Identify ALL potential incident dates and their frequency of mention
4. For each potential date, evaluate:
   a) Frequency of mention in the document
   b) Strength of contextual evidence linking it to the incident
   c) Consistency with other details in the document
5. For use of force incidents:
   - Verify the date matches when force was FIRST applied
   - For incidents spanning midnight, confirm the START date is captured
6. For misconduct incidents:
   - Verify the date matches when the misconduct behavior occurred
7. If there are conflicting dates, determine which has the most reliable supporting evidence
8. Only identify multiple dates if they clearly refer to separate, distinct incidents
9. Verify each date includes month, day, and year
10. If there are conflicting incident dates, I should choose the incident date that is referenced the most time as this is likely the correct date as opposed to noise.
</verification_process>

<critical_considerations>
- Use of force incident dates are ONLY when force was first physically applied
- Misconduct incident dates are ONLY when the misconduct behavior occurred
- Report filing dates, investigation dates, and administrative dates are NOT incident dates
- When multiple dates are mentioned for the same incident, select the MOST RELIABLE based on:
  a) Frequency of mention
  b) Strength of supporting context
  c) Consistency with other details
- Only return multiple dates when they clearly refer to separate, distinct incidents
- For incidents spanning multiple days, the START date is the incident date
- Look for clear language connecting the date to the specific incident occurrence
</critical_considerations>

<output_format>
VERIFICATION RESULT: [CONFIRMED / CORRECTED / REJECTED]

INCIDENT TYPE: [Use of Force / Misconduct / Both / Unclear]

PRIMARY INCIDENT DATE: [MM/DD/YYYY or "No clear incident date found"]

CONFIDENCE LEVEL: [High/Medium/Low]

JUSTIFICATION:
[Detailed explanation of your verification, including frequency analysis and why this date was selected over any conflicting dates]

KEY EVIDENCE:
[Direct quotes from the document that support your determination]

CONFLICTING DATES ANALYSIS (if applicable):
[Analysis of any conflicting dates, their frequency of mention, and why they were rejected]

ADDITIONAL INCIDENTS (only if clearly separate incidents):
- Incident: [Brief description]
  Date: [MM/DD/YYYY]
  Evidence: [Direct quote supporting this as a separate incident date]
</output_format>
</incident_date_verification_prompt>
"""

format_conversion_template = """If an incident date is stated, return the date in ISO-8601 format.
If there are multiple incident dates, list them all separated by commas.
If the incident date is not stated, return None. Include nothing else in your response.

Valid responses include:
2024-07-18
2022-05-23
None
2021-01-02
None
None
1999-04-11, 2001-05-23
None
1999-04-11, 1999-04-12, 1999-04-13
1999-04-11, 1999-04-12, 1999-04-13, 2022-05-23, 2022-05-23

Below is the document that you are tasked with reviewing:

--------------------

{{ source_text }}

--------------------
"""

# Citation prompts from incident_date_extraction/extract/src/citations.py
primary_citation_template = """
<objective>
Analyze this page of text to determine if it contains citations that support the SPECIFIC incident date identified in the date extraction analysis.
</objective>

<date_extraction_context>
The incident date extraction analysis identified the following:

EXTRACTED INCIDENT DATE: {{incident_date}}

{{date_extraction_context}}

You should look for citations that mention or support this SPECIFIC date as the incident date.
</date_extraction_context>

<incident_date_citation_definition>
INCIDENT DATE CITATIONS must clearly reference when the incident occurred:
- Direct statements like "the incident occurred on [date]"
- "On [date], the shooting took place"
- "The use of force incident on [date]"
- Dates in incident report headers or summaries
- Dates connected to phrases like "date of incident", "incident date", "occurred on"

NOT incident date citations:
- Report filing dates ("report filed on")
- Investigation dates ("investigation began on")
- Interview dates ("witness interviewed on")
- Court dates or hearing dates
- Administrative dates (case opened, closed, reviewed)
- Dates without clear connection to the incident occurrence
</incident_date_citation_definition>

<targeted_search_guidance>
Based on the date extraction context above, prioritize finding citations that:
1. Mention the exact date {{incident_date}} or dates very close to it
2. Connect this date to the actual incident occurrence (not administrative actions)
3. Provide context about what happened on this date
4. Use language that clearly indicates this was when the incident took place
</targeted_search_guidance>

<page_text>
{{page_text}}
</page_text>

<output_instructions>
Focus on finding citations that specifically support the INCIDENT DATE identified in the extraction analysis.
Respond with 'YES' or 'NO' followed by your reasoning and an exact quote.

Format:
[YES/NO]: [Your reasoning explaining whether this page contains citations supporting the SPECIFIC incident date]
Quote: "[Exact text from the page that supports your decision]"
</output_instructions>
"""

validator_citation_template = """
<objective>
Verify whether this page contains a HIGH-QUALITY citation that proves when an incident occurred.
You must distinguish between actual incident dates and administrative dates.
</objective>

<extracted_incident_date>
{{incident_date}}
</extracted_incident_date>

<incident_date_citation_definition>
INCIDENT DATE CITATIONS must clearly state WHEN THE INCIDENT OCCURRED:
- Direct statements: "the incident occurred on [date]"
- Temporal context: "On [date], officers used force"
- Report headers: "Date of Incident: [date]"
- Clear incident narratives: "The shooting took place on [date]"

EXPLICITLY NOT incident date citations:
- Report filing/completion dates
- Investigation start/end dates
- Interview dates
- Court or hearing dates
- Any administrative processing dates
- Dates mentioned without incident context
</incident_date_citation_definition>

<primary_analysis>
{{primary_analysis}}
</primary_analysis>

<page_text>
{{page_text}}
</page_text>

<validation_criteria>
For a citation to be valid:
1. It must mention a date that matches or is very close to {{incident_date}}
2. The date must be clearly connected to the INCIDENT OCCURRENCE (not administrative actions)
3. The quote must be an exact excerpt from the page text
4. The context must make it clear this is when the incident happened

Ask yourself:
- Does this date reference when force was used, when the shooting occurred, or when misconduct happened?
- Or does it reference when a report was filed, investigation occurred, or other administrative action?
</validation_criteria>

<output_format>
Return your analysis as a JSON object with the following structure:
{
    "final_decision": "YES" or "NO",
    "validator_reasoning": "Detailed explanation of why this is or is not a valid incident date citation",
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
