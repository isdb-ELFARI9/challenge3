from utils.llm import call_gemini_llm
import json

def build_document_composer_prompt(user_context, spia_out, arda_out, stsa_out, reasoning_trace) -> str:
    return f"""
You are the Document Composer Agent for an Islamic finance standards review system.

Your job is to assemble a complete, standards-compliant document for the updated FAS, using the following information: user context, Shariah solution, updated accounting rules, updated FAS sections, and the full reasoning trace.

Follow this exact JSON structure for the document (fill in each section as completely as possible, using the information provided):
{{
  "pages": [
    {{
      "title": "Title & Effective Date",
      "content": {{
        "standard_number": "[Generated FAS Number]",
        "title": "[Generated Title of the Standard]",
        "example_title": "Accounting for Innovative Digital Assets",
        "effective_date": "[DD Month YYYY (Gregorian)] / [DD Month YYYY (Hijri)]",
        "issued_by": "AI-Driven Standard Enhancement System (Conceptual)"
      }}
    }},
    {{
      "title": "Objective, Scope & Key Principles",
      "content": {{
        "objective": [
          "Briefly state the primary goal(s) of this standard.",
          "Highlight any significant changes or clarifications."
        ],
        "scope": {{
          "applicable_transactions_entities": "Clearly define the transactions, events, instruments, or types of Islamic financial institutions to which this standard applies.",
          "exclusions": "Explicitly state any related transactions, instruments, or scenarios that are not covered by this standard."
        }},
        "key_shariah_principles": [
          "Concisely outline the core Shari'ah principles that form the basis for the accounting treatments prescribed in this standard."
        ]
      }}
    }},
    {{
      "title": "Definitions",
      "content": {{
        "definitions": [
          {{ "term": "[Term 1]", "definition": "Clear and concise definition" }},
          {{ "term": "[Term 2]", "definition": "Clear and concise definition" }},
          {{ "term": "[Term 3]", "definition": "Clear and concise definition" }}
        ]
      }}
    }},
    {{
      "title": "Recognition and Initial Measurement",
      "content": {{
        "recognition": {{
          "general_criteria": "State the conditions under which an item should be recognized in the financial statements.",
          "specific_points": "Detail specific recognition timing or events for the main elements covered by the standard."
        }},
        "initial_measurement": {{
          "general_principle": "State the general principle for measuring the item upon its initial recognition.",
          "specific_guidance": "Provide specific guidance on how to determine the initial measurement amount for different components or scenarios."
        }}
      }}
    }},
    {{
      "title": "Subsequent Measurement & De-recognition",
      "content": {{
        "subsequent_measurement": {{
          "measurement_basis": "Specify how the item(s) should be measured in reporting periods after initial recognition.",
          "changes_in_value": "Explain how gains or losses arising from re-measurement are to be recognized.",
          "profit_loss_recognition": "Detail how profits or losses are to be recognized and allocated."
        }},
        "de_recognition": {{
          "criteria": "State the conditions under which an item should be removed from the statement of financial position.",
          "accounting": "Explain how to account for any gain or loss arising on de-recognition."
        }}
      }}
    }},
    {{
      "title": "Disclosure Requirements",
      "content": {{
        "qualitative_disclosures": [
          "Description of the nature of activities or instruments.",
          "Key accounting policies adopted.",
          "Information about risk management practices.",
          "Shari'ah compliance aspects."
        ],
        "quantitative_disclosures": [
          "Breakdown of carrying amounts.",
          "Reconciliation of movements in carrying amounts.",
          "Income/expense recognized.",
          "Maturity analysis.",
          "Information on impaired assets and provisions."
        ]
      }}
    }},
    {{
      "title": "Basis for Conclusions (Summary)",
      "content": {{
        "rationale": "Provide a concise summary of the main reasons why the specific accounting treatments were adopted.",
        "shariah_compliance": "Briefly reiterate how the standard ensures alignment with Shari'ah principles.",
        "addressing_gaps": "Explain how this standard addresses the specific user context or gap."
      }}
    }}
  ]
}}

For each section, use the most relevant information from the agent outputs and reasoning trace. If a section is not directly covered, synthesize a concise, standards-appropriate entry based on the context.

Example: For 'Definitions', extract or synthesize key terms from the Shariah and accounting updates. For 'Basis for Conclusions', summarize the rationale and compliance points from the reasoning trace.

User context: {user_context}

SPIA output: {spia_out}

ARDA output: {arda_out}

STSA output: {stsa_out}

Reasoning trace: {reasoning_trace}

Respond ONLY with the JSON object for the document, following the required structure.
"""

def document_composer_agent(user_context, spia_out, arda_out, stsa_out, reasoning_trace) -> dict:
    prompt = build_document_composer_prompt(user_context, spia_out, arda_out, stsa_out, reasoning_trace)
    response = call_gemini_llm(prompt)
    response = response.strip()
    if response.startswith("```"):
        response = response.split('\n', 1)[1]
        if response.endswith("```"):
            response = response.rsplit('\n', 1)[0]
    return json.loads(response) 