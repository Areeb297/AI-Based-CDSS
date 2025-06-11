import os
import json
import re # For regular expression operations
from dotenv import load_dotenv # Import load_dotenv
from pathlib import Path

# --- Potentially User-Configurable Paths ---
pdf_path = "CDSS_Expected_Outputs_Cases_ALL.pdf"
ref_path = "references_list_key_brackets_value_plain.json"
faiss_index_path = "faiss_cdss_pdf"

# --- Environment Variable Setup ---
load_dotenv() # Load environment variables from .env file

# We will now load and use specific keys for each service directly in their initialization,
# so the generic api_key loading block below is removed.

# --- Import Langchain and related libraries ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document # Ensure Document is imported

import os
from dotenv import load_dotenv
import openai
import base64
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import requests
import re

# --- FastAPI and Pydantic for API creation ---
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# === Load references for post-processing ===
try:
    with open(ref_path, 'r') as f:
        references_dict = json.load(f)
except FileNotFoundError:
    print(f"Error: References file not found at {ref_path}")
    references_dict = {} # Default to empty if not found, or raise error

# === PDF Loading and Vector Store Creation (RAG pipeline setup) ===

# 1. Load PDF document
try:
    loader = PyPDFLoader(pdf_path)
    pdf_document_pages = loader.load_and_split() 
except FileNotFoundError:
    raise FileNotFoundError(f"Error: PDF file not found at {pdf_path}. Please check the path.")
except Exception as e:
    raise RuntimeError(f"Error loading PDF: {e}")


# 2. Split document pages into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
doc_chunks = splitter.split_documents(pdf_document_pages)

# 3. Filter document chunks to ensure they have valid content
filtered_doc_chunks_for_faiss = []
print(f"\nProcessing {len(doc_chunks)} document chunks from PDF...")
for i, chunk in enumerate(doc_chunks):
    content = chunk.page_content
    metadata = chunk.metadata

    if isinstance(content, str) and content.strip():
        filtered_doc_chunks_for_faiss.append(chunk) 
    else:
        page_info = metadata.get('page', 'N/A') if metadata else 'N/A'
        source_info = metadata.get('source', 'N/A') if metadata else 'N/A'
        content_preview = str(content)[:100] + "..." if content and len(str(content)) > 100 else str(content)
        print(f"Warning: Skipping document chunk from source '{source_info}', page {page_info} (original chunk index {i}) due to empty or invalid content. Type: {type(content)}, Content preview: '{content_preview}'")

print(f"Number of valid document chunks for FAISS after filtering: {len(filtered_doc_chunks_for_faiss)}")

if not filtered_doc_chunks_for_faiss:
    raise ValueError("No valid document chunks with content found in the PDF after processing. Cannot build FAISS index.")

# (Optional verification block for filtered_doc_chunks_for_faiss can remain if you find it useful)
print("\n--- Verifying filtered_doc_chunks_for_faiss before FAISS.from_documents ---")
# ... (verification code from your script) ...
if not isinstance(filtered_doc_chunks_for_faiss, list):
    print(f"CRITICAL ERROR: filtered_doc_chunks_for_faiss is not a list. Type: {type(filtered_doc_chunks_for_faiss)}")
    raise TypeError("filtered_doc_chunks_for_faiss is not a list.")
else:
    print(f"filtered_doc_chunks_for_faiss is confirmed to be a list with {len(filtered_doc_chunks_for_faiss)} Document objects.")
    all_docs_have_valid_content = True
    for idx, doc_item in enumerate(filtered_doc_chunks_for_faiss):
        if not isinstance(doc_item, Document):
            print(f"CRITICAL ERROR: Element at index {idx} is NOT a Document object. Type: {type(doc_item)}")
            all_docs_have_valid_content = False
            break
        if not isinstance(doc_item.page_content, str) or not doc_item.page_content.strip():
            page_info = doc_item.metadata.get('page', 'N/A') if doc_item.metadata else 'N/A'
            print(f"CRITICAL ERROR: Document object at index {idx} (page {page_info}) has invalid or empty page_content. Type: {type(doc_item.page_content)}")
            all_docs_have_valid_content = False
            break
    if not all_docs_have_valid_content:
        raise TypeError("CRITICAL: Not all Document objects in filtered_doc_chunks_for_faiss have valid, non-empty page_content. Check logs.")
    else:
        print("All Document objects in filtered_doc_chunks_for_faiss have been verified. Proceeding to FAISS.")
print("--- End of verification for filtered_doc_chunks_for_faiss ---\n")


# 3. Initialize embeddings model for ACTUAL OPENAI
actual_openai_api_key = os.getenv("ACTUAL_OPENAI_API_KEY")
if not actual_openai_api_key:
    raise ValueError("ACTUAL_OPENAI_API_KEY not found in .env file. This key is needed for OpenAI embeddings.")

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=actual_openai_api_key
    # By not specifying openai_api_base_url, it will default to the official OpenAI API endpoint.
)
print("OpenAIEmbeddings initialized to use actual OpenAI API.")

# 4. Create FAISS vector store from document chunks
index_file = Path(faiss_index_path) / "index.faiss"
pkl_file = Path(faiss_index_path) / "index.pkl"

if index_file.exists() and pkl_file.exists():
    print(f"Loading existing FAISS index from {faiss_index_path}")
    # When loading, FAISS just needs an embedding function. The key used during its creation doesn't need to be active
    # for loading if the loaded data doesn't require re-embedding with that specific key immediately.
    # However, the `embeddings` object passed here should be compatible (same model, dimensions).
    vectordb = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
else:
    print(f"FAISS index not found. Building new index at {faiss_index_path} ...")
    vectordb = FAISS.from_documents(filtered_doc_chunks_for_faiss, embeddings)
    vectordb.save_local(faiss_index_path)
    print(f"FAISS index built and saved at {faiss_index_path}")

# === Define Expected Output Structure and Prompt Template ===
EXPECTED_KEYS = [
    "Drug-Drug Interactions", "Drug-Allergy", "Drug-Disease Contraindications",
    "Ingredient Duplication", "Pregnancy Warnings", "Lactation Warnings",
    "General Precautions", "Therapeutic Class Conflicts", "Warning Labels",
    "Indications", "Severity", "References"
]

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
IMPORTANT INSTRUCTIONS:
- If the input 'drugs' list contains more than two drugs, the "Drug-Drug Interactions" category MUST list interactions for all unique pairs of drugs.
- Your output MUST include ALL of these categories, in this exact order and with exact names:
  "Drug-Drug Interactions",
  "Drug-Allergy",
  "Drug-Disease Contraindications",
  "Ingredient Duplication",
  "Pregnancy Warnings",
  "Lactation Warnings",
  "General Precautions",
  "Therapeutic Class Conflicts",
  "Warning Labels",
  "Indications",
  "Severity",
  "References"
- DO NOT skip, rename, or reorder categories.
- If a category has no relevant content in the context, return a list with one string: ["None"]
- If the patient is male, always use: "None (not applicable in male patients). [3]" for Pregnancy and Lactation Warnings. (Ensure reference [3] is defined in your reference list if used this way, or adjust the instruction)
- Every fact/warning/statement in every category MUST cite all its reference numbers from the context at the end in square brackets, e.g., [1][2].
- NEVER use any reference number unless it appears in the context provided to you.
- Group related facts/risks and always cite ALL references shown in the context for those facts.
- For "Severity", always provide a single string in a list: ["High"], ["Moderate"], or ["Low"], based on the most serious risk identified from the context.
- "References" must be present and include every unique reference number used in the rest of the output, in ascending order, with their full citation text, as in the example.
- Your output must ALWAYS match the example’s category order, structure, and referencing style—no explanations, no text outside the JSON object itself.
- CRITICAL JSON FORMATTING: All JSON string values MUST be properly escaped. For example, newline characters within a string value MUST be represented as '\\\\n', not as a literal newline. Do not include any unescaped control characters (like raw newlines or tabs) within string values. The entire output must be a single, valid JSON object.

### Example input:
{{
  "drugs": ["lisinopril", "ibuprofen", "aspirin"],
  "age": 34,
  "gender": "Female",
  "allergies": ["sulfa drugs"],
  "diagnosis": "Gestational Hypertension"
}}

### Example output:
{{
  "Drug-Drug Interactions": [
    "Increased risk of kidney injury and hyperkalemia when ACE inhibitors (like lisinopril) are combined with NSAIDs like ibuprofen. [1][2]",
    "Increased risk of bleeding when ACE inhibitors (like lisinopril) are combined with aspirin. [X][Y]",
    "Increased risk of GI bleeding when NSAIDs like ibuprofen are combined with aspirin. [A][B]"
  ],
  "Drug-Allergy": [
    "No known cross-reactivity between lisinopril, ibuprofen or aspirin and sulfa drugs. [3]"
  ],
  "Drug-Disease Contraindications": [
    "NSAIDs like ibuprofen may worsen hypertension and impair kidney function, especially during pregnancy. [4]"
  ],
  "Ingredient Duplication": [
    "No overlapping active ingredients. [5]"
  ],
  "Pregnancy Warnings": [
    "Lisinopril is contraindicated in pregnancy due to risk of fetal toxicity and malformations. [6][7]"
  ],
  "Lactation Warnings": [
    "Ibuprofen is generally considered compatible with breastfeeding, but should be used with caution. [8]"
  ],
  "General Precautions": [
    "Monitor blood pressure, kidney function, and for signs of hyperkalemia when combining lisinopril and ibuprofen. [1][2]"
  ],
  "Therapeutic Class Conflicts": [
    "ACE inhibitors and NSAIDs both can affect renal function; risk is additive. [1][2]"
  ],
  "Warning Labels": [
    "NSAIDs may worsen hypertension and kidney function in pregnancy. [2][4]",
    "Lisinopril can cause fetal toxicity if used during pregnancy. [6][7]"
  ],
  "Indications": [
    "Lisinopril: ACE inhibitor used for hypertension and heart failure. [7]",
    "Ibuprofen: Analgesic and anti-inflammatory used for pain, fever, and inflammation. [9]",
    "Aspirin: NSAID used for pain, fever, inflammation, and antiplatelet effects. [C]"
  ],
  "Severity": ["High"],
  "References": [
    "[1] Whelton A. Nephrotoxicity of nonsteroidal anti-inflammatory drugs: physiologic foundations and clinical implications. Am J Med. 1999;106(5B):13S-24S. [PMID:10390106]",
    "[2] FDA label for ibuprofen: [https://www.accessdata.fda.gov/drugsatfda_docs/label/2016/017463s117lbl.pdf]",
    "[3] Lexicomp Drug Allergy Checker. No cross-reactivity between sulfa drugs and NSAIDs/ACE inhibitors.",
    "[4] American College of Obstetricians and Gynecologists (ACOG). Hypertension in pregnancy. Practice Bulletin No. 203.",
    "[5] Best clinical practice/expert consensus.",
    "[6] FDA label for lisinopril: [https://www.accessdata.fda.gov/drugsatfda_docs/label/2016/019690s043lbl.pdf]",
    "[7] UpToDate: Use of antihypertensive drugs in pregnancy and lactation.",
    "[8] Hale TW. Medications and Mothers' Milk. Ibuprofen considered safe during lactation.",
    "[9] FDA label for ibuprofen: [https://www.accessdata.fda.gov/drugsatfda_docs/label/2016/017463s117lbl.pdf]",
    "[A] Reference A text.",
    "[B] Reference B text.",
    "[C] Reference C text.",
    "[X] Reference X text.",
    "[Y] Reference Y text."
  ]
}}
Copy this output structure, style, and formatting exactly. All lists, commas, and brackets must match this format.

Clinical Query:
{question}

CONTEXT (Relevant Knowledge & Example Outputs from the PDF Document):
{context}

FILL EVERY CATEGORY that has any information in the context. No explanations. Only a valid JSON object.
"""
)




# ------- PATHWAY PROMPT -------
PATHWAY_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a clinical decision support system.
For every clinical scenario, ALWAYS output 3 clinical management pathways (as JSON), in this order:
- "Option 1" (lowest risk, guideline-adherent)
- "Option 2" (medium risk, mitigated with monitoring)
- "Option 3" (high risk, unsafe, not recommended)

For each pathway, include:
- "recommendation": Specify exact drug name, dose, route, frequency, maximum duration, any adjuncts (e.g., add PPI for GI protection), explicit alternatives, what to avoid, and for how long. For the "high" risk pathway, ALWAYS include: "Do not use unless life-threatening situation and no alternatives."
- "details": concise clinical rationale for this pathway.
- "tradeoff": describe pros and cons (e.g., "slower pain control, lowest risk" vs. "fastest pain control, high risk of bleeding").
- "monitoring": frequency and specifics (e.g., "check INR every 3 days, watch for GI bleeding signs daily").
- In every recommendation, details, tradeoff, or monitoring field, cite the relevant reference numbers inline at the end (e.g., [1][2][3]) if supported by the context. If there is no reference, cite [no reference].

**For Saudi Arabia:**
- Do not mention alcohol or its avoidance (alcohol is prohibited and not available).
- Use only locally available medications and standard practice in Saudi Arabia.

**References requirements:**
- Your output MUST be a single JSON object with "clinical_pathways" and "References" as top-level keys.
- ALWAYS include the "References" object at the end, mapping every cited number in the pathways to the full citation (with link) from the context. If there is no reference, do not include it in the References object.

**Style and diversity requirements:**
- Vary your wording, structure, and phrasing for every test case.
- Do not copy the example output or use the same template; use synonyms, reword sentences, and make the answer feel fresh for each scenario.
- Make every pathway reflect the patient's age, gender, diagnosis, and specific drug details.
- If applicable, highlight any unique Saudi practice or local guideline.
- Avoid "boilerplate" language.

ALWAYS output valid JSON with this structure and nothing else.

EXAMPLE INPUT 1:
{{{{
    "input": {{{{
      "drugs": ["warfarin", "ibuprofen"],
      "age": 70,
      "gender": "Male",
      "allergies": [],
      "diagnosis": "Atrial Fibrillation and Osteoarthritis"
    }}}},

    "expected_output": {{{{
      "Drug-Drug Interactions": [
        "Risk of serious bleeding, especially GI, when warfarin is combined with NSAIDs such as ibuprofen."
      ],
      "Drug-Allergy": [
        "No allergy conflicts."
      ],
      "Drug-Disease Contraindications": [
        "NSAIDs like ibuprofen may worsen control of hypertension and can increase risk of cardiovascular events."
      ],
      "Ingredient Duplication": [
        "No overlapping ingredients; both increase bleeding risk through different mechanisms."
      ],
      "Pregnancy Warnings": [
        "None (not applicable in male patients)."
      ],
      "Lactation Warnings": [
        "None (not applicable in male patients)."
      ],
      "General Precautions": [
        "Monitor for bleeding signs like bruising or dark stools.",
        "Monitor for signs of kidney damage due to ibuprofen, especially in elderly."
      ],
      "Therapeutic Class Conflicts": [
        "Both affect clotting and may increase bleeding risk when used together."
      ],
      "Warning Labels": [
        "Risk of serious bleeding, especially GI, when warfarin is combined with NSAIDs.",
        "Risk of gastrointestinal bleeding is higher in older adults taking NSAIDs."
      ],
      "Indications": [
        "Warfarin: Prevention and treatment of thromboembolic disorders such as atrial fibrillation and deep vein thrombosis.",
        "Ibuprofen: Analgesic and anti-inflammatory used for pain, fever, and inflammation."
      ]
    }}}},
    "severity": "high"
}}}}

EXAMPLE OUTPUT 1:
{{{{
  "clinical_pathways": [
    {{{{
      "recommendation": "Acetaminophen 500mg orally every 6 hours as needed for pain, not exceeding 3g per day, for up to 10 days. Avoid NSAIDs entirely. Review pain control and function after 7 days. [1][2][3]",
      "details": "Avoids NSAID-warfarin interaction, minimizing bleeding risk. [1][2]",
      "tradeoff": "Pain relief may be slower or less potent, but risk of major bleeding is minimized. [2][3]",
      "monitoring": "Check INR as per anticoagulation protocol. Review pain and function at 7 days. [2][4]"
    }}}},
    {{{{
      "recommendation": "If pain is significant, ibuprofen 200mg orally twice daily after food for up to 3 days, with omeprazole 20mg daily for GI protection, may be considered under close INR monitoring. Stop NSAID immediately if any bleeding or easy bruising occurs. [2][3][4]",
      "details": "Short-term NSAID use with GI protection and INR monitoring offers improved pain relief with a controlled risk of bleeding. [2][4]",
      "tradeoff": "Better symptom control but requires vigilant monitoring for complications. [3][4]",
      "monitoring": "Check INR before, during, and after NSAID course; monitor for GI symptoms and bruising daily. [2][4]"
    }}}},
    {{{{
      "recommendation": "Ibuprofen 400mg orally three times daily for 7 days without any GI protection or INR monitoring. Do not use unless life-threatening situation and no alternatives. [no reference]",
      "details": "Extended, high-dose NSAID therapy in anticoagulated elderly carries a very high risk of GI and intracranial hemorrhage. [no reference]",
      "tradeoff": "Rapid and powerful pain relief but at the cost of life-threatening complications. [no reference]",
      "monitoring": "None. [no reference]"
    }}}}
  ],
  "References": {{{{
    "[1]": "FDA label for warfarin: https://www.accessdata.fda.gov/drugsatfda_docs/label/...",
    "[2]": "Saudi MOH Anticoagulation Guidelines: https://moh.gov.sa/anticoag-guidelines.pdf",
    "[3]": "Saudi MOH Pain Management Guidelines: https://moh.gov.sa/pain-guidelines.pdf",
    "[4]": "Best Practices in INR Monitoring, Saudi MOH 2021: https://moh.gov.sa/inr-monitoring.pdf"
  }}}}
}}}}

EXAMPLE INPUT 2:
{{{{
"input": {{{{
        "drugs": ["lisinopril", "ibuprofen"],
        "age": 34,
        "gender": "Female",
        "allergies": ["sulfa drugs"],
        "diagnosis": "Gestational Hypertension"
}}}},

"expected_output": {{{{
            "Drug-Drug Interactions": [
                "Increased risk of kidney injury and hyperkalemia when ACE inhibitors like lisinopril are combined with NSAIDs."
            ],
            "Drug-Allergy": [
                "No known cross-reactivity between lisinopril or ibuprofen and sulfa drugs."
            ],
            "Drug-Disease Contraindications": [
                "NSAIDs like ibuprofen may worsen hypertension and impair kidney function, especially during pregnancy."
            ],
            "Ingredient Duplication": [
                "No overlapping active ingredients."
            ],
            "Pregnancy Warnings": [
                "Lisinopril is contraindicated in pregnancy due to risk of fetal toxicity and malformations."
            ],
            "Lactation Warnings": [
                "Ibuprofen is generally considered compatible with breastfeeding, but should be used with caution."
            ],
            "General Precautions": [
                "Monitor blood pressure, kidney function, and for signs of hyperkalemia when combining lisinopril and NSAIDs."
            ],
            "Therapeutic Class Conflicts": [
                "ACE inhibitors and NSAIDs both can affect renal function; risk is additive."
            ],
            "Warning Labels": [
                "NSAIDs may worsen hypertension and kidney function in pregnancy.",
                "Lisinopril can cause fetal toxicity if used during pregnancy."
            ],
            "Indications": [
                "Lisinopril: ACE inhibitor used for hypertension and heart failure.",
                "Ibuprofen: Analgesic and anti-inflammatory used for pain, fever, and inflammation."
            ]
}}}},
"severity": "high"
}}}}

EXAMPLE OUTPUT 2:
{{{{
  "clinical_pathways": [
    {{{{
      "recommendation": "For gestational hypertension, discontinue lisinopril immediately and use paracetamol 500mg orally every 6-8 hours as needed for pain, not exceeding 4g daily. NSAIDs are contraindicated in pregnancy. Consult an obstetrician for alternative antihypertensive therapy. [5][6][7]",
      "details": "Paracetamol is the preferred analgesic in pregnancy; both lisinopril and NSAIDs are contraindicated due to fetal risk. [6][7]",
      "tradeoff": "Prioritizes maternal and fetal safety, but may offer only moderate pain relief. [5][7]",
      "monitoring": "Monitor maternal blood pressure and renal function every clinic visit. [7][8]"
    }}}},
    {{{{
      "recommendation": "If pain is uncontrolled, ibuprofen 200mg orally twice daily with meals for up to 48 hours may be considered under strict specialist supervision, only in the second trimester, and never with ACE inhibitors. [6][8]",
      "details": "Short-term, low-dose NSAID use in the second trimester is possible under supervision, but with risk. [6][8]",
      "tradeoff": "Improved pain control, but moderate risk to fetal renal development and pregnancy complications. [8]",
      "monitoring": "Obstetrician should monitor maternal kidney function and fetal wellbeing closely during and after use. [7][8]"
    }}}},
    {{{{
      "recommendation": "Continue lisinopril and initiate ibuprofen 400mg orally three times daily for 7 days in pregnancy with no monitoring. Do not use unless life-threatening situation and no alternatives. [no reference]",
      "details": "Combining ACE inhibitors and NSAIDs in pregnancy leads to major risk of fetal toxicity, malformation, and maternal kidney injury. [no reference]",
      "tradeoff": "Strongest pain and BP control, but highest risk for catastrophic fetal and maternal outcomes. [no reference]",
      "monitoring": "None. [no reference]"
    }}}}
  ],
  "References": {{{{
    "[5]": "WHO Model List of Essential Medicines, 2023: https://www.who.int/publications/i/item/WHO-MED-2023.04",
    "[6]": "Saudi MOH Guidelines for Hypertension in Pregnancy: https://www.moh.gov.sa/Hypertension-Pregnancy-Guidelines.pdf",
    "[7]": "FDA label for lisinopril: https://www.accessdata.fda.gov/drugsatfda_docs/label/...",
    "[8]": "Management of non-obstetric pain in pregnancy, UpToDate, 2022."
  }}}}
}}}}

SCENARIO:
{question}

CONTEXT:
{context}
"""
)


# === Initialize LLM for GROQ API and RAG Chain ===
# === Initialize LLM for GROQ API and RAG Chain ===
groq_api_key_for_llm = os.getenv("GROQ_API_KEY")
if not groq_api_key_for_llm:
    raise ValueError("GROQ_API_KEY not found in .env file. This key is needed for the Groq LLM.")

llm = ChatOpenAI(
    model="llama3-70b-8192", # Or your preferred Groq model
    temperature=0.0,
    openai_api_key=groq_api_key_for_llm,
    openai_api_base="https://api.groq.com/openai/v1" # <<< CORRECTED PARAMETER NAME
)
print("ChatOpenAI (LLM) initialized to use Groq API.")

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 12}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

llm_pathways = ChatOpenAI(
    model="llama3-70b-8192", # Or your preferred Groq model
    temperature=0.1,
    openai_api_key=groq_api_key_for_llm,
    openai_api_base="https://api.groq.com/openai/v1" # <<< CORRECTED PARAMETER NAME
)

# Initialize the LLM chain for pathways
rag_chain_pathways = RetrievalQA.from_chain_type(
    llm=llm_pathways,  # (or llm if using the same LLM)
    retriever=vectordb.as_retriever(search_kwargs={"k": 17}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PATHWAY_PROMPT},
)


# === FastAPI Application Setup ===
app = FastAPI(
    title="Clinical Decision Support API",
    description="API for assessing clinical cases using RAG.",
    version="1.0.1"
)

class ClinicalRequest(BaseModel):
    drugs: list
    age: int
    gender: str
    allergies: list = []
    diagnosis: str = ''
    
class PathwayTestCase(BaseModel):
    input: dict
    expected_output: dict = {}
    severity: str = ""


def enforce_contract(output_json, references_dict_lookup):
    processed_json = {}
    for key in EXPECTED_KEYS:
        processed_json[key] = output_json.get(key, ["None"] if key != "References" else [])

    ref_nums_extracted = set()
    for key, values in processed_json.items():
        if key == "References": continue
        if isinstance(values, list):
            for value_item in values:
                if isinstance(value_item, str): 
                    found_numbers = re.findall(r"[(\d+)]", value_item) # Corrected regex
                    for num_str in found_numbers:
                        ref_nums_extracted.add(num_str)
    
    refs_full_text = []
    valid_ref_nums_int = []
    for num_str in ref_nums_extracted:
        try:
            valid_ref_nums_int.append(int(num_str))
        except ValueError:
            print(f"Warning: Could not convert extracted reference '{num_str}' to an integer. Skipping.")
            
    for num_int in sorted(list(set(valid_ref_nums_int))): 
        ref_key_in_dict = f"[{num_int}]"
        if ref_key_in_dict in references_dict_lookup:
            refs_full_text.append(f"{ref_key_in_dict} {references_dict_lookup[ref_key_in_dict]}")
        else:
            refs_full_text.append(f"{ref_key_in_dict} Reference text not found.")
    
    processed_json["References"] = refs_full_text
    return processed_json

def extract_json_from_response(text: str) -> dict:
    text = text.strip()
    match_json_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if match_json_block:
        json_str = match_json_block.group(1)
    else:
        match_curly_braces = re.search(r"(\{[\s\S]*\})", text)
        if not match_curly_braces:
            raise ValueError("No JSON object found in LLM output. Output was: " + text)
        json_str = match_curly_braces.group(1)
    
    json_str = json_str.strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        error_message = (
            f"Failed to decode JSON from LLM output. Error: {e}. "
            f"Problematic JSON string (after extraction attempts) was:\n>>>\n{json_str}\n<<<"
        )
        raise ValueError(error_message)

@app.post("/clinical-assess")
def assess_case(request: ClinicalRequest):
    user_query = (
        f"Input:\n{request.dict()}\n"
        "Generate structured output as shown in the clinical decision support format provided in the prompt."
    )
    try:
        result = rag_chain.invoke({"query": user_query})
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        return {"error": f"Error during RAG chain invocation: {str(e)}", "raw_output": None}

    llm_output_text = result.get("result", "")
    try:
        output_json = extract_json_from_response(llm_output_text)
    except ValueError as e:
        print(f"Error extracting JSON: {e}")
        return {"error": str(e), "raw_output": llm_output_text, "source_documents": result.get("source_documents")}

    try:
        final_output_json = enforce_contract(output_json, references_dict)
    except Exception as e:
        print(f"Error enforcing contract: {e}")
        return {"error": f"Error enforcing contract: {str(e)}", "raw_llm_json": output_json, "raw_output": llm_output_text}
    
    return final_output_json

# --- Endpoint 2: /clinical-pathways (structured like /clinical-assess) ---
@app.post("/clinical-pathways")
def clinical_pathways(case: PathwayTestCase):
    user_query = (
        f"Patient scenario:\n{json.dumps(case.input, ensure_ascii=False)}\n"
        "Generate three clinical pathways with references as shown in the pathways output format."
    )
    try:
        result = rag_chain_pathways.invoke({"query": user_query})
    except Exception as e:
        print(f"Error during RAG chain invocation for /clinical-pathways: {e}")
        return {"error": f"Error during RAG chain invocation: {str(e)}", "raw_output": None}

    llm_output_text = result.get("result", "")
    try:
        output_json = extract_json_from_response(llm_output_text)
    except ValueError as e:
        print(f"Error extracting JSON for /clinical-pathways: {e}")
        return {"error": str(e), "raw_output": llm_output_text, "source_documents": result.get("source_documents")}

    return output_json


"""# ==================== OPENROUTER SECTION ===================="""

# Install required packages for HTTP requests and API server
# !pip install requests fastapi uvicorn requests python-multipart python-dotenv

# ---- DETAILED PROMPT CONFIGURATION ----
# System prompt defines the AI assistant's expertise and behavior
system_prompt = (
    "You are a highly trained, detail-oriented clinical radiology assistant. Your answers must always include clear laterality, precise anatomic locations, and mention of classic radiological signs if present. If the findings are ambiguous, state so honestly and recommend further steps."
)

# User prompt provides specific instructions for chest X-ray interpretation
# This prompt ensures thorough analysis and prevents missing subtle findings
user_prompt = (
    "You are a radiology assistant reviewing a chest X-ray for a physician. "
    "You must actively search for and describe any possible abnormal findings, even if subtle or borderline. "
    "Never declare the X-ray completely normal unless you are absolutely certain. "
    "If there is any uncertainty, mention and describe all possible abnormalities, opacities, densities, lesions, or subtle changes. "
    "Include a brief differential diagnosis for any finding, and always comment on the most clinically significant abnormalities first.\n\n"
    "Write your report using this exact format:\n\n"
    "1. AP (Frontal) View: [A detailed, well-structured paragraph describing all findings, both positive and negative, including any subtle abnormalities, their location and laterality, and classic radiological signs if present.]\n\n"
    "2. Lateral View: [A paragraph describing findings for the lateral view, including confirmation or refutation of AP findings, and any new abnormalities seen only on the lateral view.]\n\n"
    "3. Impression: [A concise summary diagnosis, mentioning any abnormal findings, differentials, and clinical recommendations.]\n\n"
    "4. Recommendations: [A short paragraph with specific next steps, such as further imaging, clinical follow-up, or limitations of the study.]\n\n"
    "Do NOT use bullet points, asterisks, or special characters. Each section must be a numbered header followed by a paragraph, separated by a blank line. Focus on clinical safety and do not miss subtle findings."
)

# Get OpenRouter API key from environment variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Check if API key is loaded
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

# ---- REPORT PARSING FUNCTION ----
def parse_report(report_text):
    """
    Parse the AI-generated report into structured sections
    
    Args:
        report_text (str): Raw text report from the AI model
    
    Returns:
        dict: Dictionary with keys for each report section
    """
    # Define expected section keys for the structured report
    keys = ["AP_View", "Lateral_View", "Impression", "Recommendations"]
    
    # Use regex to split the report by numbered section headers
    # This pattern matches variations of section names
    sections = re.split(r'\n*\d\.\s*(?:AP \(Frontal\) View|Lateral View|Impression|Recommendations):', report_text)
    
    # Clean up sections - remove empty strings and whitespace
    sections = [s.strip() for s in sections if s.strip()]
    
    # Map sections to dictionary keys, handling missing sections gracefully
    report_dict = {k: (sections[i] if i < len(sections) else "") for i, k in enumerate(keys)}
    return report_dict

# ---- FASTAPI APPLICATION SETUP ----
# Create FastAPI instance for the web service
app = FastAPI()

# Define the main endpoint for report generation
@app.post("/generate_image_report/")
async def generate_report(
    image: UploadFile = File(...),  # Required image file upload
    query: str = Form("Patient presents with cough and fever. Please interpret the attached chest X-ray."),  # Optional clinical context
):
    """
    Generate a structured radiology report from an uploaded medical image
    
    Args:
        image: Uploaded image file (chest X-ray)
        query: Clinical context or specific questions about the image
    
    Returns:
        JSONResponse: Structured report with separate sections
    """
    # Read the uploaded image into memory
    img_bytes = await image.read()
    
    # Convert image to base64 for API transmission
    b64_img = base64.b64encode(img_bytes).decode('utf-8')
    
    # Combine the standard prompt with user's specific query
    full_user_message = user_prompt + "\n\n" + query
    
    # Prepare the API request payload
    data = {
        "model": "openai/gpt-4.1",  # Using Llama 3.2 90B Vision model
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": full_user_message},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64_img}}
            ]}
        ],
        "max_tokens": 1000,      # Allow longer responses for detailed reports
        "temperature": 0.2,     # Lower temperature for more consistent medical terminology
        "top_p": 0.7           # Nucleus sampling for balanced creativity vs accuracy
    }
    
    # Set request headers with API key
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Make the API request to OpenRouter
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    
    # Parse the API response
    result = response.json()
    
    # Check if the response contains valid choices
    if "choices" in result:
        # Extract the generated report text
        raw_report = result["choices"][0]["message"]["content"]
        
        # Parse the report into structured sections
        structured_report = parse_report(raw_report)
        
        # Return the structured report as JSON
        return JSONResponse(structured_report)
    else:
        # Return error details if the API call failed
        return JSONResponse({"error": result}, status_code=400)

# ---- RUN THE FASTAPI SERVER ----
# Start the server on localhost port 8000
# Access the API at http://localhost:8000/generate_report/

# --- Main execution block to run the FastAPI app with Uvicorn ---
if __name__ == "__main__":
    print("Starting FastAPI server with Uvicorn...")
    port = int(os.environ.get("PORT", 10000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)