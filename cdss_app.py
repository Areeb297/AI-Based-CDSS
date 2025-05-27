import os
import json
import re # For regular expression operations
from dotenv import load_dotenv # Import load_dotenv

# --- Potentially User-Configurable Paths ---
# Update these paths if your files are located elsewhere or named differently.
pdf_path = "CDSS_Expected_Outputs_Cases_ALL.pdf"  # Path to the PDF document
ref_path = "references_list_key_brackets_value_plain.json" # Path to the JSON file containing references

# --- Environment Variable Setup ---
load_dotenv() # Load environment variables from .env file

# Set API key and base URL for the OpenAI-compatible LLM service (e.g., Groq)
# The API key is now loaded from the .env file.
# Make sure you have a .env file in the same directory as your script with:
#
# You can still set a default or directly assign if the .env variable isn't found,
# or rely on it being set in the environment. For this example, we'll prioritize
# the .env file and then allow for an explicit environment setting.

# It's good practice to check if the key was loaded
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY not found in .env file or environment variables.")
    # You could fall back to a hardcoded key here for development, but it's not recommended for production:
    # api_key = "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" # Fallback, not recommended
else:
    os.environ["GROQ_API_KEY"] = api_key # Set it for Langchain if it expects it in os.environ

# The API base can still be set directly or also moved to .env if preferred
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1" # Replace with your actual base URL if needed
# Or, to load from .env:
# OPENAI_API_BASE="https://api.groq.com/openai/v1"
# And in Python:
# api_base = os.getenv("OPENAI_API_BASE", "https://api.groq.com/openai/v1") # With a default
# os.environ["OPENAI_API_BASE"] = api_base


# --- Import Langchain and related libraries ---
from langchain_community.document_loaders import PyPDFLoader

# Example of how you might access them (optional, for verification)
# print(f"Using API Key: {os.getenv('OPENAI_API_KEY')}")
# print(f"Using API Base: {os.getenv('OPENAI_API_BASE')}")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- FastAPI and Pydantic for API creation ---
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn # ASGI server to run FastAPI

# === Load references for post-processing ===
# This dictionary will be used to look up the full text of references
# based on the numbers cited by the LLM.
with open(ref_path, 'r') as f: # Ensure 'r' mode for reading
    references_dict = json.load(f)

# === PDF Loading and Vector Store Creation (RAG pipeline setup) ===

# 1. Load PDF document
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split() # Loads PDF and splits it into pages (each page is a Document)

# 2. Split documents into smaller chunks
# This is important for effective retrieval, as LLMs have context window limits.
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = splitter.split_documents(pages) # Further splits pages into smaller text chunks
texts = [doc.page_content for doc in docs] # Extract text content
metadatas = [doc.metadata for doc in docs] # Extract metadata (e.g., page number)

# 3. Initialize embeddings model
# This model converts text chunks into numerical vectors.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create FAISS vector store from document chunks
# FAISS allows for efficient similarity search on the text vectors.
# This is the "Retrieval" part of RAG.
# Check if a local FAISS index already exists to save time.
faiss_index_path = "faiss_cdss_pdf"
if os.path.exists(faiss_index_path):
    print(f"Loading existing FAISS index from {faiss_index_path}")
    vectordb = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True) # Added allow_dangerous_deserialization for newer Langchain versions
else:
    print(f"Creating new FAISS index and saving to {faiss_index_path}")
    vectordb = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectordb.save_local(faiss_index_path)


# === Define Expected Output Structure and Prompt Template ===

# List of keys that MUST be present in the final JSON output.
EXPECTED_KEYS = [
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
]

# Define the prompt template for the LLM.
# This template guides the LLM to generate output in the desired format and style.
# It includes instructions on structure, content, and reference citation.
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
    "[2] FDA label for ibuprofen: [https://www.accessdata.fda.gov/drugsatfda_docs/label/2016/017463s117lbl.pdf](https://www.accessdata.fda.gov/drugsatfda_docs/label/2016/017463s117lbl.pdf)",
    "[3] Lexicomp Drug Allergy Checker. No cross-reactivity between sulfa drugs and NSAIDs/ACE inhibitors.",
    "[4] American College of Obstetricians and Gynecologists (ACOG). Hypertension in pregnancy. Practice Bulletin No. 203.",
    "[5] Best clinical practice/expert consensus.",
    "[6] FDA label for lisinopril: [https://www.accessdata.fda.gov/drugsatfda_docs/label/2016/019690s043lbl.pdf](https://www.accessdata.fda.gov/drugsatfda_docs/label/2016/019690s043lbl.pdf)",
    "[7] UpToDate: Use of antihypertensive drugs in pregnancy and lactation.",
    "[8] Hale TW. Medications and Mothers' Milk. Ibuprofen considered safe during lactation.",
    "[9] FDA label for ibuprofen: [https://www.accessdata.fda.gov/drugsatfda_docs/label/2016/017463s117lbl.pdf](https://www.accessdata.fda.gov/drugsatfda_docs/label/2016/017463s117lbl.pdf)",
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
# === Initialize LLM and RAG Chain ===

# Initialize the LLM (Language Model). Here, using a Llama model via Groq.
# Temperature=0.0 aims for more deterministic and less creative outputs.
llm = ChatOpenAI(model="llama3-70b-8192", temperature=0.0)

# Create the RetrievalQA chain.
# This chain combines the retriever (FAISS vector store) with the LLM.
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 12}), # Retrieve top 12 relevant chunks
    return_source_documents=True, # Useful for debugging, to see what context was retrieved
    chain_type_kwargs={"prompt": prompt}, # Pass the custom prompt to the LLM part of the chain
)

# === FastAPI Application Setup ===

app = FastAPI(
    title="Clinical Decision Support API",
    description="API for assessing clinical cases using RAG.",
    version="1.0.1"
)

# Pydantic model for input validation. This defines the structure of the request body.
class ClinicalRequest(BaseModel):
    drugs: list
    age: int
    gender: str
    allergies: list = []
    diagnosis: str = ''

# Function to enforce the output contract (JSON structure and reference formatting)
def enforce_contract(output_json, references_dict_lookup):
    """
    Ensures the output JSON conforms to EXPECTED_KEYS and formats references.

    Args:
        output_json (dict): The JSON output from the LLM.
        references_dict_lookup (dict): The dictionary of reference texts.

    Returns:
        dict: The processed and validated JSON output.
    """
    processed_json = {}

    # Ensure all expected keys are present and in order
    for key in EXPECTED_KEYS:
        if key in output_json:
            processed_json[key] = output_json[key]
        else:
            # Provide default "None" value if key is missing, except for "References"
            processed_json[key] = ["None"] if key != "References" else []

    # Collect all unique reference numbers cited in the output
    ref_nums = set()
    for key, values in processed_json.items():
        if key == "References": # Skip the References key itself for now
            continue
        if isinstance(values, list):
            for value_item in values:
                # Find all occurrences of [number]
                found_numbers = re.findall(r"\[(\d+)\]", str(value_item))
                for num in found_numbers:
                    ref_nums.add(num)

    # Build the "References" list with full citations
    refs_full_text = []
    # Sort reference numbers numerically before looking them up
    for num_str in sorted(list(ref_nums), key=int):
        ref_key_in_dict = f"[{num_str}]" # The format references are stored in references_dict
        if ref_key_in_dict in references_dict_lookup:
            refs_full_text.append(f"{ref_key_in_dict} {references_dict_lookup[ref_key_in_dict]}")
        else:
            # If a reference number is cited but not found, add a placeholder or log an error.
            # For robustness, you might want to handle this case more explicitly.
            refs_full_text.append(f"[{num_str}] Reference text not found.")

    processed_json["References"] = refs_full_text
    return processed_json

# Function to robustly extract JSON from the LLM's text response
def extract_json_from_response(text: str) -> dict:
    """
    Extracts a JSON object from a string, trying to handle markdown code fences.
    """
    text = text.strip()
    
    # Try to find JSON within ```json ... ``` or ``` ... ```
    # Using re.DOTALL (via [\s\S]) to ensure a_is_right_now.s_between_curlies works across newlines.
    match_json_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if match_json_block:
        json_str = match_json_block.group(1)
    else:
        # Fallback to finding the first '{' to the last '}'
        # This is your original robust regex for a general JSON blob
        match_curly_braces = re.search(r"(\{[\s\S]*\})", text)
        if not match_curly_braces:
            raise ValueError("No JSON object found in LLM output. Output was: " + text)
        json_str = match_curly_braces.group(1)

    json_str = json_str.strip() # Ensure no leading/trailing whitespace in the extracted JSON string

    try:
        # Attempt to parse the extracted string as JSON
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # The error "Invalid control character" means json.loads itself can't handle it.
        # Prompt engineering is the primary way to fix this.
        # Auto-repairing arbitrary JSON errors from LLMs is complex and brittle.
        error_message = (
            f"Failed to decode JSON from LLM output. Error: {e}. "
            f"Problematic JSON string (after extraction attempts) was:\n>>>\n{json_str}\n<<<"
        )
        raise ValueError(error_message)



# API endpoint to assess a clinical case
@app.post("/clinical-assess")
def assess_case(request: ClinicalRequest):
    """
    API endpoint to process a clinical request and return a structured assessment.
    """
    # Format the user's input into a query string for the RAG chain.
    user_query = (
        f"Input:\n{request.dict()}\n"
        "Generate structured output as shown in the clinical decision support format provided in the prompt."
    )

    # Invoke the RAG chain
    try:
        result = rag_chain.invoke({"query": user_query}) # Use invoke for LCEL chains
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        return {"error": f"Error during RAG chain invocation: {str(e)}", "raw_output": None}


    llm_output_text = result.get("result", "")

    # Attempt to extract the JSON object from the LLM's response
    try:
        output_json = extract_json_from_response(llm_output_text)
    except ValueError as e:
        print(f"Error extracting JSON: {e}")
        # If JSON extraction fails, return an error and the raw LLM output for debugging.
        return {"error": str(e), "raw_output": llm_output_text, "source_documents": result.get("source_documents")}

    # Enforce the output contract (structure and references)
    try:
        final_output_json = enforce_contract(output_json, references_dict)
    except Exception as e:
        print(f"Error enforcing contract: {e}")
        return {"error": f"Error enforcing contract: {str(e)}", "raw_llm_json": output_json, "raw_output": llm_output_text}

    # Return the structured and validated JSON response.
    # Optionally, include source documents if needed for tracing or debugging.
    # return {"assessment": final_output_json, "source_documents": result.get("source_documents")}
    return final_output_json


# --- Main execution block to run the FastAPI app with Uvicorn ---
if __name__ == "__main__":
    print("Starting FastAPI server with Uvicorn...")
    # The host "0.0.0.0" makes the server accessible from other devices on the network.
    # Port 8000 is a common choice for web applications.
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # To access the API documentation (Swagger UI), navigate to http://localhost:8000/docs
    # or http://<your-machine-ip>:8000/docs in your web browser.