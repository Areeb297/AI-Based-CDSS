# CDSS Functions and Components Documentation

## Overview

This document provides comprehensive documentation for all internal functions, components, and utilities used in the Clinical Decision Support System (CDSS) application.

## Table of Contents

1. [Core Components](#core-components)
2. [RAG Pipeline Components](#rag-pipeline-components)
3. [Utility Functions](#utility-functions)
4. [LLM Integration Functions](#llm-integration-functions)
5. [Data Processing Functions](#data-processing-functions)
6. [Prompt Templates](#prompt-templates)
7. [Configuration Management](#configuration-management)

## Core Components

### FastAPI Application Instance

```python
app = FastAPI(
    title="Clinical Decision Support API",
    description="API for assessing clinical cases using RAG.",
    version="1.0.1"
)
```

**Description:** Main FastAPI application instance with metadata configuration.

**Features:**
- Auto-generated OpenAPI documentation
- Built-in validation and serialization
- Async request handling
- Error handling middleware

### Vector Store and Embeddings

```python
# OpenAI Embeddings Configuration
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=actual_openai_api_key
)

# FAISS Vector Store
vectordb = FAISS.from_documents(filtered_doc_chunks_for_faiss, embeddings)
```

**Description:** Core vector storage and embedding components for semantic search.

**Key Features:**
- Text embedding using OpenAI's ada-002 model
- FAISS vector store for efficient similarity search
- Persistent storage with save/load functionality
- Configurable search parameters

## RAG Pipeline Components

### 1. Document Loading and Processing

#### `load_pdf_documents()`

```python
def load_pdf_documents(pdf_path: str) -> List[Document]:
    """
    Load and process PDF documents for RAG pipeline.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        List[Document]: List of processed document chunks
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        RuntimeError: If PDF processing fails
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pdf_document_pages = loader.load_and_split()
        return pdf_document_pages
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading PDF: {e}")
```

#### `split_documents()`

```python
def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for processing.
    
    Args:
        documents (List[Document]): List of document pages
        
    Returns:
        List[Document]: List of chunked documents
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100
    )
    return splitter.split_documents(documents)
```

#### `filter_document_chunks()`

```python
def filter_document_chunks(doc_chunks: List[Document]) -> List[Document]:
    """
    Filter document chunks to ensure valid content.
    
    Args:
        doc_chunks (List[Document]): Raw document chunks
        
    Returns:
        List[Document]: Filtered valid document chunks
        
    Raises:
        ValueError: If no valid chunks found
    """
    filtered_chunks = []
    
    for i, chunk in enumerate(doc_chunks):
        content = chunk.page_content
        metadata = chunk.metadata
        
        if isinstance(content, str) and content.strip():
            filtered_chunks.append(chunk)
        else:
            # Log warning for invalid chunks
            page_info = metadata.get('page', 'N/A') if metadata else 'N/A'
            print(f"Warning: Skipping chunk {i} from page {page_info}")
    
    if not filtered_chunks:
        raise ValueError("No valid document chunks found")
    
    return filtered_chunks
```

### 2. Vector Store Management

#### `create_or_load_vector_store()`

```python
def create_or_load_vector_store(
    faiss_index_path: str, 
    documents: List[Document], 
    embeddings: OpenAIEmbeddings
) -> FAISS:
    """
    Create new or load existing FAISS vector store.
    
    Args:
        faiss_index_path (str): Path to FAISS index directory
        documents (List[Document]): Document chunks for indexing
        embeddings (OpenAIEmbeddings): Embedding model
        
    Returns:
        FAISS: Configured vector store instance
    """
    index_file = Path(faiss_index_path) / "index.faiss"
    pkl_file = Path(faiss_index_path) / "index.pkl"
    
    if index_file.exists() and pkl_file.exists():
        print(f"Loading existing FAISS index from {faiss_index_path}")
        return FAISS.load_local(
            faiss_index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        print(f"Building new FAISS index at {faiss_index_path}")
        vectordb = FAISS.from_documents(documents, embeddings)
        vectordb.save_local(faiss_index_path)
        return vectordb
```

### 3. RAG Chain Configuration

#### `create_rag_chain()`

```python
def create_rag_chain(
    llm: ChatOpenAI, 
    vectordb: FAISS, 
    prompt: PromptTemplate,
    search_k: int = 12
) -> RetrievalQA:
    """
    Create RetrievalQA chain for RAG pipeline.
    
    Args:
        llm (ChatOpenAI): Language model instance
        vectordb (FAISS): Vector store for retrieval
        prompt (PromptTemplate): Prompt template for formatting
        search_k (int): Number of documents to retrieve
        
    Returns:
        RetrievalQA: Configured RAG chain
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": search_k}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
```

## Utility Functions

### 1. JSON Processing

#### `extract_json_from_response()`

```python
def extract_json_from_response(text: str) -> dict:
    """
    Extract JSON object from LLM response text.
    
    Args:
        text (str): Raw LLM response text
        
    Returns:
        dict: Parsed JSON object
        
    Raises:
        ValueError: If no valid JSON found or parsing fails
        
    Examples:
        >>> extract_json_from_response('```json\n{"key": "value"}\n```')
        {'key': 'value'}
        
        >>> extract_json_from_response('Some text {"key": "value"} more text')
        {'key': 'value'}
    """
    text = text.strip()
    
    # Try to find JSON in code blocks first
    match_json_block = re.search(
        r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", 
        text, 
        re.IGNORECASE
    )
    
    if match_json_block:
        json_str = match_json_block.group(1)
    else:
        # Look for JSON object in text
        match_curly_braces = re.search(r"(\{[\s\S]*\})", text)
        if not match_curly_braces:
            raise ValueError(f"No JSON object found in LLM output: {text}")
        json_str = match_curly_braces.group(1)
    
    try:
        return json.loads(json_str.strip())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to decode JSON: {e}. "
            f"Problematic JSON: {json_str}"
        )
```

### 2. Contract Enforcement

#### `enforce_contract()`

```python
def enforce_contract(
    output_json: dict, 
    references_dict_lookup: dict,
    expected_keys: List[str] = None
) -> dict:
    """
    Enforce output contract and populate references.
    
    Args:
        output_json (dict): Raw LLM output JSON
        references_dict_lookup (dict): Reference database
        expected_keys (List[str]): Required output keys
        
    Returns:
        dict: Processed JSON with enforced contract
        
    Examples:
        >>> output = {"Drug-Drug Interactions": ["Risk of bleeding [1]"]}
        >>> refs = {"[1]": "FDA label for warfarin"}
        >>> enforce_contract(output, refs)
        {
            "Drug-Drug Interactions": ["Risk of bleeding [1]"],
            "References": ["[1] FDA label for warfarin"]
        }
    """
    if expected_keys is None:
        expected_keys = EXPECTED_KEYS
    
    processed_json = {}
    
    # Ensure all expected keys are present
    for key in expected_keys:
        if key == "References":
            processed_json[key] = []
        else:
            processed_json[key] = output_json.get(key, ["None"])
    
    # Extract reference numbers from content
    ref_nums_extracted = set()
    for key, values in processed_json.items():
        if key == "References":
            continue
        if isinstance(values, list):
            for value_item in values:
                if isinstance(value_item, str):
                    found_numbers = re.findall(r"\[(\d+)\]", value_item)
                    ref_nums_extracted.update(found_numbers)
    
    # Build references list
    refs_full_text = []
    for num_str in sorted(ref_nums_extracted, key=int):
        ref_key = f"[{num_str}]"
        if ref_key in references_dict_lookup:
            refs_full_text.append(
                f"{ref_key} {references_dict_lookup[ref_key]}"
            )
        else:
            refs_full_text.append(f"{ref_key} Reference text not found.")
    
    processed_json["References"] = refs_full_text
    return processed_json
```

### 3. Validation Functions

#### `validate_clinical_request()`

```python
def validate_clinical_request(request: ClinicalRequest) -> bool:
    """
    Validate clinical request parameters.
    
    Args:
        request (ClinicalRequest): Clinical request object
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If validation fails
        
    Examples:
        >>> request = ClinicalRequest(
        ...     drugs=["aspirin"], 
        ...     age=25, 
        ...     gender="Female"
        ... )
        >>> validate_clinical_request(request)
        True
    """
    if not request.drugs:
        raise ValueError("At least one drug must be specified")
    
    if request.age < 0 or request.age > 150:
        raise ValueError("Age must be between 0 and 150")
    
    if request.gender not in ["Male", "Female"]:
        raise ValueError("Gender must be 'Male' or 'Female'")
    
    return True
```

## LLM Integration Functions

### 1. LLM Initialization

#### `initialize_groq_llm()`

```python
def initialize_groq_llm(
    api_key: str, 
    model: str = "llama3-70b-8192",
    temperature: float = 0.0
) -> ChatOpenAI:
    """
    Initialize Groq LLM instance.
    
    Args:
        api_key (str): Groq API key
        model (str): Model name
        temperature (float): Sampling temperature
        
    Returns:
        ChatOpenAI: Configured LLM instance
        
    Raises:
        ValueError: If API key is missing
    """
    if not api_key:
        raise ValueError("Groq API key is required")
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base="https://api.groq.com/openai/v1"
    )
```

#### `initialize_openai_embeddings()`

```python
def initialize_openai_embeddings(
    api_key: str,
    model: str = "text-embedding-ada-002"
) -> OpenAIEmbeddings:
    """
    Initialize OpenAI embeddings instance.
    
    Args:
        api_key (str): OpenAI API key
        model (str): Embedding model name
        
    Returns:
        OpenAIEmbeddings: Configured embeddings instance
        
    Raises:
        ValueError: If API key is missing
    """
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=api_key
    )
```

### 2. OpenRouter Integration

#### `make_openrouter_request()`

```python
def make_openrouter_request(
    api_key: str,
    model: str,
    messages: List[dict],
    tools: List[dict] = None,
    max_tokens: int = 3000,
    temperature: float = 0.0
) -> dict:
    """
    Make request to OpenRouter API.
    
    Args:
        api_key (str): OpenRouter API key
        model (str): Model name (e.g., "openai/gpt-4o")
        messages (List[dict]): Chat messages
        tools (List[dict]): Function tools
        max_tokens (int): Maximum response tokens
        temperature (float): Sampling temperature
        
    Returns:
        dict: API response
        
    Raises:
        requests.RequestException: If API request fails
        
    Examples:
        >>> messages = [{"role": "user", "content": "Hello"}]
        >>> response = make_openrouter_request(
        ...     api_key="your-key",
        ...     model="openai/gpt-4o", 
        ...     messages=messages
        ... )
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = {
            "type": "function", 
            "function": {"name": tools[0]["function"]["name"]}
        }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    return response.json()
```

## Data Processing Functions

### 1. Image Processing

#### `encode_image_to_base64()`

```python
def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64 string.
    
    Args:
        image_bytes (bytes): Raw image bytes
        
    Returns:
        str: Base64 encoded image string
        
    Examples:
        >>> with open("image.jpg", "rb") as f:
        ...     image_bytes = f.read()
        >>> b64_image = encode_image_to_base64(image_bytes)
        >>> b64_image.startswith("iVBORw0KGgo")  # PNG example
        True
    """
    return base64.b64encode(image_bytes).decode('utf-8')
```

#### `classify_image_type()`

```python
def classify_image_type(
    api_key: str,
    image_b64: str,
    model: str = "openai/gpt-4o"
) -> str:
    """
    Classify X-ray image type using OpenRouter API.
    
    Args:
        api_key (str): OpenRouter API key
        image_b64 (str): Base64 encoded image
        model (str): Model for classification
        
    Returns:
        str: "Chest" or "Skeletal"
        
    Raises:
        ValueError: If classification fails
        
    Examples:
        >>> image_type = classify_image_type(
        ...     api_key="your-key",
        ...     image_b64="base64-string"
        ... )
        >>> image_type in ["Chest", "Skeletal"]
        True
    """
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "text", 
                "text": "Is this a 'Chest' X-ray or a 'Skeletal' X-ray? Respond with only one of these two words."
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            }
        ]
    }]
    
    response = make_openrouter_request(
        api_key=api_key,
        model=model,
        messages=messages,
        max_tokens=10
    )
    
    result = response['choices'][0]['message']['content']
    
    if 'chest' in result.lower():
        return "Chest"
    elif 'skeletal' in result.lower():
        return "Skeletal"
    else:
        raise ValueError(f"Could not classify image type: {result}")
```

### 2. Report Generation

#### `generate_radiology_report()`

```python
def generate_radiology_report(
    api_key: str,
    image_b64: str,
    image_type: str,
    clinical_query: str = "",
    model: str = "openai/gpt-4o"
) -> dict:
    """
    Generate structured radiology report.
    
    Args:
        api_key (str): OpenRouter API key
        image_b64 (str): Base64 encoded image
        image_type (str): "Chest" or "Skeletal"
        clinical_query (str): Clinical context
        model (str): Model for analysis
        
    Returns:
        dict: Structured radiology report
        
    Raises:
        ValueError: If report generation fails
        
    Examples:
        >>> report = generate_radiology_report(
        ...     api_key="your-key",
        ...     image_b64="base64-string",
        ...     image_type="Chest",
        ...     clinical_query="Patient with chest pain"
        ... )
        >>> "View" in report
        True
    """
    # Select appropriate prompts and tools based on image type
    if image_type == "Chest":
        system_prompt = CHEST_SYSTEM_PROMPT
        user_prompt = CHEST_USER_PROMPT
        tools = [CHEST_TOOL]
        required_keys = ["View", "Pneumothorax", "Findings", "Impression", "Recommendations"]
    else:  # Skeletal
        system_prompt = SKELETAL_SYSTEM_PROMPT
        user_prompt = SKELETAL_USER_PROMPT
        tools = [SKELETAL_TOOL]
        required_keys = ["View", "Findings", "Impression", "Recommendations"]
    
    # Prepare messages
    full_user_message = user_prompt
    if clinical_query:
        full_user_message += f"\n\nClinical Information: {clinical_query}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": full_user_message},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]}
    ]
    
    # Make API request
    response = make_openrouter_request(
        api_key=api_key,
        model=model,
        messages=messages,
        tools=tools
    )
    
    # Extract structured report
    if response.get("choices") and response["choices"][0].get("message", {}).get("tool_calls"):
        tool_call = response["choices"][0]["message"]["tool_calls"][0]
        if tool_call["function"]["name"] == "create_radiology_report":
            report_data = json.loads(tool_call["function"]["arguments"])
            
            # Ensure all required keys are present
            for key in required_keys:
                report_data.setdefault(key, "No information provided by model.")
            
            return report_data
    
    raise ValueError("Failed to generate structured report")
```

## Prompt Templates

### 1. Clinical Assessment Prompt

```python
CLINICAL_ASSESSMENT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
IMPORTANT INSTRUCTIONS:
- Your output MUST include ALL of these categories in exact order:
  "Drug-Drug Interactions", "Drug-Allergy", "Drug-Disease Contraindications",
  "Ingredient Duplication", "Pregnancy Warnings", "Lactation Warnings",
  "General Precautions", "Therapeutic Class Conflicts", "Warning Labels",
  "Indications", "Severity", "References"
- Every fact MUST cite reference numbers from context [1][2]
- If no relevant content, use ["None"]
- For male patients: "None (not applicable in male patients). [3]"
- Severity: ["High"], ["Moderate"], or ["Low"]
- References must include every cited number with full citation

Clinical Query: {question}
Context: {context}

Output only valid JSON object.
"""
)
```

### 2. Clinical Pathways Prompt

```python
CLINICAL_PATHWAYS_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Generate 3 clinical pathways in JSON format:
- "Option 1" (lowest risk, guideline-adherent)
- "Option 2" (medium risk, with monitoring)
- "Option 3" (high risk, not recommended)

Each pathway includes:
- "recommendation": Exact drug, dose, route, frequency, duration
- "details": Clinical rationale
- "tradeoff": Pros and cons
- "monitoring": Specific monitoring requirements

Cite references [1][2] from context.
For Saudi Arabia: No alcohol mentions, use local medications.

Scenario: {question}
Context: {context}
"""
)
```

### 3. Radiology Prompts

#### Chest X-ray Prompt

```python
CHEST_SYSTEM_PROMPT = """
You are an expert radiologist focused on emergency chest radiography.
Your mission is to identify life-threatening conditions, especially pneumothorax.
You must be meticulous and err on the side of caution.
"""

CHEST_USER_PROMPT = """
Follow two-pass workflow:

PASS 1 – Pneumothorax Check:
1. Acknowledge: "I understand a subtle pneumothorax may be present"
2. Trace LEFT LUNG periphery from apex downwards
3. Look for visceral pleural line separated from chest wall
4. Note absence of lung markings outside this line
5. Conclude: "✓ Pneumothorax detected" or "✗ No pneumothorax"

PASS 2 – Structured Report:
Use exact five-section format with pneumothorax finding.
"""
```

#### Skeletal X-ray Prompt

```python
SKELETAL_SYSTEM_PROMPT = """
You are an expert radiologist specializing in emergency and diagnostic radiology.
Reports are professional, concise, and clinically focused.
CRITICAL: Patient's right side appears on left side of image and vice versa.
"""

SKELETAL_USER_PROMPT = """
Review X-ray in TWO STEPS:

PASS 1 – Chain-of-Thought Analysis:
For each bone/joint, show your work:
1. "I trace the [bone] cortex and observe..."
2. Note lucent lines, cortical step-offs, or obscuration
3. Conclude "✓ Intact" or "✗ Possible fracture"

PASS 2 – Structured Report:
Generate formal four-section report.
"""
```

## Configuration Management

### 1. Environment Configuration

#### `load_environment_config()`

```python
def load_environment_config() -> dict:
    """
    Load and validate environment configuration.
    
    Returns:
        dict: Configuration dictionary
        
    Raises:
        ValueError: If required environment variables are missing
        
    Examples:
        >>> config = load_environment_config()
        >>> "GROQ_API_KEY" in config
        True
    """
    load_dotenv()
    
    config = {
        "ACTUAL_OPENAI_API_KEY": os.getenv("ACTUAL_OPENAI_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "PORT": int(os.getenv("PORT", 10000))
    }
    
    # Validate required keys
    required_keys = ["ACTUAL_OPENAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"]
    missing_keys = [key for key in required_keys if not config[key]]
    
    if missing_keys:
        raise ValueError(f"Missing required environment variables: {missing_keys}")
    
    return config
```

### 2. File Path Configuration

#### `get_file_paths()`

```python
def get_file_paths() -> dict:
    """
    Get configured file paths.
    
    Returns:
        dict: Dictionary of file paths
        
    Examples:
        >>> paths = get_file_paths()
        >>> paths["pdf_path"]
        'CDSS_Expected_Outputs_Cases_ALL.pdf'
    """
    return {
        "pdf_path": "CDSS_Expected_Outputs_Cases_ALL.pdf",
        "ref_path": "references_list_key_brackets_value_plain.json",
        "faiss_index_path": "faiss_cdss_pdf"
    }
```

### 3. Global Constants

```python
# Expected output structure keys
EXPECTED_KEYS = [
    "Drug-Drug Interactions", "Drug-Allergy", "Drug-Disease Contraindications",
    "Ingredient Duplication", "Pregnancy Warnings", "Lactation Warnings",
    "General Precautions", "Therapeutic Class Conflicts", "Warning Labels",
    "Indications", "Severity", "References"
]

# Radiology tool definitions
CHEST_TOOL = {
    "type": "function",
    "function": {
        "name": "create_radiology_report",
        "description": "Creates a structured chest radiology report.",
        "parameters": {
            "type": "object",
            "properties": {
                "View": {"type": "string"},
                "Pneumothorax": {"type": "string"},
                "Findings": {"type": "string"},
                "Impression": {"type": "string"},
                "Recommendations": {"type": "string"}
            },
            "required": ["View", "Pneumothorax", "Findings", "Impression", "Recommendations"]
        }
    }
}

SKELETAL_TOOL = {
    "type": "function",
    "function": {
        "name": "create_radiology_report",
        "description": "Creates a structured skeletal radiology report.",
        "parameters": {
            "type": "object",
            "properties": {
                "View": {"type": "string"},
                "Findings": {"type": "string"},
                "Impression": {"type": "string"},
                "Recommendations": {"type": "string"}
            },
            "required": ["View", "Findings", "Impression", "Recommendations"]
        }
    }
}
```

## Error Handling Functions

### 1. Custom Exception Classes

```python
class CDSSException(Exception):
    """Base exception for CDSS application."""
    pass

class RAGPipelineError(CDSSException):
    """Exception raised for RAG pipeline errors."""
    pass

class LLMServiceError(CDSSException):
    """Exception raised for LLM service errors."""
    pass

class ValidationError(CDSSException):
    """Exception raised for validation errors."""
    pass
```

### 2. Error Handling Decorators

```python
def handle_api_errors(func):
    """
    Decorator to handle common API errors.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.RequestException as e:
            return JSONResponse(
                content={"error": "API request failed", "details": str(e)},
                status_code=500
            )
        except json.JSONDecodeError as e:
            return JSONResponse(
                content={"error": "Invalid JSON response", "details": str(e)},
                status_code=500
            )
        except Exception as e:
            return JSONResponse(
                content={"error": "Internal server error", "details": str(e)},
                status_code=500
            )
    return wrapper
```

## Performance Optimization

### 1. Caching Functions

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_cached_embeddings(text: str) -> List[float]:
    """
    Get cached embeddings for text.
    
    Args:
        text (str): Input text
        
    Returns:
        List[float]: Embedding vector
    """
    # Implementation would call embedding service
    pass

@lru_cache(maxsize=64)
def get_cached_references(ref_numbers: tuple) -> List[str]:
    """
    Get cached reference texts.
    
    Args:
        ref_numbers (tuple): Reference numbers
        
    Returns:
        List[str]: Reference texts
    """
    # Implementation would look up references
    pass
```

### 2. Async Functions

```python
import asyncio
from typing import List, Coroutine

async def process_multiple_requests(
    requests: List[dict]
) -> List[dict]:
    """
    Process multiple clinical requests concurrently.
    
    Args:
        requests (List[dict]): List of clinical requests
        
    Returns:
        List[dict]: List of responses
    """
    tasks = [process_single_request(req) for req in requests]
    return await asyncio.gather(*tasks)

async def process_single_request(request: dict) -> dict:
    """
    Process single clinical request asynchronously.
    
    Args:
        request (dict): Clinical request
        
    Returns:
        dict: Clinical assessment result
    """
    # Implementation would call RAG chain
    pass
```

This comprehensive documentation covers all the major functions, components, and utilities in the CDSS application. Each function is documented with parameters, return values, examples, and error handling information to facilitate easy understanding and maintenance.