import os
import re
import asyncio
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Pydantic Models ---

class PatientDataInput(BaseModel):
    drugs: List[str] = Field(..., example=["warfarin", "ibuprofen"])
    age: int = Field(..., example=65)
    gender: str = Field(..., example="Male")
    allergies: Optional[List[str]] = Field(default=[], example=["penicillin"])
    diagnosis: str = Field(..., example="I10 (Hypertension)")

class AlertDetail(BaseModel):
    question: str
    answer: str

# The response model will be a dictionary where keys are alert categories (strings)
# and values are AlertDetail objects. FastAPI handles Dict[str, AlertDetail] directly.

# --- Groq Client Initialization ---
# IMPORTANT: Set the GROQ_API_KEY environment variable before running the application.
# export GROQ_API_KEY='your_actual_api_key'
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    # In a real application, you might want to raise a more specific error or log this.
    # For now, we'll let Groq client handle it or raise an error if it's used without a key.
    print("Warning: GROQ_API_KEY environment variable not set. Ensure .env file is present and correctly formatted or an environment variable is set.")
    # raise ValueError("GROQ_API_KEY environment variable not set. The application cannot start.")

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama3-70b-8192" # "mixtral-8x7b-32768" is another option if needed

# --- Helper Functions ---

def clean_answer(text: str) -> str:
    """Cleans the raw answer from the AI model."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "Answer:" in cleaned:
        cleaned = cleaned.split("Answer:")[-1]
    cleaned = cleaned.strip().replace("\\n", " ")
    return cleaned

async def generate_alerts_for_patient(patient_data: PatientDataInput) -> Dict[str, str]:
    """
    Generates clinical alerts for a single patient by querying the Groq API.
    """
    def q(text: str) -> str:
        # Helper to include age/gender in every question
        return f"For a {patient_data.age}-year-old {patient_data.gender.lower()} patient: {text}"

    drug_list_str = ', '.join(patient_data.drugs)
    allergy_list_str = ', '.join(patient_data.allergies) if patient_data.allergies else ""

    questions = {}

    questions["Drug-Drug Interactions"] = q(f"What is the main risk of combining {drug_list_str}?")
    
    if patient_data.allergies:
        questions["Drug-Allergy"] = q(f"Do {drug_list_str} conflict with a {allergy_list_str} allergy?")
    else:
        questions["Drug-Allergy"] = "N/A" # Or None, handled below

    questions["Drug-Disease Contraindications"] = q(f"Do {drug_list_str} worsen {patient_data.diagnosis}?")
    questions["Ingredient Duplication"] = q(f"Do {drug_list_str} contain overlapping active ingredients?")
    # Updated General Precautions as per the user's example
    questions["General Precautions"] = q(f"What should this patient be cautious about when using {drug_list_str}?")
    questions["Therapeutic Class Conflicts"] = q(f"Are there therapeutic class conflicts between {drug_list_str}?")
    questions["Warning Labels"] = q(f"What is the key warning label for {drug_list_str}?")

    if patient_data.gender.lower() == "female":
        if 12 <= patient_data.age <= 50: # Common child-bearing age range
            questions["Pregnancy Warnings"] = q(f"Are {drug_list_str} safe during pregnancy?")
        else:
            questions["Pregnancy Warnings"] = "N/A" # Or None
        questions["Lactation Warnings"] = q(f"Are {drug_list_str} safe during lactation?")
    else:
        questions["Pregnancy Warnings"] = "N/A" # Or None
        questions["Lactation Warnings"] = "N/A" # Or None
    
    alerts: Dict[str, str] = {}

    for category, question_text in questions.items():
        if not question_text or question_text == "N/A":
            alerts[category] = "N/A"
        else:
            try:
                chat_completion = await asyncio.to_thread(
                    client.chat.completions.create,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a clinical assistant. Answer only with the key clinical risk or note in one short sentence. Do not include explanations or extra details."
                        },
                        {
                            "role": "user",
                            "content": question_text
                        }
                    ],
                    model=MODEL_NAME,
                )
                raw_answer = chat_completion.choices[0].message.content
                final_answer = clean_answer(raw_answer)
                alerts[category] = final_answer
            except Exception as e:
                # Log the error e
                print(f"Error calling Groq API for category {category}: {e}")
                alerts[category] = "Error retrieving answer from AI."

    return alerts

# --- FastAPI Application ---

app = FastAPI(
    title="Clinical Decision Support System API",
    description="Provides clinical alerts based on patient data using AI.",
    version="1.0.0"
)

@app.post("/generate_clinical_alerts", response_model=Dict[str, str])
async def create_clinical_alerts(patient_input: PatientDataInput):
    """
    Receives patient data, generates clinical questions, queries an AI model,
    and returns potential clinical alerts.
    """
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="GROQ_API_KEY not configured on the server. Ensure .env file is present and correctly formatted or an environment variable is set. Cannot process request."
        )
    try:
        alerts = await generate_alerts_for_patient(patient_input)
        return alerts
    except Exception as e:
        # Log the exception e
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# To run this application:
# 1. Save it as main.py
# 2. Set your Groq API key: export GROQ_API_KEY='your_gsk_...'
#    Alternatively, create a .env file in the same directory as main.py with the line:
#    GROQ_API_KEY='your_gsk_...'
# 3. Install FastAPI, Uvicorn, and python-dotenv: pip install fastapi uvicorn python-dotenv
# 4. Run Uvicorn: uvicorn main:app --reload
#
# Example aiohttp POST request (or use curl, Postman, etc.):
#
# import asyncio
# import aiohttp
# import json
# 
# async def main_request():
#     url = "http://127.0.0.1:8000/generate_clinical_alerts"
#     payload = {
#         "drugs": ["lisinopril", "ibuprofen"],
#         "age": 30,
#         "gender": "Female",
#         "allergies": ["sulfa drugs"],
#         "diagnosis": "Pregnancy (first trimester)"
#     }
#     async with aiohttp.ClientSession() as session:
#         async with session.post(url, json=payload) as response:
#             print("Status:", response.status)
#             print("Content-type:", response.headers['content-type'])
#             html = await response.text()
#             print("Body:", html[:500] + "...") # Print first 500 chars
#             # To parse JSON:
#             # response_json = await response.json()
#             # print("Parsed JSON:", json.dumps(response_json, indent=2))
# 
# if __name__ == "__main__":
#    # This part is for example client request, not part of the FastAPI app itself.
#    # To run the client example, you'd typically run it separately after starting the server.
#    # asyncio.run(main_request())
#    pass 