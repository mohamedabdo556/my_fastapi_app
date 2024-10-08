from fastapi import FastAPI, Path
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure the Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model
model = genai.GenerativeModel('gemini-pro')

# Create FastAPI instance
app = FastAPI()

# Available languages and enhancement types
AVAILABLE_LANGUAGES = ["Spanish", "French", "English", "Arabic"]
AVAILABLE_ENHANCEMENT_TYPES = ["Formal", "Friendly", "Concise", "Detailed"]

@app.get("/")
def read_root():
    return {"message": "Welcome to the ExpressAbleAI API"}

@app.post("/translate/{text}/{target_language}")
def translate(
    text: str = Path(..., description="The text to be translated."),
    target_language: str = Path(..., description="[Spanish, French, English, Arabic]")
):
    # Validate input
    if target_language not in AVAILABLE_LANGUAGES:
        return {"Error": "Invalid target_language. Please provide a valid language."}

    # Perform translation
    try:
        prompt = f"Translate the following text into {target_language}:\n\n{text}"
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=512,
            )
        )
        return {"translated_text": response.text}
    except Exception as e:
        return {"Error": f"Translation failed: {str(e)}"}

@app.post("/enhance/{text}/{enhancement_type}")
def enhance(
    text: str = Path(..., description="The text to be enhanced."),
    enhancement_type: str = Path(..., description="Formal, Friendly, Concise, Detailed]")
):
    # Validate input
    if enhancement_type not in AVAILABLE_ENHANCEMENT_TYPES:
        return {"Error": "Invalid enhancement_type. Please provide a valid enhancement type."}

    # Perform enhancement
    try:
        prompt = f"Enhance the following text to make it more {enhancement_type.lower()}:\n\n{text}"
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=512,
            )
        )
        return {"enhanced_text": response.text}
    except Exception as e:
        return {"Error": f"Enhancement failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
