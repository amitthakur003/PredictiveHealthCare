import google.generativeai as genai
import os
from dotenv import load_dotenv
from config.settings import AI_CONFIG

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(AI_CONFIG['model_name'])
else:
    model = None

def get_ai_insights(question, data_summary):
    """Get AI-powered insights using Gemini API"""
    if not model:
        return "AI assistant unavailable. Configure GOOGLE_API_KEY in .env"
    
    try:
        prompt = f"""
You are an expert healthcare claims analyst AI assistant.

Current Data Summary:
{data_summary}

User Question: {question}

Provide clear, actionable insights focusing on:
1. Explaining predictions/patterns simply
2. Identifying risks/opportunities
3. Suggesting recommendations
4. Using healthcare domain knowledge

Keep under {AI_CONFIG['max_response_length']} words.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"