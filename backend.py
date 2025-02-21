import os
import re
import google.generativeai as genai  # Correct import
import flask 
from flask import request, jsonify
from dotenv import load_dotenv

app = flask.Flask(__name__)
load_dotenv()

# Initialize the Gemini API client
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def get_tomato_disease_recommendations(disease, temperature, moisture):
    print(f"Getting recommendations for {disease} with temperature {temperature}°C and moisture {moisture}Hg.")

    system_instruction = (
        "You are SmartFarm, an expert in tomato farming. "
        "Your responses MUST be **concise**"
        "No additional explanations, only structured numbered points."
        "Do not use the asteriks, instead, make the text Bold, and use numbered list, of 1. 2. 3. etc for the infos."
        "The recommendations in line with the environmental conditions in the response must be tailored to reflect the conditions, not generalized."
    )
    
    prompt = (
        f"My tomatoes have {disease}. "
        f"The farm conditions are Temp={temperature}°C, Moisture={moisture}Hg. "
        "Provide a **very concise** response with:\n\n"
        "- **Brief (2-line) description -  first indicate the title, 'Brief Description'**\n"
        "- **Causes (max 6 bullets, min 3 bullets)** -- indicate the title, 'Causes'\n"
        "- **Recommended Actions (max 6 bullets, min 3 bullets)** -- indicate the title, 'Recommended Actions'\n"
        
    )

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        extracted_text = response.text if hasattr(response, "text") else "Error processing the response."

        # Ensure it's not exceeding limits
        extracted_text = "\n".join(extracted_text.split("\n")) 
        formatted_text = format_recommendations(extracted_text)

    except Exception as e:
        formatted_text = f"Error: {str(e)}"

    return {"recommendations": formatted_text }

@app.route('/recommendations/<disease>/<temperature>/<moisture>', methods=['GET'])
def get_recommendations(disease, temperature, moisture):
    temperature = float(temperature)
    moisture = float(moisture)
    
    recommendations = get_tomato_disease_recommendations(disease, temperature, moisture)
    return flask.jsonify(recommendations)


def format_recommendations(text):
    # Convert **bold text** to <b>bold text</b>
    formatted_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

    # Replace bulleted points (*) with numbered points
    lines = formatted_text.split("\n")
    numbered_text = []
    counter = 1

    for line in lines:
        if line.strip().startswith("*"):
            numbered_text.append(f"{counter}. {line.strip()[1:].strip()}")
            counter += 1
        else:
            numbered_text.append(line)

    return "\n".join(numbered_text)



def send_prompt_to_gemini(user_prompt):
    print(f"Sending prompt to Gemini AI: {user_prompt}")
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(user_prompt)
        newtext =  response.text if hasattr(response, "text") and response.text else "I'm not sure how to respond."
        print("The received response was: " + newtext)
    except Exception as e:
        return f"Error: {str(e)}"
    
    return {"message": newtext}

@app.route('/chat/<message>', methods=['GET'])
def chat(message):
    print(f"Received message: {message}")
    user_message = message.strip()
    bot_reply = send_prompt_to_gemini(user_message)
    print(f"Sending reply: {bot_reply['message']}")
    return flask.jsonify(bot_reply)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug = False)
