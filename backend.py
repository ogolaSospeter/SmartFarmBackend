import os
import re
import google.generativeai as genai  # Correct import
import flask 
from flask import request, jsonify
from dotenv import load_dotenv
import tensorflow as tf


app = flask.Flask(__name__)
load_dotenv()

# Load EfficientNet model
model_path = "efficientnetmodel.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load labels from the text file
with open('labels.txt', 'r') as f:
    labels = f.read().splitlines()

# Initialize the Gemini API client
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel("gemini-2.0-flash")

def get_tomato_disease_recommendations(disease, temperature, moisture):
    print(f"Getting recommendations for {disease} with temperature {temperature}°C and moisture {moisture}Hg.")

    _system_instruction = (
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
        "- **Brief (2 - 4 line) description -  first indicate the title, 'Brief Description'**\n"
        "- **Causative Agent **  - State the causativer agent of the disease, whether it is fungi/fungal infection, pests, etc.**\n"
        "- **Causes (max 6 bullets, min 3 bullets)** -- indicate the title, 'Causes' -- Begin numbering from 1\n"
        "- **Recommended Actions (max 7 bullets, min 4 bullets)** -- indicate the title, 'Recommended Actions' --  Begin numbering from 1\n" 
    )

    try:
        model = genai.GenerativeModel("gemini-2.0-flash",system_instruction = _system_instruction)
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

# Create a chat session (this maintains memory across interactions)
chat_sessions = {}

def send_prompt_to_gemini(user_id, user_prompt):
    _system_instruction = (
        "You are SmartFarmBot, a knowledgeable virtual assistant specialized in Tomato Farming. "
        "Your goal is to assist tomato farmers with accurate, factual, and practical advice. "
        "Maintain a human-like, simple, and conversational tone. "
        "Keep responses concise and relevant—avoid lengthy explanations unless absolutely necessary. "
        "If a farmer's question is unclear, ask a brief clarification question instead of making assumptions. "
    )

    # Retrieve or create a chat session for the user
    if user_id not in chat_sessions:
        model = genai.GenerativeModel("gemini-2.0-flash",system_instruction = _system_instruction)
        chat_sessions[user_id] = model.start_chat(history=[])  # New chat session with memory

    chat = chat_sessions[user_id]

    print(f"Sending prompt to Gemini AI: {user_prompt}")

    try:
        response = chat.send_message(user_prompt)  # Uses chat session memory
        newtext = response.text if hasattr(response, "text") and response.text else "I'm not sure how to respond."
        print("The received response was: " + newtext)

    except Exception as e:
        return {"message": f"Error: {str(e)}"}

    return {"message": newtext}

@app.route('/chat/<user_id>/<message>', methods=['GET'])
def chat(user_id, message):
    print(f"Received message from user {user_id}: {message}")
    user_message = message.strip()
    bot_reply = send_prompt_to_gemini(user_id, user_message)
    print(f"Sending reply: {bot_reply['message']}")
    return jsonify(bot_reply)

"""
This model is responsible for generating generalized tomato crops management and good farming practises.
It can provide recommendations for various tomato diseases, pests, and diseases, and it can also provide guidance on how to manage the tomatoes in various environments.

"""
def generate_management_practises():
    _system_instruction = (
        "As SmartFarmBot, you are a tomato management and good farming practises generator model. "
        "Your goal is to provide accurate, concise, and practical advice on tomato farming or management. "
        "Maintain a human-like, simple, and conversational tone. "
        "Keep responses concise and relevant—avoid lengthy explanations unless absolutely necessary. "
        "Your  generated content MUST be only for tomatoes farming practices."
        "Choose the topic yourself without requiring a specific user input. "
        "Ensure that your responses are practical, simple, and relevant for small scale farmers in Kenya. "
        "Do NOT generate long paragraphs—keep responses concise and straight to the point, unless where there is need to expound."
        "Where necessary, number the points in 1. ,2. ,3. ,"
    )
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash", system_instruction= _system_instruction)
        response = model.generate_content("Provide a useful tip for tomato farming.")
        newtext =  response.text if hasattr(response, "text") and response.text else "I'm not sure how to provide a practical tip."
        return {"management_practises": newtext}
    except Exception as e:
        return {"management_practises": f"Error: {str(e)}"}
    
@app.route('/managementpractises', methods=['GET'])
def get_management_practises():
    # print("Generating management practises")
    # print("Generated practices: \n{}".format(generate_management_practises()))
    return generate_management_practises()

# Integration of the Model for Image classification.
def classify_image(image_data):
    try:
        print("\n\nClassifying image...")
    # Preprocess the image to fit EfficientNet input
        image = Image.open(BytesIO(image_data))
        image = image.resize((224, 224))  # EfficientNet expects 224x224 images
        image = np.array(image).astype(np.float32)
        image = np.expand_dims(image, axis=0)
        # Normalize the image (efficientnet expects this preprocessing)
        image = image / 255.0
        # Get model input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("\n\nInput details: ", input_details)
        print("\n\nOutput details: ", output_details)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()

        # Get the classification result
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data[0])

        return labels[predicted_class]  # Return the label of the predicted class
    except Exception as e:
        print("\n\nError in image classification: ", str(e))
        return "Error in classification"

# Endpoint to handle image upload and classify disease
@app.route('/classify-disease', methods=['POST'])
def classify_disease():
    try:
        print("\n\nReceived request to classify disease")
        # Ensure the request contains an image
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        image_data = image_file.read()

        # Classify the image
        predicted_disease = classify_image(image_data)
        print(f"\n\nPredicted disease: {predicted_disease}")

        return jsonify({"predicted_disease": predicted_disease})

    except Exception as e:
        print("An error has occured: ", str(e))
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug = False)
