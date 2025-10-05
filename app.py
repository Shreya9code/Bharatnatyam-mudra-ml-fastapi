from fastapi import FastAPI, UploadFile, File,Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import mediapipe as mp
from pydantic import BaseModel, Field
from google import genai 
from dotenv import load_dotenv
import json # To parse the Gemini output
import sys # For error logging
import os
import io
import base64

# Load environment variables from .env file
load_dotenv()

# Gemini client initialization
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    GEMINI_MODEL = "gemini-2.5-flash"
except Exception as e:
    print(f"Warning: Could not initialize Gemini client. Ensure API key is set. Error: {e}", file=sys.stderr)
    client = None

# App initialization
app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ML model
model = tf.keras.models.load_model("best_mudra_model.keras")
IMG_SIZE = (128, 128)
class_names = ['Alapadmam', 'Anjali', 'Aralam', 'Ardhachandran', 'Ardhapathaka',
               'Berunda', 'Bramaram', 'Chakra', 'Chandrakala', 'Chaturam',
               'Garuda', 'Hamsapaksha', 'Hamsasyam', 'Kangulam', 'Kapith',
               'Kapotham', 'Karkatta', 'Kartariswastika', 'Kartrimukha', 'Katakamukha',
               'Katakavardhana', 'Katrimukha', 'Khatva', 'Kilaka', 'Kurma',
               'Matsya', 'Mayura', 'Mrigasirsha', 'Mukulam', 'Mushti',
               'Nagabandha', 'Padmakosha', 'Pasha', 'Pathaka', 'Pushpaputa',
               'Sakata', 'Samputa', 'Sandamsha', 'Sarpasirsha', 'Shanka',
               'Shivalinga', 'Shukatundam', 'Sikharam', 'Simhamukham', 'Suchi',
               'Swastikam', 'Tamarachudam', 'Tripathaka', 'Trishulam', 'Varaha']

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Pydantic Model for Structured Gemini Output ---

# Define the data structure we want Gemini to return
class MudraDetails(BaseModel):
    meaning: str = Field(description="A single paragraph explanation of the mudra's meaning and cultural significance.")
    innerThought: str = Field(description="A short, concise phrase representing the emotion or concept associated with the mudra.")
    commonMistakes: list[str] = Field(description="A list of 3 common mistakes beginners make when performing this mudra.")

class HandAnalysisResponse(BaseModel):
    num_hands: int
    finger_confidence: dict
    mudra_predictions: list
    annotated_image: str = None  # base64 encoded image

# Utility Functions
def finger_angle(p1, p2, p3):
    """Return angle (in degrees) at p2 formed by p1–p2–p3"""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def angle_to_conf(angle, finger_name='default'):
    if finger_name == 'Thumb':
        return max(0, min(100, (angle - 40) / (160 - 40) * 100))
    else:
        return max(0, min(100, (angle - 60) / (180 - 60) * 100))
    
def image_to_base64(image_array):
    """Convert numpy image array to base64 string"""
    success, encoded_image = cv2.imencode('.jpg', image_array)
    if success:
        return base64.b64encode(encoded_image).decode('utf-8')
    return None

# FastAPI Endpoints
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Simple mudra prediction from image"""
    image = Image.open(file.file).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    top3_indices = np.argsort(preds[0])[-3:][::-1]

    results = [{"class": class_names[i], "probability": float(preds[0][i])} for i in top3_indices]
    return {"predictions": results}

@app.get("/mudra_info", response_model=MudraDetails)
async def get_mudra_details(mudra_name: str = Query(..., title="Mudra Name")):
    """
    Queries the Gemini API for descriptive details about a given Mudra name.
    The response is forced into the MudraDetails JSON structure.
    """
    if client is None:
        return {"error": "Gemini client not initialized. Check API key setup."}

    # 1. Construct the detailed prompt
    prompt = f"""
    You are an expert in classical Indian dance (Bharatanatyam/Kathak). 
    Provide the cultural meaning, associated inner thought/emotion, and a list of 3 common beginner mistakes for the hand gesture (Mudra) called **{mudra_name}**.
    Your output MUST strictly follow the requested JSON schema.
    """

    # 2. Define the response configuration to force JSON output
    # We use the schema generated from the Pydantic model MudraDetails
    response_schema = MudraDetails.model_json_schema()

    config = {
        "response_mime_type": "application/json",
        "response_schema": response_schema
    }

    try:
        # 3. Call the Gemini API with structured output configuration
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
            config=config
        )
        
        # 4. Parse the JSON response text
        # The result.text is guaranteed to be a JSON string matching the schema
        gemini_json_str = response.text.strip()
        details_data = json.loads(gemini_json_str)

        # 5. Return the validated Pydantic model
        return MudraDetails(**details_data)

    except Exception as e:
        print(f"Error calling Gemini API for {mudra_name}: {e}", file=sys.stderr)
        # Return a fallback response
        return MudraDetails(
            meaning="Details unavailable due to an external service error.",
            innerThought=f"Could not retrieve details for {mudra_name}.",
            commonMistakes=["Check server logs for Gemini API error."]
        )

@app.post("/hand_analysis")
async def hand_analysis(file: UploadFile = File(...), include_annotated_image: bool = False):
    """
    Comprehensive hand analysis including:
    - Number of hands detected
    - Finger confidence scores
    - Mudra predictions
    - Optional annotated image with landmarks
    """
    try:
        # Read and convert image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Initialize MediaPipe hands
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.6
        ) as hands:
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            h, w, _ = frame.shape

            # Initialize response data
            num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            all_finger_confidence = {}
            all_mudra_predictions = []
            annotated_frame = frame.copy()

            finger_joints = {
                "Thumb": [1, 2, 3, 4],
                "Index": [5, 6, 7, 8],
                "Middle": [9, 10, 11, 12],
                "Ring": [13, 14, 15, 16],
                "Little": [17, 18, 19, 20],
            }
            if results.multi_hand_landmarks:
                for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw landmarks on annotated image
                    mp_drawing.draw_landmarks(
                        annotated_frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS
                    )

                    # Convert landmarks to pixel coordinates
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                    # Calculate finger confidence
                    finger_scores = {}
                    for name, idx in finger_joints.items():
                        try:
                            ang = finger_angle(landmarks[idx[0]], landmarks[idx[1]], landmarks[idx[2]])
                            finger_scores[name] = round(angle_to_conf(ang, name), 2)
                        except Exception:
                            finger_scores[name] = 0.0

                    all_finger_confidence[f"hand_{hand_no + 1}"] = finger_scores

                    # Mudra prediction for this hand
                    x_coords = [p[0] for p in landmarks]
                    y_coords = [p[1] for p in landmarks]
                    x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                    
                    # Add padding
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)
                    
                    hand_crop = frame[y1:y2, x1:x2]
                    if hand_crop.size > 0:
                        hand_resized = cv2.resize(hand_crop, IMG_SIZE)
                        hand_array = np.expand_dims(hand_resized, axis=0).astype(np.float32) / 255.0
                        pred = model.predict(hand_array, verbose=0)
                        top_indices = np.argsort(pred[0])[-3:][::-1]
                        
                        hand_predictions = []
                        for idx in top_indices:
                            hand_predictions.append({
                                "class": class_names[idx],
                                "probability": float(pred[0][idx]),
                                "confidence": float(pred[0][idx]) * 100
                            })
                        
                        all_mudra_predictions.append({"hand_number": hand_no + 1,"predictions": hand_predictions})

                        # Add label to annotated image
                        if hand_predictions:
                            top_pred = hand_predictions[0]
                            label = f"Hand {hand_no+1}: {top_pred['class']} ({top_pred['confidence']:.1f}%)"
                            cv2.putText(annotated_frame, label, (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Prepare response
            response_data = {
                "num_hands": num_hands,
                "finger_confidence": all_finger_confidence,
                "mudra_predictions": all_mudra_predictions
            }
            # Include annotated image if requested
            if include_annotated_image:
                response_data["annotated_image"] = image_to_base64(annotated_frame)

            return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in hand analysis: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Mudra Recognition API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
