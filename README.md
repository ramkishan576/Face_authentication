import os
import face_recognition
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from io import BytesIO
from typing import List, Dict
 
# FastAPI application instance
app = FastAPI()
 
# Define a constant for the API Key
API_KEY = "your_secret_api_key"  # Replace with your actual API Key
 
# Function to load images from a folder
def load_images_from_folder(folder_path: str) -> List[str]:
    """
    Given a folder path, this function loads all image file paths (.png, .jpg, .jpeg).
    """
    image_paths = []
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(image_path)
    return image_paths
 
# Function to generate embeddings for given image paths
def generate_face_embeddings(image_paths: List[str]) -> List[List[float]]:
    """
    This function generates embeddings for each image given its file path.
    """
    embeddings = []
    for image_path in image_paths:
        try:
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                embeddings.append(encoding[0])
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    return embeddings
 
# Function to precompute embeddings for all known images and save them in a pickle file
def precompute_known_embeddings(known_images_folder: str) -> Dict[str, List[List[float]]]:
    """
    This function computes embeddings for all known images inside a folder,
    where each subfolder represents a user (user ID).
    """
    known_embeddings = {}
    
    for user_folder in os.listdir(known_images_folder):
        user_folder_path = os.path.join(known_images_folder, user_folder)
        
        if os.path.isdir(user_folder_path):
            image_paths = load_images_from_folder(user_folder_path)
            embeddings = generate_face_embeddings(image_paths)
            if embeddings:
                known_embeddings[user_folder] = embeddings
 
    # Save embeddings to a pickle file to avoid recomputing on every request
    with open('known_embeddings.pkl', 'wb') as f:
        pickle.dump(known_embeddings, f)
    
    print("Known embeddings precomputed and saved.")
    return known_embeddings
 
# Function to generate the embedding for an unknown image (uploaded by the user)
def generate_unknown_embedding(image_data: bytes) -> List[float]:
    """
    This function generates the embedding for an unknown image uploaded by the user.
    """
    try:
        image = face_recognition.load_image_file(BytesIO(image_data))
        encoding = face_recognition.face_encodings(image)
        if encoding:
            return encoding[0]
        else:
            raise ValueError("No face found in the uploaded image.")
    except Exception as e:
        print(f"Error generating embedding for the image: {e}")
        return None
 
# Function to compare embeddings and identify the profile
def compare_embeddings(known_embeddings: Dict[str, List[List[float]]], unknown_embedding: List[float]) -> str:
    """
    This function compares the embeddings of the known users with the unknown embedding
    to identify the profile ID.
    """
    for user_id, embeddings in known_embeddings.items():
        for known_embedding in embeddings:
            if (known_embedding == unknown_embedding).all():  # Direct comparison
                return user_id
    return None  # If no match is found
 
# API Key Validation function
def check_api_key(x_api_key: str = Header(...)):
    """
    Validate API Key passed in the request header for security.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
 
@app.post("/identify_user/")
async def identify_user(file: UploadFile = File(...), x_api_key: str = Header(...)):
    """
    This endpoint receives an image, generates its embedding, and compares it with known embeddings.
    If a match is found, the user's profile ID is returned.
    """
    # API Key validation
    check_api_key(x_api_key)
 
    try:
        # Step 1: Load Known Embeddings from file if already computed, else compute new ones
        if os.path.exists('known_embeddings.pkl'):
            with open('known_embeddings.pkl', 'rb') as f:
                known_embeddings = pickle.load(f)
        else:
            known_embeddings = precompute_known_embeddings("/path/to/known_images")  # Update path
        
        # Step 2: Generate Unknown Image Embedding from uploaded image
        image_data = await file.read()
        unknown_embedding = generate_unknown_embedding(image_data)
 
        if unknown_embedding is None:
            raise HTTPException(status_code=400, detail="No face found in the uploaded image.")
        
        # Step 3: Compare Embeddings and Identify the User Profile
        profile_id = compare_embeddings(known_embeddings, unknown_embedding)
        
        if profile_id:
            return {"profile_id": profile_id}
        else:
            raise HTTPException(status_code=404, detail="No matching profile found.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
