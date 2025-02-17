from fastapi import FastAPI, UploadFile
from deepface import DeepFace
import mediapipe as mp
import cv2
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import models
from openai import OpenAI
import json
import base64
from enum import Enum
import numpy as np

app = FastAPI()

# Initialize Mediapipe's Face Mesh model for landmark detection
mp_face_mesh = mp.solutions.face_mesh

@app.post("/faceDetection/")
async def check_faces(file1: UploadFile, file2: UploadFile) -> models.FaceDetectionModel:
    # Read uploaded images into memory
    file1_content = await file1.read()
    file2_content = await file2.read()
    
    file1_image = bytes_to_image(file1_content)
    file2_image = bytes_to_image(file2_content)
        
    deepface_models = [
        "VGG-Face", 
        "Facenet512", 
        "GhostFaceNet"
    ]
    
    deepface_result_list = []
    
    # Run DeepFace verification to check if the two images are of the same person
    for model in deepface_models:
        result = DeepFace.verify(file1_image, file2_image, model_name=model)

        deepface_verification_model = models.DeepFaceVerificationModel(
            model=result["model"],
            deepface_verify= result["verified"],
            threshold=result["threshold"],
            cosine_distance= result["distance"],
            detector_backend=result["detector_backend"]            
        )
        
        deepface_result_list.append(deepface_verification_model)
    
    # Combine results into a structured response model
    output = models.FaceDetectionModel(
        deepface_verification_list=deepface_result_list
    )
    
    landmarks1, landmarks2 = get_landmarks(file1_image, file2_image)
    if landmarks1 != None or landmarks2 != None:
        output.face1_face_landmarks = landmarks1
        output.face2_face_landmarks = landmarks2

    # Return results as a JSON response
    json_compatible_item_data = jsonable_encoder(output)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/checkByLLM/")
async def verify_by_lmm(file1: UploadFile, file2: UploadFile, faceDetectionInput: str) -> models.LLMResultModel:
    file1_content = await file1.read()
    file2_content = await file2.read()
    
    file1_image = bytes_to_image(file1_content)
    file2_image = bytes_to_image(file2_content)
    
    face1_cropped = crop_face(file1_image)
    face2_cropped = crop_face(file2_image)
    
    base64_image_original = encode_image(face1_cropped)
    base64_image_second = encode_image(face2_cropped)
    
    prompt = f"""
    Context:
    You are an advanced AI specializing in face verification and deepfake detection.  
    Your task: Determine if two images belong to the same person or if the second image is a deepfake.  

    Input Data:
    - JSON input:
        - Results from multiple DeepFace models 
            - Best for verifying real images: VGG-Face and GhostFaceNet
	        - Best for detecting deepfakes: Facenet512 followed by GhostFaceNet
        - face landmarks from the two images, including 'image_shape' for normalization
    - Two images: for visual analysis

    Task Instructions:
    1. DeepFace Verification: Analyze the results of DeepFace models and dynamically weight their influence based on their performance as provided in the input data.
    2. Landmark Comparison: Use `image_shape` from each landmark set to normalize coordinates and compare distances between key points. Detect proportional deviations and asymmetry.  
    3. Deepfake & Visual Analysis: Examine the two uploaded images for inconsistencies and decide whether the second image is a real match or a deepfake.  
    4. Final Decision Making: Evaluate findings from all analyses.

    Output Format:
    {{
        "verified": true/false,
        "confidence_score": "confidence score between 0.0 and 1.0",
        "explanation": "Brief justification summarizing DeepFace verification, landmark analysis, and visual findings."
    }}
    
    Input JSON: "
    {faceDetectionInput}"
    """
    try:
        client = OpenAI(
            api_key="api-key"
        )
        
        chat_completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role":"user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{prompt}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image_original}", "detail": "low"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image_second}", "detail": "low"}
                        }
                    ]                
                }
            ],
        )
        
        # parse response to json
        response_content = chat_completion.choices[0].message.content
        
        # remove markdown format from json response
        if response_content.startswith("```json"):
            response_content = response_content[7:]
        if response_content.endswith("```"):
            response_content = response_content[:-3]

        # remove additional parts 
        response_content = response_content.strip()

        result = json.loads(response_content)

        # format response to model
        llm_result_model = models.LLMResultModel(
            verified=result.get("verified"),
            confidence_score=result.get("confidence_score"),
            explanation=result.get("explanation"),
            test_valid=True
        )
        
        return llm_result_model
    except Exception as e:
        return models.LLMResultModel(
            verified=False, 
            explanation=f"An error occurred during verification: {str(e)}",
            confidence_score=0,
            test_valid=False
        )
        
class DetectorBackend(Enum):
    OPENCV = "1"
    RETINAFACE = "2"

@app.post("/modelValidation/")
async def validate_models(file1: UploadFile, file2: UploadFile, detectorBackend: DetectorBackend) -> list[models.DeepFaceVerificationModel]:
    # Read uploaded images into memory
    file1_content = await file1.read()
    file2_content = await file2.read()

    # Save images as temporary files for processing
    with open("file1_temp.jpg", "wb") as f:
        f.write(file1_content)
    with open("file2_temp.jpg", "wb") as f:
        f.write(file2_content)
    
    detectorBackendStr = ""
    if detectorBackend == DetectorBackend.OPENCV:
        detectorBackendStr = "opencv"
    elif detectorBackend == DetectorBackend.RETINAFACE:
        detectorBackendStr = "retinaface"
        
    deepface_models = [
        "VGG-Face", 
        "Facenet", 
        "Facenet512", 
        "OpenFace", 
        #"DeepFace", not tested because older version of tensorflow needed
        "DeepID", 
        "ArcFace", 
        "Dlib", 
        "SFace",
        "GhostFaceNet"
    ]
    
    deepface_verification_list = []
    
    # Run DeepFace verifications to check which performs best
    for model in deepface_models:
        result = DeepFace.verify("file1_temp.jpg", "file2_temp.jpg", model_name=model, detector_backend=detectorBackendStr,)
        
        deepface_verification_model = models.DeepFaceVerificationModel(
            model=result["model"],
            deepface_verify= result["verified"],
            threshold=result["threshold"],
            cosine_distance= result["distance"],
            detector_backend=result["detector_backend"]            
        )
        
        deepface_verification_list.append(deepface_verification_model)

    return deepface_verification_list

#region landmarks

def extract_landmarks_from_face(image):
    # Extract facial landmarks from an image using Mediapipe's Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        # Convert the BGR image to RGB before processing.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
        results = face_mesh.process(image_rgb)
        
        FACEMESH_INDICES = {
            "left_eye": [33, 133],  # Outer points for left eye
            "right_eye": [362, 263],  # Outer points for right eye
            "mouth": [61, 291, 78, 308],  # Outer points of mouth
            "left_iris": [473],  # central point of left iris
            "right_iris": [468],  # central point of right iris
        }

        EXTRA_INDICES = {
            "nose_tip": [1],  # nose tip
            "chin": [199],  # chin
            "left_cheek": [50],  # left cheek
            "right_cheek": [280]  # right cheek
        }

        landmarks_data = {}

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for feature, indices in FACEMESH_INDICES.items():
                    landmarks_data[feature] = [
                    {
                        "x": face_landmarks.landmark[i].x, 
                        "y": face_landmarks.landmark[i].y
                    }
                    for i in indices
            ]

        for feature, indices in EXTRA_INDICES.items():
            landmarks_data[feature] = [
                {
                    "x": face_landmarks.landmark[i].x, 
                    "y": face_landmarks.landmark[i].y
                }
                for i in indices
            ]
            
        return models.FaceLandmarks.from_dict(landmarks_data)
                
def get_landmarks(image1, image2):  
    # crop faces to 128x128 format
    face1 = crop_face(image1)
    face2 = crop_face(image2)
    
    if face1 is None or face2 is None:
        return 0
    
    try:
        landmarks1 = extract_landmarks_from_face(face1)
        landmarks2 = extract_landmarks_from_face(face2)
        
        return landmarks1, landmarks2
    except Exception as e:
        return None, None

#endregion

#region helper

def bytes_to_image(image_bytes):
    image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def crop_face(image):    
    try:
        detections = DeepFace.extract_faces(image)
    except Exception as e:
        print("Error in face recognition", e)
        return []
    
    for face in enumerate(detections):
        x, y, w, h = face[1]['facial_area']['x'], face[1]['facial_area']['y'], face[1]['facial_area']['w'], face[1]['facial_area']['h']

        # get face area and crop face
        x, y = max(0, x), max(0, y)
        w, h = min(image.shape[1] - x, w), min(image.shape[0] - y, h)
        face_crop = image[y:y+h, x:x+w]

        # resize face to 128 x 128 pixels
        face_resized = cv2.resize(face_crop, (128,128))
        return face_resized        

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

#endregion