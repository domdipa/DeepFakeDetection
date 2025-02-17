from pydantic import BaseModel
from typing import Optional

class LandmarkPoint(BaseModel):
    x: float
    y: float

class FaceLandmarks(BaseModel):
    image_shape: str
    left_eye: list[LandmarkPoint]
    right_eye: list[LandmarkPoint]
    mouth: list[LandmarkPoint]
    left_iris: list[LandmarkPoint]
    right_iris: list[LandmarkPoint]
    nose_tip: list[LandmarkPoint]
    chin: list[LandmarkPoint]
    left_cheek: list[LandmarkPoint]
    right_cheek: list[LandmarkPoint]
    
    @classmethod
    def from_dict(cls, data):
        #convert dict to FaceLandmarks object
        return cls(
            image_shape="128x128",
            **{key: [LandmarkPoint(**point) for point in value] for key, value in data.items()}
        )
    
class DeepFaceVerificationModel(BaseModel):
    model: str
    deepface_verify: bool
    threshold:float
    cosine_distance: float
    detector_backend: str

class FaceDetectionModel(BaseModel):
    deepface_verification_list: list[DeepFaceVerificationModel]
    face1_face_landmarks: Optional[FaceLandmarks] = None
    face2_face_landmarks: Optional[FaceLandmarks] = None
    
class LLMResultModel(BaseModel):
    verified: bool
    confidence_score: float
    explanation: str
    test_valid: bool

class DeepFaceValidationTestModel(BaseModel):
    deepface_verification_test: list[DeepFaceVerificationModel]