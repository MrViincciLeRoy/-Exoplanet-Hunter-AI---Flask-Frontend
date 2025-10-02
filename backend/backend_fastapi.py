"""
🪐 Exoplanet Hunter AI - FastAPI Backend
High-performance API for exoplanet classification
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import pickle
import numpy as np
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Exoplanet Hunter AI API",
    description="Advanced machine learning API for exoplanet detection and classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELS AND SCHEMAS
# ============================================================================

class ExoplanetFeatures(BaseModel):
    """Input features for exoplanet classification"""
    # Primary features (required)
    koi_period: float = Field(..., description="Orbital period in days", gt=0)
    koi_duration: float = Field(..., description="Transit duration in hours", gt=0)
    koi_depth: float = Field(..., description="Transit depth in ppm", gt=0)
    koi_srad: float = Field(..., description="Stellar radius in solar radii", gt=0)
    koi_steff: float = Field(..., description="Stellar effective temperature in K", gt=0)
    
    # Advanced features (optional)
    koi_impact: Optional[float] = Field(None, description="Impact parameter", ge=0, le=1)
    koi_prad: Optional[float] = Field(None, description="Planet radius in Earth radii", gt=0)
    koi_smass: Optional[float] = Field(None, description="Stellar mass in solar masses", gt=0)
    koi_slogg: Optional[float] = Field(None, description="Stellar surface gravity (log g)", gt=0)
    koi_insol: Optional[float] = Field(None, description="Insolation flux in Earth flux", gt=0)
    koi_teq: Optional[float] = Field(None, description="Equilibrium temperature in K", gt=0)
    koi_model_snr: Optional[float] = Field(None, description="Model signal-to-noise ratio", gt=0)
    
    @validator('koi_steff')
    def validate_temperature(cls, v):
        if v < 2000 or v > 50000:
            raise ValueError('Stellar temperature must be between 2000K and 50000K')
        return v
    
    @validator('koi_period')
    def validate_period(cls, v):
        if v > 10000:
            raise ValueError('Orbital period seems unrealistic (>10000 days)')
        return v

    class Config:
        schema_extra = {
            "example": {
                "koi_period": 5.7,
                "koi_duration": 3.2,
                "koi_depth": 150.5,
                "koi_srad": 1.05,
                "koi_steff": 5750,
                "koi_impact": 0.4,
                "koi_prad": 1.8,
                "koi_smass": 1.02,
                "koi_slogg": 4.5,
                "koi_insol": 1.2,
                "koi_teq": 290,
                "koi_model_snr": 25.5
            }
        }

class PredictionRequest(BaseModel):
    """Request body for batch predictions"""
    features: ExoplanetFeatures

class PredictionResponse(BaseModel):
    """Response with prediction results"""
    prediction: str = Field(..., description="Predicted class")
    confidence: float = Field(..., description="Prediction confidence", ge=0, le=1)
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    uncertainty: float = Field(..., description="Prediction uncertainty", ge=0, le=1)
    flags: List[str] = Field(default=[], description="Constitutional AI flags")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: bool
    feature_count: int

# ============================================================================
# MODEL LOADING
# ============================================================================

class ModelManager:
    """Manages loading and inference of ML models"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        self.feature_config = None
        self.class_names = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
        
    def load_models(self):
        """Load all required models and preprocessors"""
        try:
            models_dir = Path(__file__).parent / "models"
            
            # Load main model
            model_path = models_dir / "final_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("✓ Main model loaded successfully")
            else:
                logger.warning(f"Model not found at {model_path}")
            
            # Load scaler
            scaler_path = models_dir / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✓ Scaler loaded successfully")
            
            # Load imputer
            imputer_path = models_dir / "imputer.pkl"
            if imputer_path.exists():
                with open(imputer_path, 'rb') as f:
                    self.imputer = pickle.load(f)
                logger.info("✓ Imputer loaded successfully")
            
            # Load label encoder
            encoder_path = models_dir / "label_encoder.pkl"
            if encoder_path.exists():
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info("✓ Label encoder loaded successfully")
            
            # Load feature config
            config_path = Path(__file__).parent / "data" / "processed" / "feature_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.feature_config = json.load(f)
                logger.info("✓ Feature config loaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def prepare_features(self, features: ExoplanetFeatures) -> np.ndarray:
        """Prepare input features for model inference"""
        # Define feature order (must match training)
        feature_names = [
            'koi_period', 'koi_duration', 'koi_depth', 'koi_srad', 'koi_steff',
            'koi_impact', 'koi_prad', 'koi_smass', 'koi_slogg', 
            'koi_insol', 'koi_teq', 'koi_model_snr'
        ]
        
        # Extract features in correct order
        feature_dict = features.dict()
        feature_values = []
        
        for fname in feature_names:
            value = feature_dict.get(fname)
            # Use NaN for missing optional features
            feature_values.append(value if value is not None else np.nan)
        
        # Convert to numpy array
        X = np.array(feature_values).reshape(1, -1)
        
        # Apply imputer if available (handles missing values)
        if self.imputer is not None:
            X = self.imputer.transform(X)
        
        # Apply scaler if available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def predict(self, features: ExoplanetFeatures) -> Dict:
        """Make prediction with constitutional AI checks"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Prepare features
        X = self.prepare_features(features)
        
        # Get prediction and probabilities
        prediction_idx = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get class name
        if self.label_encoder is not None:
            predicted_class = self.label_encoder.inverse_transform([prediction_idx])[0]
        else:
            predicted_class = self.class_names[prediction_idx]
        
        # Calculate confidence and uncertainty
        confidence = float(np.max(probabilities))
        uncertainty = 1.0 - confidence
        
        # Create probability dictionary
        prob_dict = {
            self.class_names[i]: float(probabilities[i]) 
            for i in range(len(self.class_names))
        }
        
        # Constitutional AI checks
        flags = []
        explanation = None
        
        # Check for high uncertainty
        if uncertainty > 0.4:
            flags.append("HIGH_UNCERTAINTY")
            explanation = f"High uncertainty ({uncertainty:.3f}). Recommend human review."
        
        # Check for ambiguous predictions
        sorted_probs = sorted(probabilities, reverse=True)
        if len(sorted_probs) > 1 and sorted_probs[0] - sorted_probs[1] < 0.1:
            flags.append("AMBIGUOUS_PREDICTION")
            if explanation:
                explanation += " Multiple classes have similar probabilities."
            else:
                explanation = "Multiple classes have similar probabilities."
        
        # Check for unusual input values
        if features.koi_prad and features.koi_prad > 20:
            flags.append("UNUSUAL_PLANET_SIZE")
        
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "uncertainty": uncertainty,
            "flags": flags,
            "explanation": explanation
        }

# Initialize model manager
model_manager = ModelManager()

# ============================================================================
# STARTUP AND SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("🚀 Starting Exoplanet Hunter AI API...")
    success = model_manager.load_models()
    if success:
        logger.info("✓ All models loaded successfully")
    else:
        logger.warning("⚠ Some models failed to load")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down Exoplanet Hunter AI API...")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "🪐 Exoplanet Hunter AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": model_manager.model is not None,
        "feature_count": 12
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict exoplanet classification
    
    - **features**: Dictionary of exoplanet features
    
    Returns prediction with confidence scores and constitutional AI checks.
    """
    try:
        # Make prediction
        result = model_manager.predict(request.features)
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input values: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model-info", tags=["Model"])
async def model_info():
    """Get information about the loaded model"""
    if model_manager.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_type": type(model_manager.model).__name__,
        "classes": model_manager.class_names,
        "feature_count": 12,
        "scaler_loaded": model_manager.scaler is not None,
        "imputer_loaded": model_manager.imputer is not None
    }

@app.get("/features", tags=["Model"])
async def get_features():
    """Get expected input features"""
    return {
        "required_features": [
            "koi_period",
            "koi_duration", 
            "koi_depth",
            "koi_srad",
            "koi_steff"
        ],
        "optional_features": [
            "koi_impact",
            "koi_prad",
            "koi_smass",
            "koi_slogg",
            "koi_insol",
            "koi_teq",
            "koi_model_snr"
        ]
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "docs": "/docs"
    }

@app.exception_handler(500)
async def server_error_handler(request, exc):
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "contact": "Check logs for details"
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
