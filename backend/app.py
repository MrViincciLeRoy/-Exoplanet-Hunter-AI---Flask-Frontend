"""
Exoplanet Hunter AI - FastAPI Backend
High-performance API for advanced exoplanet classification
Works standalone without advanced_model.py source code
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import pickle
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
import joblib
import warnings
import sys

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MINIMAL CLASS DEFINITIONS FOR PICKLE DESERIALIZATION
# ============================================================================

class TSFreshFeatureExtractor:
    """Stub class for unpickling - feature extraction not used in API"""
    def __init__(self, *args, **kwargs):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X, y=None):
        return X

class CNNTransformerExoplanetDetector:
    """Stub class for unpickling - actual Keras model loaded separately"""
    def __init__(self, *args, **kwargs):
        self.model = None
    
    def predict(self, X):
        if self.model:
            return self.model.predict(X)
        return None
    
    def predict_proba(self, X):
        return self.predict(X)

class AdvancedExoplanetDetectionSystem:
    """Stub class for unpickling - components loaded separately"""
    def __init__(self, *args, **kwargs):
        self.cnn_transformer = None
        self.ensemble = None
        self.constitutional = None
        self.feature_extractor = None
    
    def predict(self, X):
        pass
    
    def predict_proba(self, X):
        pass

class ConstitutionalExoplanetClassifier:
    """
    Minimal stub class to allow unpickling of constitutional wrapper.
    We only need to extract the threshold attributes, not run the full model.
    """
    def __init__(self, base_model=None, uncertainty_threshold=0.3, 
                 confirmation_threshold=0.85, rules=None):
        self.base_model = base_model
        self.uncertainty_threshold = uncertainty_threshold
        self.confirmation_threshold = confirmation_threshold
        self.rules = rules or []
    
    def predict(self, X):
        """Stub method - not used in API"""
        pass
    
    def predict_proba(self, X):
        """Stub method - not used in API"""
        pass

# Register classes for pickle compatibility
# This allows unpickling objects that reference 'advanced_model' module
# Handle different execution contexts (direct run, uvicorn, gunicorn, etc.)
current_module = sys.modules[__name__]

# Register under all possible module names
module_names = ['advanced_model', 'app', 'main', __name__]
for module_name in module_names:
    if module_name not in sys.modules or sys.modules[module_name] is None:
        sys.modules[module_name] = current_module

# Explicitly add to current module's namespace to ensure pickle can find them
current_module.TSFreshFeatureExtractor = TSFreshFeatureExtractor
current_module.CNNTransformerExoplanetDetector = CNNTransformerExoplanetDetector
current_module.AdvancedExoplanetDetectionSystem = AdvancedExoplanetDetectionSystem
current_module.ConstitutionalExoplanetClassifier = ConstitutionalExoplanetClassifier

# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Exoplanet Hunter AI API",
    description="Advanced machine learning API for exoplanet detection and classification",
    version="2.0.0",
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
    """Request body for predictions"""
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
    model_type: str

# ============================================================================
# MODEL LOADING
# ============================================================================

class ModelManager:
    """Manages loading and inference of advanced ML models"""
    
    def __init__(self):
        self.cnn_transformer_model = None
        self.ensemble_model = None
        self.constitutional_wrapper = None
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        self.feature_names = None
        self.class_names = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
        self.use_advanced = False
        
    def load_models(self):
        """Load all required models and preprocessors"""
        try:
            models_dir = Path(__file__).parent / "models"
            advanced_dir = models_dir / "advanced_system"
            
            # Load label encoder first (critical for class mapping)
            encoder_path = models_dir / "label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                self.class_names = self.label_encoder.classes_.tolist()
                logger.info(f"Label encoder loaded: {self.class_names}")
            
            # Try to load advanced system
            if advanced_dir.exists():
                logger.info("Attempting to load advanced model system...")
                
                # Load feature names
                feature_path = advanced_dir / "feature_names.pkl"
                if feature_path.exists():
                    self.feature_names = joblib.load(feature_path)
                    logger.info(f"Feature names loaded ({len(self.feature_names)} features)")
                else:
                    # Fallback to training data
                    train_path = Path(__file__).parent / "data" / "processed" / "X_train.csv"
                    if train_path.exists():
                        X_train = pd.read_csv(train_path)
                        self.feature_names = X_train.columns.tolist()
                        logger.info(f"Features inferred from training data")
                
                # Load CNN-Transformer (just the Keras model, no custom classes needed)
                cnn_path = advanced_dir / "cnn_transformer.keras"
                if cnn_path.exists():
                    try:
                        self.cnn_transformer_model = keras.models.load_model(
                            str(cnn_path),
                            compile=False
                        )
                        logger.info("CNN-Transformer loaded")
                    except Exception as e:
                        logger.warning(f"Could not load CNN-Transformer: {e}")
                
                # Load ensemble
                ensemble_path = advanced_dir / "ensemble.pkl"
                if ensemble_path.exists():
                    self.ensemble_model = joblib.load(ensemble_path)
                    logger.info("Ensemble model loaded")
                
                # Load constitutional wrapper (just for thresholds)
                const_path = advanced_dir / "constitutional.pkl"
                if const_path.exists():
                    try:
                        self.constitutional_wrapper = joblib.load(const_path)
                        logger.info("Constitutional AI config loaded")
                    except Exception as e:
                        logger.warning(f"Could not load constitutional wrapper: {e}")
                        # Create default wrapper
                        self.constitutional_wrapper = ConstitutionalExoplanetClassifier()
                        logger.info("Using default constitutional AI config")
                
                # Check if advanced system fully loaded
                if self.cnn_transformer_model and self.ensemble_model:
                    self.use_advanced = True
                    logger.info("✅ Advanced system fully loaded!")
            
            # Fallback to original model
            if not self.use_advanced:
                logger.info("Loading fallback model...")
                model_path = models_dir / "final_model.pkl"
                if model_path.exists():
                    self.ensemble_model = joblib.load(model_path)
                    logger.info("Fallback model loaded")
                elif models_dir.exists():
                    # Try to find any .pkl model file
                    pkl_files = list(models_dir.glob("*.pkl"))
                    for pkl_file in pkl_files:
                        if "model" in pkl_file.name.lower():
                            try:
                                self.ensemble_model = joblib.load(pkl_file)
                                logger.info(f"Loaded model from {pkl_file.name}")
                                break
                            except:
                                continue
            
            # Load scaler
            scaler_path = models_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded")
            
            # Load imputer
            imputer_path = models_dir / "imputer.pkl"
            if imputer_path.exists():
                self.imputer = joblib.load(imputer_path)
                logger.info("Imputer loaded")
            
            # Ensure we have a constitutional wrapper
            if self.constitutional_wrapper is None:
                self.constitutional_wrapper = ConstitutionalExoplanetClassifier()
                logger.info("Created default constitutional AI config")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def prepare_features(self, features: ExoplanetFeatures) -> np.ndarray:
        """Prepare input features for model inference"""
        # If we have feature names from advanced model, use those
        if self.feature_names:
            feature_dict = features.dict()
            feature_values = []
            
            for fname in self.feature_names:
                # Try to extract the feature value
                value = feature_dict.get(fname)
                feature_values.append(value if value is not None else np.nan)
            
            X = np.array(feature_values).reshape(1, -1)
        else:
            # Fallback: use standard feature order
            feature_names = [
                'koi_period', 'koi_duration', 'koi_depth', 'koi_srad', 'koi_steff',
                'koi_impact', 'koi_prad', 'koi_smass', 'koi_slogg', 
                'koi_insol', 'koi_teq', 'koi_model_snr'
            ]
            
            feature_dict = features.dict()
            feature_values = []
            
            for fname in feature_names:
                value = feature_dict.get(fname)
                feature_values.append(value if value is not None else np.nan)
            
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
        if self.ensemble_model is None:
            raise RuntimeError("Model not loaded")
        
        # Prepare features
        X = self.prepare_features(features)
        
        # Make prediction based on model type
        if self.use_advanced and self.cnn_transformer_model:
            # Advanced system prediction
            cnn_proba = self.cnn_transformer_model.predict(X, verbose=0)[0]
            ensemble_proba = self.ensemble_model.predict_proba(X)[0]
            
            # Combine predictions (weighted average)
            probabilities = 0.5 * cnn_proba + 0.5 * ensemble_proba
            prediction_idx = np.argmax(probabilities)
            
            # Simple uncertainty estimation
            uncertainty = 1.0 - float(np.max(probabilities))
        else:
            # Standard model prediction
            probabilities = self.ensemble_model.predict_proba(X)[0]
            prediction_idx = np.argmax(probabilities)
            uncertainty = 1.0 - float(np.max(probabilities))
        
        # Get class name
        predicted_class = self.class_names[prediction_idx]
        confidence = float(np.max(probabilities))
        
        # Create probability dictionary
        prob_dict = {
            self.class_names[i]: float(probabilities[i]) 
            for i in range(len(self.class_names))
        }
        
        # Constitutional AI checks (reimplemented)
        flags = []
        explanation = []
        
        # Get thresholds from constitutional wrapper
        uncertainty_threshold = getattr(self.constitutional_wrapper, 'uncertainty_threshold', 0.3)
        confirmation_threshold = getattr(self.constitutional_wrapper, 'confirmation_threshold', 0.85)
        
        # Check for high uncertainty
        if uncertainty > uncertainty_threshold:
            flags.append("HIGH_UNCERTAINTY")
            explanation.append(f"High uncertainty ({uncertainty:.3f}). Recommend human review.")
        
        # Check for ambiguous predictions
        sorted_probs = sorted(probabilities, reverse=True)
        if len(sorted_probs) > 1 and sorted_probs[0] - sorted_probs[1] < 0.15:
            flags.append("AMBIGUOUS_PREDICTION")
            explanation.append("Multiple classes have similar probabilities.")
        
        # Check for CONFIRMED class with low confidence
        if predicted_class == "CONFIRMED" and confidence < confirmation_threshold:
            flags.append("LOW_CONFIDENCE_CONFIRMED")
            explanation.append(f"CONFIRMED classification requires ≥{confirmation_threshold*100:.0f}% confidence for scientific rigor.")
        
        # Check for significant CANDIDATE signal
        if "CANDIDATE" in self.class_names:
            candidate_idx = self.class_names.index("CANDIDATE")
            if probabilities[candidate_idx] > 0.3 and prediction_idx != candidate_idx:
                flags.append("SIGNIFICANT_CANDIDATE_SIGNAL")
                explanation.append(f"Notable CANDIDATE probability ({probabilities[candidate_idx]:.3f}).")
        
        # Check for unusual input values
        if features.koi_prad and features.koi_prad > 20:
            flags.append("UNUSUAL_PLANET_SIZE")
            explanation.append("Planet radius exceeds typical values.")
        
        if features.koi_period and features.koi_period < 0.5:
            flags.append("ULTRA_SHORT_PERIOD")
            explanation.append("Ultra-short orbital period detected.")
        
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "uncertainty": uncertainty,
            "flags": flags,
            "explanation": " ".join(explanation) if explanation else None
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
        logger.info("✅ All models loaded successfully")
    else:
        logger.warning("⚠️  Some models failed to load")

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
        "message": "Exoplanet Hunter AI API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "models_loaded": model_manager.ensemble_model is not None,
        "feature_count": len(model_manager.feature_names) if model_manager.feature_names else 12,
        "model_type": "Advanced CNN-Transformer + Ensemble" if model_manager.use_advanced else "Standard Ensemble"
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
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/model-info", tags=["Model"])
async def model_info():
    """Get information about the loaded model"""
    if model_manager.ensemble_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_type": "Advanced CNN-Transformer + Ensemble" if model_manager.use_advanced else "Standard Ensemble",
        "classes": model_manager.class_names,
        "feature_count": len(model_manager.feature_names) if model_manager.feature_names else 12,
        "scaler_loaded": model_manager.scaler is not None,
        "imputer_loaded": model_manager.imputer is not None,
        "cnn_transformer_loaded": model_manager.cnn_transformer_model is not None,
        "constitutional_ai_enabled": model_manager.constitutional_wrapper is not None
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
        ],
        "all_features": model_manager.feature_names if model_manager.feature_names else []
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return {
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "docs": "/docs"
    }

@app.exception_handler(500)
async def server_error_handler(request, exc):
    """Handle 500 errors"""
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
