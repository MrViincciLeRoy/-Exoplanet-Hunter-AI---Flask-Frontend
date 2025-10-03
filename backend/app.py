"""
Exoplanet Hunter AI - FastAPI Backend
High-performance API for advanced exoplanet classification
Integrated with NASA Exoplanet Archive for confirmed planet data
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
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
import requests
from datetime import datetime, timedelta
from functools import lru_cache

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
current_module = sys.modules[__name__]
module_names = ['advanced_model', 'app', 'main', __name__]
for module_name in module_names:
    if module_name not in sys.modules or sys.modules[module_name] is None:
        sys.modules[module_name] = current_module

current_module.TSFreshFeatureExtractor = TSFreshFeatureExtractor
current_module.CNNTransformerExoplanetDetector = CNNTransformerExoplanetDetector
current_module.AdvancedExoplanetDetectionSystem = AdvancedExoplanetDetectionSystem
current_module.ConstitutionalExoplanetClassifier = ConstitutionalExoplanetClassifier

# ============================================================================
# NASA EXOPLANET ARCHIVE INTEGRATION
# ============================================================================

class NASAExoplanetArchive:
    """Interface to NASA Exoplanet Archive TAP/API service"""
    
    # TAP service endpoint (preferred)
    TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    # Legacy API endpoint (backup)
    API_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
    
    def __init__(self):
        self._cache = {}
        self._cache_time = {}
        self._cache_duration = timedelta(hours=24)  # Cache for 24 hours
        
    @lru_cache(maxsize=1000)
    def search_by_name(self, planet_name: str) -> Optional[Dict]:
        """Search for a confirmed planet by name"""
        try:
            # Use TAP service for confirmed planets
            query = f"""
            SELECT TOP 1 
                pl_name, hostname, sy_dist, pl_orbper, pl_rade, pl_bmasse,
                pl_eqt, pl_insol, pl_dens, disc_year, discoverymethod,
                disc_facility, disc_telescope, pl_orbsmax, pl_orbeccen,
                st_teff, st_rad, st_mass, st_age, sy_snum, sy_pnum
            FROM ps
            WHERE LOWER(pl_name) = LOWER('{planet_name}')
            """
            
            params = {
                'query': query,
                'format': 'json'
            }
            
            response = requests.get(self.TAP_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data and len(data) > 0:
                return self._format_planet_data(data[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Error searching NASA archive: {e}")
            return None
    
    def search_similar_planets(self, period: float, radius: float, 
                              temp: float, limit: int = 5) -> List[Dict]:
        """Find similar confirmed planets based on characteristics"""
        try:
            # Search for planets with similar orbital period and radius
            period_range = period * 0.2  # ±20%
            radius_range = radius * 0.3 if radius else 2.0  # ±30% or default
            
            query = f"""
            SELECT TOP {limit}
                pl_name, hostname, pl_orbper, pl_rade, pl_eqt,
                disc_year, discoverymethod, sy_dist
            FROM ps
            WHERE pl_orbper BETWEEN {period - period_range} AND {period + period_range}
            AND pl_rade IS NOT NULL
            ORDER BY ABS(pl_orbper - {period}) + ABS(pl_rade - {radius if radius else 1.0})
            """
            
            params = {
                'query': query,
                'format': 'json'
            }
            
            response = requests.get(self.TAP_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return [self._format_planet_data(p) for p in data]
            
        except Exception as e:
            logger.error(f"Error searching similar planets: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get current exoplanet archive statistics"""
        try:
            # Check cache first
            cache_key = 'statistics'
            if cache_key in self._cache:
                if datetime.now() - self._cache_time[cache_key] < self._cache_duration:
                    return self._cache[cache_key]
            
            query = """
            SELECT 
                COUNT(*) as total_planets,
                COUNT(DISTINCT hostname) as total_systems,
                MIN(disc_year) as first_discovery,
                MAX(disc_year) as latest_discovery
            FROM ps
            """
            
            params = {
                'query': query,
                'format': 'json'
            }
            
            response = requests.get(self.TAP_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            stats = data[0] if data else {}
            
            # Cache the result
            self._cache[cache_key] = stats
            self._cache_time[cache_key] = datetime.now()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def _format_planet_data(self, raw_data: Dict) -> Dict:
        """Format raw NASA archive data into a clean structure"""
        return {
            'name': raw_data.get('pl_name'),
            'host_star': raw_data.get('hostname'),
            'distance_pc': raw_data.get('sy_dist'),
            'orbital_period_days': raw_data.get('pl_orbper'),
            'radius_earth': raw_data.get('pl_rade'),
            'mass_earth': raw_data.get('pl_bmasse'),
            'equilibrium_temp_k': raw_data.get('pl_eqt'),
            'insolation_flux': raw_data.get('pl_insol'),
            'density_g_cm3': raw_data.get('pl_dens'),
            'discovery_year': raw_data.get('disc_year'),
            'discovery_method': raw_data.get('discoverymethod'),
            'discovery_facility': raw_data.get('disc_facility'),
            'discovery_telescope': raw_data.get('disc_telescope'),
            'semi_major_axis_au': raw_data.get('pl_orbsmax'),
            'eccentricity': raw_data.get('pl_orbeccen'),
            'stellar': {
                'temperature_k': raw_data.get('st_teff'),
                'radius_solar': raw_data.get('st_rad'),
                'mass_solar': raw_data.get('st_mass'),
                'age_gyr': raw_data.get('st_age')
            },
            'system': {
                'num_stars': raw_data.get('sy_snum'),
                'num_planets': raw_data.get('sy_pnum')
            }
        }

# ============================================================================
# INITIALIZE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Exoplanet Hunter AI API",
    description="Advanced ML API for exoplanet detection with NASA Exoplanet Archive integration",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NASA archive client
nasa_archive = NASAExoplanetArchive()

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
    
    # Optional: KOI identifier for archive lookup
    kepler_name: Optional[str] = Field(None, description="Kepler planet name (e.g., 'Kepler-22 b')")
    
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
                "koi_period": 289.86,
                "koi_duration": 7.38,
                "koi_depth": 489.0,
                "koi_srad": 0.979,
                "koi_steff": 5518,
                "koi_impact": 0.283,
                "koi_prad": 2.38,
                "koi_smass": 0.97,
                "koi_slogg": 4.467,
                "koi_insol": 1.11,
                "koi_teq": 262,
                "koi_model_snr": 40.8,
                "kepler_name": "Kepler-22 b"
            }
        }

class PredictionRequest(BaseModel):
    """Request body for predictions"""
    features: ExoplanetFeatures
    include_archive_data: bool = Field(True, description="Include NASA archive data if available")
    include_similar: bool = Field(False, description="Include similar confirmed planets")

class ArchiveData(BaseModel):
    """NASA Exoplanet Archive data"""
    name: Optional[str]
    host_star: Optional[str]
    distance_pc: Optional[float]
    orbital_period_days: Optional[float]
    radius_earth: Optional[float]
    mass_earth: Optional[float]
    equilibrium_temp_k: Optional[float]
    discovery_year: Optional[int]
    discovery_method: Optional[str]
    similar_planets: Optional[List[Dict[str, Any]]]

class PredictionResponse(BaseModel):
    """Response with prediction results"""
    prediction: str = Field(..., description="Predicted class")
    confidence: float = Field(..., description="Prediction confidence", ge=0, le=1)
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    uncertainty: float = Field(..., description="Prediction uncertainty", ge=0, le=1)
    flags: List[str] = Field(default=[], description="Constitutional AI flags")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    archive_data: Optional[ArchiveData] = Field(None, description="NASA Exoplanet Archive data")
    planet_classification: Optional[str] = Field(None, description="Planet type classification")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    models_loaded: bool
    feature_count: int
    model_type: str
    nasa_archive_available: bool
    archive_stats: Optional[Dict]

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
            
            # Load label encoder first
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
                
                # Load CNN-Transformer
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
                
                # Load constitutional wrapper
                const_path = advanced_dir / "constitutional.pkl"
                if const_path.exists():
                    try:
                        self.constitutional_wrapper = joblib.load(const_path)
                        logger.info("Constitutional AI config loaded")
                    except Exception as e:
                        logger.warning(f"Could not load constitutional wrapper: {e}")
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
            
            # Ensure constitutional wrapper exists
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
        if self.feature_names:
            feature_dict = features.dict()
            feature_values = []
            
            for fname in self.feature_names:
                value = feature_dict.get(fname)
                feature_values.append(value if value is not None else np.nan)
            
            X = np.array(feature_values).reshape(1, -1)
        else:
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
        
        # Apply imputer and scaler
        if self.imputer is not None:
            X = self.imputer.transform(X)
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def classify_planet_type(self, radius: Optional[float], 
                            temp: Optional[float]) -> str:
        """Classify planet type based on radius and temperature"""
        if radius is None:
            return "Unknown"
        
        if radius < 1.25:
            planet_type = "Earth-like"
        elif radius < 2.0:
            planet_type = "Super-Earth"
        elif radius < 6.0:
            planet_type = "Neptune-like"
        else:
            planet_type = "Jupiter-like"
        
        # Add temperature classification if available
        if temp:
            if temp < 200:
                planet_type += " (Cold)"
            elif temp < 400:
                planet_type += " (Temperate)"
            elif temp < 1000:
                planet_type += " (Hot)"
            else:
                planet_type += " (Ultra-hot)"
        
        return planet_type
    
    def predict(self, features: ExoplanetFeatures, 
               include_archive: bool = True,
               include_similar: bool = False) -> Dict:
        """Make prediction with constitutional AI checks and archive data"""
        if self.ensemble_model is None:
            raise RuntimeError("Model not loaded")
        
        # Prepare features
        X = self.prepare_features(features)
        
        # Make prediction
        if self.use_advanced and self.cnn_transformer_model:
            cnn_proba = self.cnn_transformer_model.predict(X, verbose=0)[0]
            ensemble_proba = self.ensemble_model.predict_proba(X)[0]
            probabilities = 0.5 * cnn_proba + 0.5 * ensemble_proba
            prediction_idx = np.argmax(probabilities)
            uncertainty = 1.0 - float(np.max(probabilities))
        else:
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
        
        # Constitutional AI checks
        flags = []
        explanation = []
        
        uncertainty_threshold = getattr(self.constitutional_wrapper, 'uncertainty_threshold', 0.3)
        confirmation_threshold = getattr(self.constitutional_wrapper, 'confirmation_threshold', 0.85)
        
        if uncertainty > uncertainty_threshold:
            flags.append("HIGH_UNCERTAINTY")
            explanation.append(f"High uncertainty ({uncertainty:.3f}). Recommend human review.")
        
        sorted_probs = sorted(probabilities, reverse=True)
        if len(sorted_probs) > 1 and sorted_probs[0] - sorted_probs[1] < 0.15:
            flags.append("AMBIGUOUS_PREDICTION")
            explanation.append("Multiple classes have similar probabilities.")
        
        if predicted_class == "CONFIRMED" and confidence < confirmation_threshold:
            flags.append("LOW_CONFIDENCE_CONFIRMED")
            explanation.append(f"CONFIRMED classification requires ≥{confirmation_threshold*100:.0f}% confidence.")
        
        if "CANDIDATE" in self.class_names:
            candidate_idx = self.class_names.index("CANDIDATE")
            if probabilities[candidate_idx] > 0.3 and prediction_idx != candidate_idx:
                flags.append("SIGNIFICANT_CANDIDATE_SIGNAL")
                explanation.append(f"Notable CANDIDATE probability ({probabilities[candidate_idx]:.3f}).")
        
        if features.koi_prad and features.koi_prad > 20:
            flags.append("UNUSUAL_PLANET_SIZE")
            explanation.append("Planet radius exceeds typical values.")
        
        if features.koi_period and features.koi_period < 0.5:
            flags.append("ULTRA_SHORT_PERIOD")
            explanation.append("Ultra-short orbital period detected.")
        
        # Build response
        result = {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "uncertainty": uncertainty,
            "flags": flags,
            "explanation": " ".join(explanation) if explanation else None,
            "planet_classification": self.classify_planet_type(
                features.koi_prad, 
                features.koi_teq
            )
        }
        
        # Add NASA archive data if requested and prediction is CONFIRMED
        if include_archive and predicted_class == "CONFIRMED":
            archive_data = None
            similar_planets = []
            
            # Try to find exact match by name
            if features.kepler_name:
                archive_data = nasa_archive.search_by_name(features.kepler_name)
                if archive_data:
                    flags.append("CONFIRMED_IN_NASA_ARCHIVE")
                    explanation.append(f"Found in NASA archive as {archive_data['name']}.")
            
            # Find similar planets if requested
            if include_similar and features.koi_period and features.koi_prad:
                similar_planets = nasa_archive.search_similar_planets(
                    features.koi_period,
                    features.koi_prad,
                    features.koi_teq or 0,
                    limit=5
                )
            
            if archive_data or similar_planets:
                result["archive_data"] = {
                    **(archive_data or {}),
                    "similar_planets": similar_planets if include_similar else None
                }
        
        return result

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
    
    # Test NASA archive connection
    try:
        stats = nasa_archive.get_statistics()
        if stats:
            logger.info(f"✅ NASA Exoplanet Archive connected: {stats.get('total_planets', 'N/A')} planets")
    except Exception as e:
        logger.warning(f"⚠️  NASA archive connection issue: {e}")

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
        "version": "2.1.0",
        "docs": "/docs",
        "health": "/health",
        "features": ["ML Classification", "NASA Archive Integration", "Constitutional AI"]
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint with archive status"""
    # Test NASA archive
    archive_available = False
    archive_stats = None
    try:
        archive_stats = nasa_archive.get_statistics()
        archive_available = bool(archive_stats)
    except:
        pass
    
    return {
        "status": "healthy",
        "version": "2.1.0",
        "models_loaded": model_manager.ensemble_model is not None,
        "feature_count": len(model_manager.feature_names) if model_manager.feature_names else 12,
        "model_type": "Advanced CNN-Transformer + Ensemble" if model_manager.use_advanced else "Standard Ensemble",
        "nasa_archive_available": archive_available,
        "archive_stats": archive_stats
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict exoplanet classification with NASA archive integration
    
    - **features**: Exoplanet features for classification
    - **include_archive_data**: Include NASA archive data for confirmed planets
    - **include_similar**: Include similar confirmed planets from archive
    
    Returns prediction with confidence scores, constitutional AI checks, and archive data.
    """
    try:
        result = model_manager.predict(
            request.features,
            include_archive=request.include_archive_data,
            include_similar=request.include_similar
        )
        
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

@app.get("/archive/search/{planet_name}", tags=["NASA Archive"])
async def search_planet(planet_name: str):
    """
    Search for a confirmed planet in NASA Exoplanet Archive
    
    - **planet_name**: Name of the planet (e.g., 'Kepler-22 b', 'HD 209458 b')
    
    Returns detailed information from the NASA Exoplanet Archive.
    """
    try:
        planet_data = nasa_archive.search_by_name(planet_name)
        
        if planet_data:
            return {
                "found": True,
                "data": planet_data,
                "source": "NASA Exoplanet Archive"
            }
        else:
            return {
                "found": False,
                "message": f"Planet '{planet_name}' not found in NASA Exoplanet Archive",
                "suggestion": "Check spelling or try alternate names"
            }
    
    except Exception as e:
        logger.error(f"Archive search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Archive search failed: {str(e)}"
        )

@app.get("/archive/similar", tags=["NASA Archive"])
async def find_similar_planets(
    period: float = Field(..., description="Orbital period in days", gt=0),
    radius: float = Field(..., description="Planet radius in Earth radii", gt=0),
    limit: int = Field(5, description="Maximum number of results", ge=1, le=20)
):
    """
    Find confirmed planets with similar characteristics
    
    - **period**: Orbital period in days
    - **radius**: Planet radius in Earth radii
    - **limit**: Maximum number of similar planets to return
    
    Returns list of similar confirmed planets from NASA Exoplanet Archive.
    """
    try:
        similar = nasa_archive.search_similar_planets(period, radius, 0, limit)
        
        return {
            "query": {
                "period_days": period,
                "radius_earth": radius
            },
            "count": len(similar),
            "planets": similar,
            "source": "NASA Exoplanet Archive"
        }
    
    except Exception as e:
        logger.error(f"Similar planets search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similar planets search failed: {str(e)}"
        )

@app.get("/archive/statistics", tags=["NASA Archive"])
async def archive_statistics():
    """
    Get current statistics from NASA Exoplanet Archive
    
    Returns total counts and discovery information.
    """
    try:
        stats = nasa_archive.get_statistics()
        
        if stats:
            return {
                "statistics": stats,
                "source": "NASA Exoplanet Archive",
                "last_updated": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to retrieve archive statistics"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Statistics retrieval failed: {str(e)}"
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
        "constitutional_ai_enabled": model_manager.constitutional_wrapper is not None,
        "nasa_archive_integrated": True
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
            "koi_model_snr",
            "kepler_name"
        ],
        "all_features": model_manager.feature_names if model_manager.feature_names else [],
        "nasa_archive_integration": {
            "enabled": True,
            "features": ["Search by name", "Find similar planets", "Get archive statistics"]
        }
    }

@app.post("/batch-predict", tags=["Prediction"])
async def batch_predict(
    requests: List[PredictionRequest],
    max_batch_size: int = Field(100, description="Maximum batch size", ge=1, le=1000)
):
    """
    Batch prediction endpoint for multiple exoplanets
    
    - **requests**: List of prediction requests
    - **max_batch_size**: Maximum number of predictions per batch
    
    Returns predictions for all input features.
    """
    if len(requests) > max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(requests)} exceeds maximum {max_batch_size}"
        )
    
    try:
        results = []
        for req in requests:
            try:
                result = model_manager.predict(
                    req.features,
                    include_archive=req.include_archive_data,
                    include_similar=req.include_similar
                )
                results.append({
                    "success": True,
                    "prediction": result
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "total": len(requests),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/planet-types", tags=["Reference"])
async def planet_type_info():
    """
    Get information about planet type classifications
    
    Returns definitions and characteristics of different planet types.
    """
    return {
        "classifications": {
            "Earth-like": {
                "radius_range": "< 1.25 Earth radii",
                "description": "Rocky planets similar in size to Earth",
                "examples": ["Earth", "Kepler-186 f"]
            },
            "Super-Earth": {
                "radius_range": "1.25 - 2.0 Earth radii",
                "description": "Rocky planets larger than Earth but smaller than Neptune",
                "examples": ["Kepler-22 b", "Proxima Centauri b"]
            },
            "Neptune-like": {
                "radius_range": "2.0 - 6.0 Earth radii",
                "description": "Gas planets similar in size to Neptune",
                "examples": ["Neptune", "Kepler-11 f"]
            },
            "Jupiter-like": {
                "radius_range": "> 6.0 Earth radii",
                "description": "Large gas giants similar to Jupiter",
                "examples": ["Jupiter", "HD 209458 b"]
            }
        },
        "temperature_zones": {
            "Cold": "< 200 K",
            "Temperate": "200 - 400 K (potentially habitable zone)",
            "Hot": "400 - 1000 K",
            "Ultra-hot": "> 1000 K"
        }
    }

@app.get("/examples", tags=["Reference"])
async def get_examples():
    """
    Get example prediction requests for testing
    
    Returns sample data for different types of exoplanets.
    """
    return {
        "examples": [
            {
                "name": "Kepler-22 b (Confirmed Super-Earth)",
                "features": {
                    "koi_period": 289.86,
                    "koi_duration": 7.38,
                    "koi_depth": 489.0,
                    "koi_srad": 0.979,
                    "koi_steff": 5518,
                    "koi_impact": 0.283,
                    "koi_prad": 2.38,
                    "koi_smass": 0.97,
                    "koi_slogg": 4.467,
                    "koi_insol": 1.11,
                    "koi_teq": 262,
                    "koi_model_snr": 40.8,
                    "kepler_name": "Kepler-22 b"
                }
            },
            {
                "name": "Hot Jupiter Example",
                "features": {
                    "koi_period": 3.5,
                    "koi_duration": 2.8,
                    "koi_depth": 1500.0,
                    "koi_srad": 1.2,
                    "koi_steff": 6200,
                    "koi_prad": 12.0,
                    "koi_teq": 1400
                }
            },
            {
                "name": "Earth-like Candidate",
                "features": {
                    "koi_period": 365.0,
                    "koi_duration": 6.5,
                    "koi_depth": 84.0,
                    "koi_srad": 1.0,
                    "koi_steff": 5778,
                    "koi_prad": 1.0,
                    "koi_teq": 288
                }
            },
            {
                "name": "False Positive Example",
                "features": {
                    "koi_period": 15.0,
                    "koi_duration": 0.5,
                    "koi_depth": 50.0,
                    "koi_srad": 2.5,
                    "koi_steff": 4500,
                    "koi_model_snr": 5.0
                }
            }
        ],
        "usage": "Copy the features object and use it in POST /predict endpoint"
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
