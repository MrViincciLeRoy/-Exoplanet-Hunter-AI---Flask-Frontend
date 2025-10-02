"""
🪐 Exoplanet Hunter AI - Flask Frontend
Beautiful space-themed web interface
"""

from flask import Flask, render_template, request, jsonify
import requests
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))

# Backend API configuration
API_BASE_URL = os.getenv('API_URL', 'http://localhost:8000')

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page with dashboard"""
    try:
        response = requests.get(f'{API_BASE_URL}/health', timeout=5)
        api_status = 'online' if response.status_code == 200 else 'offline'
    except:
        api_status = 'offline'
        logger.warning("Backend API is not reachable")
    
    return render_template('index.html', api_status=api_status)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'POST':
        try:
            data = request.get_json()
            
            # Make prediction request to backend
            response = requests.post(
                f'{API_BASE_URL}/predict',
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return jsonify(response.json())
            else:
                return jsonify({
                    'error': 'Prediction failed',
                    'details': response.text
                }), response.status_code
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return jsonify({
                'error': 'Cannot connect to API',
                'details': str(e)
            }), 503
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return jsonify({
                'error': 'Server error',
                'details': str(e)
            }), 500
    
    return render_template('predict.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/model-stats')
def model_stats():
    """Proxy endpoint for model statistics"""
    try:
        response = requests.get(f'{API_BASE_URL}/model-info', timeout=5)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'status': 'unavailable'}), response.status_code
    except:
        return jsonify({
            'status': 'offline',
            'message': 'Backend API unavailable'
        }), 503

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'frontend',
        'backend_url': API_BASE_URL
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"Server error: {e}")
    return render_template('index.html'), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"🚀 Starting Flask frontend on port {port}")
    logger.info(f"🔗 Backend API: {API_BASE_URL}")
    
    app.run(
        debug=debug,
        host='0.0.0.0',
        port=port
    )
