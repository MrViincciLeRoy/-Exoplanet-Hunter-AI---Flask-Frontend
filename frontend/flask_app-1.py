"""
🪐 Exoplanet Hunter AI - Flask Frontend
Space-themed web interface for exoplanet classification
"""

from flask import Flask, render_template, request, jsonify
import requests
import os
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# API Configuration
API_BASE_URL = os.getenv('API_URL', 'http://localhost:8000')

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page with dashboard and model statistics"""
    try:
        # Fetch model stats from API
        response = requests.get(f'{API_BASE_URL}/health', timeout=5)
        api_status = 'online' if response.status_code == 200 else 'offline'
    except:
        api_status = 'offline'
    
    return render_template('index.html', api_status=api_status)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page with interactive form"""
    if request.method == 'POST':
        try:
            # Get form data
            data = request.get_json()
            
            # Make prediction via API
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
                }), 500
                
        except requests.exceptions.RequestException as e:
            return jsonify({
                'error': 'Cannot connect to API',
                'details': str(e)
            }), 503
        except Exception as e:
            return jsonify({
                'error': 'Server error',
                'details': str(e)
            }), 500
    
    return render_template('predict.html')

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/api/model-stats')
def model_stats():
    """Proxy endpoint to fetch model statistics"""
    try:
        response = requests.get(f'{API_BASE_URL}/health', timeout=5)
        return jsonify(response.json())
    except:
        return jsonify({
            'status': 'offline',
            'message': 'API unavailable'
        }), 503

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'frontend'
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(
        debug=os.getenv('FLASK_ENV') == 'development',
        host='0.0.0.0',
        port=port
    )
