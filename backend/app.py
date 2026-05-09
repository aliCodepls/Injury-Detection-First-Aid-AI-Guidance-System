from flask import Flask
from flask_cors import CORS
from routes.analyze import analyze_bp
from routes.health import health_bp

app = Flask(__name__)
CORS(app)  # allows requests from your phone on the same Wi-Fi

app.register_blueprint(analyze_bp, url_prefix="/api")
app.register_blueprint(health_bp, url_prefix="/api")

if __name__ == "__main__":
    # host="0.0.0.0" makes it accessible from your phone, not just localhost
    app.run(debug=True, host="0.0.0.0", port=5000)
