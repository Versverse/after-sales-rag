import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

from .api_integration import register_vector_routes

from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    origin = os.getenv("RAG_FRONTEND_ORIGIN")
    if origin:
        CORS(app, origins=[origin])
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
    
    # 注册向量数据库路由
    register_vector_routes(app)
    
    @app.route('/')
    def index():
        return "RAG Backend Service is Running!"

    return app

if __name__ == '__main__':
    app = create_app()
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("启动 RAG 后端服务...")
    print(f"API 地址: http://{host if host != '127.0.0.1' else 'localhost'}:{port}/api/vector/")
    app.run(host=host, port=port, debug=debug)
