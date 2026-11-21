"""
API集成模块
将向量数据库功能集成到Flask应用中
"""

from flask import Blueprint, request, jsonify
import os
import logging
from tempfile import TemporaryDirectory
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

from .vector_db_manager import VectorDatabaseManager
from .vector_retriever import VectorRetriever

# 创建蓝图
vector_bp = Blueprint('vector', __name__, url_prefix='/api/vector')

# 全局变量存储向量系统实例
vector_manager: VectorDatabaseManager = None
vector_retriever: VectorRetriever = None

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_vector_system(
    milvus_host: str = None,
    milvus_port: str = None,
    embedding_model: str = None,
    dashscope_api_key: str = None
):
    """初始化向量系统"""
    global vector_manager, vector_retriever
    
    try:
        vector_manager = VectorDatabaseManager(
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            embedding_model=embedding_model,
            dashscope_api_key=dashscope_api_key
        )
        vector_retriever = VectorRetriever(
            vector_manager, similarity_threshold=float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.5"))
        )
        logger.info(f"向量系统初始化成功，连接到 Milvus at {milvus_host}:{milvus_port}")
        return True
    except Exception as e:
        logger.error(f"向量系统初始化失败: {str(e)}")
        return False


@vector_bp.route('/upload_document', methods=['POST'])
def upload_document():
    """上传并处理文档"""
    global vector_manager
    
    if not vector_manager:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    try:
        data = request.get_json()
        if not data or 'file_path' not in data or 'collection_name' not in data:
            return jsonify({
                'success': False,
                'message': '请提供 file_path 和 collection_name 参数'
            }), 400
        
        file_path = data['file_path']
        collection_name = data['collection_name']
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'message': f'文件不存在: {file_path}'
            }), 400
        
        # 处理文档
        success = vector_manager.process_file(file_path, collection_name)
        
        if success:
            # 获取数据库信息
            db_info = vector_manager.get_database_info(collection_name)
            
            return jsonify({
                'success': True,
                'message': f'文档处理成功: {file_path}',
                'database_info': db_info
            })
        else:
            return jsonify({
                'success': False,
                'message': f'文档处理失败: {file_path}'
            }), 500
            
    except Exception as e:
        logger.error(f"文档上传API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'文档处理失败: {str(e)}'
        }), 500


@vector_bp.route('/upload_file', methods=['POST'])
def upload_file():
    """上传文件流处理"""
    global vector_manager
    
    if not vector_manager:
        return jsonify({'success': False, 'message': '向量系统未初始化'}), 400

    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(success=False, message="请选择要上传的文档"), 400

    collection_name = request.form.get("collection_name", "agent_rag").strip()
    filename = Path(file.filename.replace("\\", "/")).name
    suffix = Path(filename).suffix.lower()
    if not collection_name or suffix not in {".txt", ".csv", ".pdf", ".docx", ".xlsx", ".xls"}:
        return jsonify(success=False, message="请填写知识库名称并选择支持的文件格式"), 400

    try:
        with TemporaryDirectory(prefix="rag-upload-") as temp_dir:
            file_path = Path(temp_dir) / f"upload{suffix}"
            file.save(file_path)
            if not vector_manager.process_file(str(file_path), collection_name, source_name=filename):
                return jsonify(success=False, message="文档入库失败，请检查文件内容和知识库配置"), 500
        return jsonify(
            success=True,
            message=f"文件上传并处理成功: {filename}",
            database_info=vector_manager.get_database_info(collection_name),
        )
    except Exception:
        logger.exception("Failed to process uploaded document")
        return jsonify(success=False, message="文档上传失败"), 500


def _source_payload(document, score: float) -> dict:
    metadata = dict(document.metadata)
    filename = Path(str(metadata.get("source") or metadata.get("name") or "").replace("\\", "/")).name
    if "source" in metadata:
        metadata["source"] = filename
    page = metadata.get("page_label")
    if page is None and isinstance(metadata.get("page"), int):
        page = metadata["page"] + 1
    return {
        "content": document.page_content, "metadata": metadata,
        "filename": filename, "page": page, "score": score,
    }


@vector_bp.route('/query', methods=['POST'])
def query_documents():
    """查询文档"""
    global vector_retriever
    
    if not vector_retriever:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(success=False, message="请提交 JSON 对象"), 400
    question = data.get("question")
    collection_name = data.get("collection_name")
    k = data.get("k", 5)
    if not isinstance(question, str) or not question.strip():
        return jsonify(success=False, message="请输入问题"), 400
    if not isinstance(collection_name, str) or not collection_name.strip():
        return jsonify(success=False, message="请选择知识库"), 400
    if type(k) is not int or not 1 <= k <= 20:
        return jsonify(success=False, message="k 必须是 1 到 20 的整数"), 400

    result = vector_retriever.answer_question(question.strip(), collection_name.strip(), k=k)
    upstream_error = result.status in {"retrieval_error", "generation_error"}
    return jsonify(
        success=not upstream_error,
        status=result.status,
        question=question,
        answer=result.answer,
        message=result.answer if upstream_error else None,
        confidence=result.confidence,
        question_type=result.question_type,
        sources=[_source_payload(doc, score) for doc, score in zip(result.source_documents, result.scores)],
    ), 502 if upstream_error else 200


@vector_bp.route('/search', methods=['POST'])
def search_similar():
    """相似性搜索"""
    global vector_retriever
    
    if not vector_retriever:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    try:
        data = request.get_json()
        if not data or 'query' not in data or 'collection_name' not in data:
            return jsonify({
                'success': False,
                'message': '请提供 query 和 collection_name 参数'
            }), 400
        
        query = data['query']
        collection_name = data['collection_name']
        k = data.get('k', 5)
        
        # 执行相似性搜索
        results = vector_retriever.search_similar_content(query, k=k, collection_name=collection_name)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': [
                _source_payload(doc, score) for doc, score in results
            ]
        })
        
    except Exception as e:
        logger.error(f"搜索API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': '检索服务暂不可用，请稍后重试'
        }), 502


@vector_bp.route('/collection_info', methods=['GET'])
def get_collection_info():
    """获取集合信息"""
    global vector_manager
    
    if not vector_manager:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    collection_name = request.args.get('collection_name')
    if not collection_name:
        return jsonify({
            'success': False,
            'message': '请提供 collection_name 参数'
        }), 400

    try:
        db_info = vector_manager.get_database_info(collection_name)
        return jsonify({
            'success': True,
            'database_info': db_info
        })
        
    except Exception as e:
        logger.error(f"数据库信息API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'获取数据库信息失败: {str(e)}'
        }), 500


@vector_bp.route('/clear_collection', methods=['POST'])
def clear_collection():
    """清空集合"""
    global vector_manager
    
    if not vector_manager:
        return jsonify({
            'success': False,
            'message': '向量系统未初始化'
        }), 400
    
    data = request.get_json()
    if not data or 'collection_name' not in data:
        return jsonify({
            'success': False,
            'message': '请提供 collection_name 参数'
        }), 400

    collection_name = data['collection_name']

    try:
        vector_manager.clear_database(collection_name)
        return jsonify({
            'success': True,
            'message': f"集合 '{collection_name}' 已清空"
        })
        
    except Exception as e:
        logger.error(f"清空数据库API错误: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'清空数据库失败: {str(e)}'
        }), 500


# 错误处理
@vector_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'message': '接口不存在'
    }), 404


@vector_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'message': '服务器内部错误'
    }), 500


def register_vector_routes(app):
    """注册向量数据库路由到Flask应用"""
    app.register_blueprint(vector_bp)
    
    # 自动初始化向量系统
    with app.app_context():
        init_vector_system()
    
    logger.info("向量数据库API路由已注册")
