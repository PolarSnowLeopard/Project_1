from flask import Flask, request
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
from inference import InferenceModel
import torch

app = Flask(__name__)

CORS(app)

api = Api(app, version='1.0', 
         title='图神经网络推理API',
         description='用于处理图数据集并进行模型推理的API接口',
         doc='/docs')

# 创建命名空间
ns = api.namespace('api', description='推理操作')

# 定义上传文件的参数模型
upload_parser = api.parser()
upload_parser.add_argument('file',
                          type=FileStorage,
                          location='files',
                          required=True,
                          help='ZIP格式的数据集文件')

# 创建推理模型实例（全局）
inference_model = InferenceModel(
    model_path='checkpoint/model.pt',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 初始化模型
inference_model.load_model(
    in_feats=211,
    h_feats=211,
    out_feats=3
)

@ns.route('/predict')
class Prediction(Resource):
    @ns.expect(upload_parser)
    @ns.doc(responses={
        200: 'Success',
        400: 'Validation Error',
        500: 'Internal Server Error'
    })
    def post(self):
        """
        上传数据集并进行推理
        上传一个包含完整图数据集的ZIP文件，系统将进行模型推理并返回结果。
        ZIP文件必须包含以下文件：
        - meta.yaml: 数据集元信息
        - edges_*.csv: 边数据文件
        - nodes_*.csv: 节点数据文件
        """
        try:
            # 获取上传的文件
            args = upload_parser.parse_args()
            file = args['file']
            
            if not file:
                api.abort(400, "没有上传文件")
                
            if not file.filename.endswith('.zip'):
                api.abort(400, "请上传ZIP格式的文件")
            
            # 处理数据集并进行推理
            dataset_path = inference_model.process_uploaded_dataset(file)
            result = inference_model.infer(dataset_path)
            
            return {'result': result}
            
        except Exception as e:
            api.abort(500, f"处理过程出错: {str(e)}")

@ns.route('/health')
class Health(Resource):
    @ns.doc(responses={200: 'Success'})
    def get(self):
        """
        健康检查接口
        用于检查API服务是否正常运行
        """
        return {'status': 'healthy'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)