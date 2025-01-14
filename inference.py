import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import shutil
import zipfile
import yaml
from methods.train import model, initialize_features, edge_initialize_features

class InferenceModel:
    def __init__(self, model_path='checkpoint/model.pt', device='cpu'):
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = None
        
    def load_model(self, in_feats, h_feats, out_feats):
        """加载预训练模型"""
        self.model = model(in_feats, h_feats, out_feats)
            
        # 加载模型参数
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def process_uploaded_dataset(self, zip_file, save_dir='temp_dataset'):
        """
        处理上传的数据集压缩文件
        Args:
            zip_file: 上传的ZIP文件对象
            save_dir: 保存数据集的目录
        Returns:
            str: 保存的数据集路径
        """
        try:
            # 确保目录为空
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)
            
            # 解压文件
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # 修改解压方式，处理文件名编码
                for file_info in zip_ref.filelist:
                    try:
                        # 尝试使用UTF-8解码
                        file_info.filename = file_info.filename.encode('cp437').decode('utf-8')
                    except UnicodeDecodeError:
                        # 如果失败，尝试使用GBK解码
                        file_info.filename = file_info.filename.encode('cp437').decode('gbk')
                    zip_ref.extract(file_info, save_dir)
            
            # 验证必要的文件是否存在
            required_files = ['meta.yaml']
            for file in required_files:
                file_path = os.path.join(save_dir, file)
                if not os.path.exists(file_path):
                    raise Exception(f"缺少必要的文件: {file}")
                # 验证yaml文件的编码
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        yaml.safe_load(f)
                except UnicodeDecodeError:
                    # 如果UTF-8失败，尝试使用GBK重新读取
                    with open(file_path, 'r', encoding='gbk') as f:
                        content = f.read()
                    # 将内容以UTF-8重新写入
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
            
            return save_dir
            
        except zipfile.BadZipFile:
            raise Exception("无效的ZIP文件")
        except Exception as e:
            # 清理临时目录
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            raise Exception(f"处理数据集时出错: {str(e)}")
        
    def prepare_graph(self, data_path):
        """准备图数据"""
        try:
            # 加载数据集
            dataset = dgl.data.CSVDataset(
                data_path,
                force_reload=True,
                csv_kwargs={'encoding': 'utf-8'}
            )
            g = dataset[0]
            
            # 准备特征
            node_features = {}
            edge_features = {}
            feature_dim = g.nodes['app'].data['feat'].shape[1]
            edge_feature_dim = g.edges['edges_1'].data['feat'].shape[1]
            
            # 初始化特征
            initialize_features(g, feature_dim, init_type='zero')
            edge_initialize_features(g, edge_feature_dim, init_type='zero')
            
            # 收集特征
            for ntype in g.ntypes:
                feat = g.nodes[ntype].data['feat']
                if feat is not None:
                    node_features[ntype] = feat
                    
            for etype in g.etypes:
                feat = g.edges[etype].data['feat']
                if feat is not None:
                    edge_features[etype] = feat
                    
            # 将数据移到指定设备
            g = g.to(self.device)
            node_features = {ntype: feats.to(self.device) for ntype, feats in node_features.items()}
            edge_features = {etype: feats.to(self.device) for etype, feats in edge_features.items()}
            
            return g, node_features, edge_features
        except Exception as e:
            raise Exception(f"准备图数据时出错: {str(e)}")
        
    def infer(self, data_path):
        """
        进行模型推理
        Args:
            data_path: 输入数据的路径（与训练数据格式相同）
        Returns:
            str: 包含推理结果的字符串
        """
        try:
            # 准备数据
            g, node_features, edge_features = self.prepare_graph(data_path)
            
            # 进行推理
            with torch.no_grad():
                logits = self.model(g, node_features, edge_features)
                
                # 获取app节点的预测结果
                app_logits = logits['app']
                predictions = torch.argmax(app_logits, dim=1)
                
                # 将预测结果转换为可读格式
                result_str = "推理结果:\n"
                for i, pred in enumerate(predictions):
                    result_str += f"节点 {i}: 类别 {pred.item()}\n"
                
                # 清理临时数据集
                if os.path.dirname(data_path) == 'temp_dataset':
                    shutil.rmtree(data_path, ignore_errors=True)
                
                return result_str
                
        except Exception as e:
            # 确保清理临时文件
            if os.path.dirname(data_path) == 'temp_dataset':
                shutil.rmtree(data_path, ignore_errors=True)
            return f"推理过程出错: {str(e)}"

def main():
    # 示例用法
    inference_model = InferenceModel(
        model_path='checkpoint/model.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 加载模型（参数需要与训练时相同）
    inference_model.load_model(
        in_feats=211,  # 输入特征维度
        h_feats=211,   # 隐藏层维度需要与checkpoint匹配
        out_feats=3    # 输出类别数
    )
    
    # 模拟处理上传的ZIP文件
    try:
        # 这里假设有一个测试用的ZIP文件
        with open('test_dataset.zip', 'rb') as zip_file:
            dataset_path = inference_model.process_uploaded_dataset(zip_file)
            result = inference_model.infer(dataset_path)
            print(result)
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == '__main__':
    main() 