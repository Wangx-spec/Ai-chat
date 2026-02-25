# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing load method for PDF files."""

import logging
import re
from pathlib import Path
import tempfile
import base64
import requests
from tqdm import tqdm

import pandas as pd
from io import BytesIO

from graphrag.config.models.input_config import InputConfig
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.index.input.util import load_files
from graphrag.logger.base import ProgressLogger
from graphrag.storage.pipeline_storage import PipelineStorage

log = logging.getLogger(__name__)


def to_b64(file_path):
    """将文件转换为base64编码
    二进制数据（如 PDF 文件）不能直接传输。Base64 编码将二进制数据转换为 ASCII 字符串，使其可以安全地嵌入到 JSON 中。
    """
    try:
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        raise Exception(f'File: {file_path} - Info: {e}')


def do_parse(file_path, url='http://192.168.110.131:8000/predict', **kwargs):
    """调用MinerU远程Server服务解析PDF文件"""
    try:
        response = requests.post(url, json={
            'file': to_b64(file_path),
            'kwargs': kwargs
        })

        if response.status_code == 200:
            output = response.json()
            output['file_path'] = file_path
            return output
        else:
            raise Exception(response.text)
    except Exception as e:
        log.error(f'File: {file_path} - Info: {e}')
        return None


def get_parse_results(output_dir, include_content=True, include_images=False, url='http://192.168.110.131:8000/get_parse_results'):
    """获取解析结果"""
    try:
        response = requests.get(
            url, 
            params={
                'output_dir': output_dir,   # 远程服务器的存储路径
                'include_content': str(include_content).lower(),  # 解析出来的文档内容
                'include_images': str(include_images).lower()    # 解析出来的图片
            }
        )
        if response.status_code == 200:
            result = response.json()
            log.info(f"获取到结果，文件数量: {len(result.get('files', []))}")
            return result
        else:
            log.error(f"获取解析结果失败: {response.text}")
            return None
    except Exception as e:
        log.error(f"获取解析结果异常: {str(e)}")
        return None


def extract_text_from_block(block):
    """从块中提取文本内容"""
    text = ""
    if 'lines' in block:
        for line in block['lines']:
            if 'spans' in line:
                for span in line['spans']:
                    if span.get('type') == 'text' and 'content' in span:
                        text += span['content'] + " "
    return text.strip()


def get_category_name(category_id):
    """将类别ID转换为名称"""
    categories = {
        0: "title",
        1: "plain_text",
        2: "abandon",
        3: "figure",
        4: "figure_caption",
        5: "table",
        6: "table_caption",
        7: "table_footnote",
        8: "isolate_formula",
        9: "formula_caption",
        13: "embedding",
        14: "isolated",
        15: "text"
    }
    return categories.get(category_id, f"unknown_{category_id}")


def extract_structured_info(parse_content):
    """从解析结果中提取结构化信息，正确处理层级关系"""
    structured_info = {}
    
    # 初始化带有默认值的结构
    content_types = {"default": "text"}  # 添加一个默认字段
    structure = {"has_content": True}    # 添加一个默认字段
    
    # 处理图片信息
    if 'images' in parse_content and parse_content['images']:
        content_types["images"] = len(parse_content['images'])
    
    # 处理内容列表
    if 'contents' in parse_content:
        for file_name, content in parse_content['contents'].items():
            if file_name.endswith('.json'):
                # 处理JSON文件内容
                if 'pdf_info' in content:
                    for page_info in content['pdf_info']:
                        # 提取页面信息
                        if 'page_idx' in page_info:
                            page_key = f"page_{page_info['page_idx']}"
                            structure[page_key] = {
                                "blocks": len(page_info.get('para_blocks', [])),
                                "images": len(page_info.get('images', [])),
                                "tables": len(page_info.get('tables', []))
                            }
    
    # 只有当有实际内容时才添加到结构化信息中
    if len(content_types) > 1:  # 不只有默认字段
        structured_info["content_types"] = content_types
    
    if len(structure) > 1:  # 不只有默认字段
        structured_info["structure"] = structure
    
    return structured_info


async def process_pdf_file(path: str, group: dict | None, storage: PipelineStorage):
    """处理单个PDF文件"""
    if group is None:
        group = {}
    
    # 获取实际文件路径
    if hasattr(storage, 'get_file_path'):
        file_path = storage.get_file_path(path)
    else:
        # 如果存储没有提供获取文件路径的方法，则需要临时下载文件
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(await storage.get(path, as_bytes=True))
            file_path = temp_file.name
    
    # 调用远程服务解析PDF
    result = do_parse(file_path)
    if not result or 'output_dir' not in result:
        raise ValueError(f"解析PDF文件失败: {file_path}")
    
    # 获取输出目录
    output_dir = result['output_dir']
    if not output_dir.endswith('/auto'):
        output_dir = f"{output_dir}/auto"
    
    # 获取解析结果
    parse_content = get_parse_results(output_dir)
    if not parse_content:
        raise ValueError(f"获取解析结果失败: {output_dir}")
    
    # 提取Markdown内容作为主文本
    text_content = ""
    if 'markdown_text' in parse_content and parse_content['markdown_text']:
        text_content = parse_content['markdown_text']
    
    # 创建DataFrame行
    data = pd.DataFrame([{
        "text": text_content,
        "title": Path(path).name,
        "id": gen_sha512_hash({"text": text_content, "path": path}, ["text", "path"])
    }])
    
    # 添加分组信息
    for key, value in group.items():
        data[key] = value
    
    # 添加元数据
    metadata = {
        "file_path": path,
        "output_dir": output_dir,
        "parse_time": pd.Timestamp.now().isoformat(),
    }
    
    # 提取结构化信息
    structured_info = extract_structured_info(parse_content)
    if structured_info:
        metadata.update(structured_info)
    else:
        # 确保metadata中有一个有效的结构，而不是空结构
        metadata["content_types"] = {"default": "text"}
    
    data["metadata"] = [metadata]
    
    # 添加创建日期
    creation_date = await storage.get_creation_date(path)
    data["creation_date"] = creation_date
    
    return data


async def load_pdf(
    config: InputConfig,
    progress: ProgressLogger | None,
    storage: PipelineStorage,
) -> pd.DataFrame:
    """Load PDF inputs from a directory using remote parsing service."""
    log.info("Loading PDF files from %s", config.base_dir)
    
    async def load_file(path: str, group: dict | None) -> pd.DataFrame:
        if group is None:
            group = {}
        try:
            # 以二进制方式获取PDF内容
            buffer = BytesIO(await storage.get(path, as_bytes=True))
        
            # 将二进制内容保存到临时文件, 使用临时文件路径调用处理函数
            # 因为远程服务需要的是文件路径
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(buffer.getvalue())
                file_path = temp_file.name
            
            # 调用MinerU远程Server服务解析PDF
            result = do_parse(file_path)
            if not result or 'output_dir' not in result:
                log.error(f"解析PDF文件失败: {file_path}")
                # 创建一个包含错误信息的DataFrame
                data = pd.DataFrame([{
                    "text": f"[解析失败] {path}",
                    "title": Path(path).name,  # 使用文件名作为标题
                    "id": path  # 使用文件路径作为ID
                }])
            else:
                # 获取输出目录
                output_dir = result['output_dir']
                if not output_dir.endswith('/auto'):
                    output_dir = f"{output_dir}/auto"
                
                # 从output_dir中提取唯一标识符作为ID
                # 格式通常是: /home/07_minerU/tmp/1742279064_be34722f-7fc8-432b-87ba-9acaa0b021d4/auto
                id_match = re.search(r'\/([^\/]+)\/auto$', output_dir)
                doc_id = id_match.group(1) if id_match else path
                
                # 获取解析结果
                parse_content = get_parse_results(output_dir)
                if not parse_content:
                    log.error(f"获取解析结果失败: {output_dir}")
                    # 创建一个包含错误信息的DataFrame
                    data = pd.DataFrame([{
                        "text": f"[获取解析结果失败] {path}",
                        "title": Path(path).name,  # 使用文件名作为标题
                        "id": doc_id  # 使用提取的ID
                    }])
                else:
                    # 提取Markdown内容作为主文本
                    text_content = ""
                    if 'markdown_text' in parse_content and parse_content['markdown_text']:
                        text_content = parse_content['markdown_text']
             
                    # 创建DataFrame行
                    data = pd.DataFrame([{
                        "text": text_content,
                        "title": Path(path).name,  # 使用文件名作为标题
                        "id": doc_id  # 使用提取的ID
                    }])
                    
        
                    # 添加元数据
                    metadata = {
                        "file_path": path,
                        "output_dir": output_dir,
                        "parse_time": pd.Timestamp.now().isoformat(),
                        "doc_id": doc_id  # 在元数据中保存ID
                    }
                    
                    # 提取结构化信息
                    structured_info = extract_structured_info(parse_content)
                    if structured_info:
                        metadata.update(structured_info)
                    
                    data["metadata"] = [metadata]
            
            # 添加分组信息
            for key, value in group.items():
                data[key] = value
            
            # 添加创建日期
            creation_date = await storage.get_creation_date(path)
            data["creation_date"] = creation_date
            return data
            
        except Exception as e:
            log.exception(f"处理PDF文件时出错: {path}, 错误: {str(e)}")
            # 创建一个包含错误信息的DataFrame
            data = pd.DataFrame([{
                "text": f"[处理错误] {path}: {str(e)}",
                "title": Path(path).name,  # 使用文件名作为标题
                "id": path  # 使用文件路径作为ID
            }])
            
            # 添加分组信息
            for key, value in group.items():
                data[key] = value
            
            # 添加创建日期
            try:
                creation_date = await storage.get_creation_date(path)
                data["creation_date"] = creation_date
            except:
                data["creation_date"] = pd.Timestamp.now()
            
            return data

    # 使用现有的load_files函数来处理文件加载
    return await load_files(load_file, config, storage, progress) 