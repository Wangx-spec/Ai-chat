# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Markdown-based text chunking strategy."""

import re
from collections.abc import Iterable
import logging

from graphrag.config.models.chunking_config import ChunkingConfig
from graphrag.index.operations.chunk_text.typing import TextChunk
from graphrag.logger.progress import ProgressTicker

log = logging.getLogger(__name__)

def run_markdown(
    input: list[str],
    config: ChunkingConfig,
    tick: ProgressTicker,
) -> Iterable[TextChunk]:
    """Chunks text based on Markdown structure, keeping tables and images with their context."""
    
    print(f"开始Markdown切分，文档数量: {len(input)}")
    print(f"配置: 最大块大小={config.size}, 重叠={config.overlap}")
    
    # 标题正则表达式，匹配 # 到 ###### 的标题
    header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    # 表格正则表达式，匹配 HTML 表格或 Markdown 表格
    table_pattern = re.compile(r'(<html>.*?<table>.*?</table>.*?</html>|<table>.*?</table>)', re.DOTALL)
    
    # 图片正则表达式，匹配 Markdown 图片语法
    image_pattern = re.compile(r'!\[.*?\]\(.*?\)')
    
    for doc_idx, text in enumerate(input):
        print(f"处理文档 #{doc_idx+1}, 长度: {len(text) if text else 0} 字符")
        
        if not text:
            print(f"文档 #{doc_idx+1} 为空，跳过")
            tick(1)
            continue
            
        # 分割文本为段落
        paragraphs = text.split('\n\n')
        print(f"文档 #{doc_idx+1} 分割为 {len(paragraphs)} 个段落")
        
        current_chunk = []
        current_header_level = 0
        current_header = ""
        chunk_count = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            # 检查是否是标题
            header_match = header_pattern.match(paragraph.strip())
            
            if header_match:
                # 如果找到新标题
                header_level = len(header_match.group(1))
                header_text = header_match.group(2)
                print(f"发现标题 (级别 {header_level}): {header_text}")
                
                # 如果当前块不为空且找到更高级别的标题，输出当前块
                if current_chunk and (header_level <= current_header_level or current_header_level == 0):
                    chunk_text = "\n\n".join(current_chunk)
                    if current_header:
                        chunk_text = f"{current_header}\n\n{chunk_text}"
                    
                    chunk_count += 1
                    print(f"输出块 #{chunk_count}, 长度: {len(chunk_text)} 字符, 约 {len(chunk_text.split())} 词")
                    
                    yield TextChunk(
                        text_chunk=chunk_text,
                        source_doc_indices=[doc_idx],
                        n_tokens=len(chunk_text.split())  # 简单估计token数量
                    )
                    current_chunk = []
                
                current_header_level = header_level
                current_header = paragraph
            else:
                # 检查段落是否包含表格或图片
                has_table = bool(table_pattern.search(paragraph))
                has_image = bool(image_pattern.search(paragraph))
                
                if has_table:
                    print(f"段落 #{para_idx+1} 包含表格")
                    table_matches = table_pattern.findall(paragraph)
                    for i, table in enumerate(table_matches):
                        print(f"  表格 #{i+1} 预览: {table[:100]}...")  # 只打印前100个字符
                if has_image:
                    print(f"段落 #{para_idx+1} 包含图片")
                    image_matches = image_pattern.findall(paragraph)
                    for i, img in enumerate(image_matches):
                        print(f"  图片 #{i+1}: {img}")  # 打印完整的图片标记
                
                # 如果是表格或图片，确保与前后文本保持在一起
                if has_table or has_image:
                    # 如果当前块为空，添加当前标题作为上下文
                    if not current_chunk and current_header:
                        print(f"将标题添加到表格/图片块: {current_header}")
                        current_chunk.append(current_header)
                        current_header = ""
                    
                    # 添加包含表格或图片的段落
                    current_chunk.append(paragraph)
                    print(f"添加表格/图片段落到当前块, 当前块大小: {len(current_chunk)} 段落")
                    
                    # 如果块太大，输出当前块
                    chunk_size = len("\n\n".join(current_chunk).split())
                    if chunk_size > config.size:
                        chunk_text = "\n\n".join(current_chunk)
                        chunk_count += 1
                        print(f"块过大 ({chunk_size} > {config.size}), 输出块 #{chunk_count}")
                        
                        yield TextChunk(
                            text_chunk=chunk_text,
                            source_doc_indices=[doc_idx],
                            n_tokens=chunk_size
                        )
                        current_chunk = []
                else:
                    # 普通文本段落
                    current_chunk.append(paragraph)
                    print(f"添加普通段落到当前块, 当前块大小: {len(current_chunk)} 段落")
                    
                    # 如果块太大，输出当前块
                    chunk_size = len("\n\n".join(current_chunk).split())
                    if chunk_size > config.size:
                        chunk_text = "\n\n".join(current_chunk)
                        chunk_count += 1
                        print(f"块过大 ({chunk_size} > {config.size}), 输出块 #{chunk_count}")
                        
                        yield TextChunk(
                            text_chunk=chunk_text,
                            source_doc_indices=[doc_idx],
                            n_tokens=chunk_size
                        )
                        current_chunk = []
        
        # 处理最后一个块
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if current_header and current_header not in current_chunk:
                print(f"将标题添加到最后一个块: {current_header}")
                chunk_text = f"{current_header}\n\n{chunk_text}"
            
            chunk_count += 1
            print(f"输出最后一个块 #{chunk_count}, 长度: {len(chunk_text)} 字符, 约 {len(chunk_text.split())} 词")
                
            yield TextChunk(
                text_chunk=chunk_text,
                source_doc_indices=[doc_idx],
                n_tokens=len(chunk_text.split())
            )
        
        print(f"文档 #{doc_idx+1} 处理完成, 共生成 {chunk_count} 个块")
        tick(1) 