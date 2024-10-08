import re
import spacy
from typing import List, Tuple
from collections import deque

# 加载 spaCy 英文和中文模型
try:
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    print("正在下载 en_core_web_sm 模型...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp_en = spacy.load("en_core_web_sm")

try:
    nlp_zh = spacy.load("zh_core_web_sm")
except OSError:
    print("正在下载 zh_core_web_sm 模型...")
    from spacy.cli import download
    download("zh_core_web_sm")
    nlp_zh = spacy.load("zh_core_web_sm")


def protect_code_blocks(text: str) -> Tuple[str, List[str]]:
    """
    使用正则表达式匹配 Markdown 中的代码块，并将其替换为带有唯一标识的占位符。
    返回处理后的文本和代码块列表。
    """
    code_block_pattern = re.compile(r'```(?:python|py|bash|sh|javascript|js|java|c\+\+|cpp|csharp|cs|ruby|go|rust|[^`])*?```', re.DOTALL)
    code_blocks = code_block_pattern.findall(text)
    placeholders = [f"{{{{CODE_BLOCK_{i}}}}}" for i in range(len(code_blocks))]
    text_without_code = code_block_pattern.sub(lambda _: placeholders.pop(0), text)
    return text_without_code, code_blocks


def protect_inline_code(text: str) -> Tuple[str, List[str]]:
    """
    使用正则表达式匹配 Markdown 中的行内代码，并将其替换为带有唯一标识的占位符。
    返回处理后的文本和行内代码列表。
    """
    inline_code_pattern = re.compile(r'`[^`]+?`')
    inline_codes = inline_code_pattern.findall(text)
    placeholders = [f"{{{{INLINE_CODE_{i}}}}}" for i in range(len(inline_codes))]
    text_without_inline = inline_code_pattern.sub(lambda _: placeholders.pop(0), text)
    return text_without_inline, inline_codes


def protect_markdown_elements(text: str) -> Tuple[str, dict]:
    """
    保护Markdown中的代码块、行内代码、表格，并返回处理后的文本和所有被保护的内容。
    """
    protected = {}

    # 保护代码块
    text, code_blocks = protect_code_blocks(text)
    for i, block in enumerate(code_blocks):
        placeholder = f"{{{{CODE_BLOCK_{i}}}}}"
        protected[placeholder] = block.strip()

    # 保护行内代码
    text, inline_codes = protect_inline_code(text)
    for i, code in enumerate(inline_codes):
        placeholder = f"{{{{INLINE_CODE_{i}}}}}"
        protected[placeholder] = code.strip()

    # 保护表格
    table_pattern = re.compile(r'(?:\|.*\|(?:\n|$))+')
    tables = table_pattern.findall(text)
    for i, table in enumerate(tables):
        placeholder = f"{{{{TABLE_{i}}}}}"
        protected[placeholder] = table.strip()
    text = table_pattern.sub(lambda _: f"{{{{TABLE_{i}}}}}", text, count=len(tables))

    return text, protected


def restore_protected_elements(sentences: List[str], protected: dict) -> List[str]:
    """
    将句子中的占位符替换回原始的被保护内容。
    """
    restored_sentences = []
    for sentence in sentences:
        for placeholder, content in protected.items():
            if placeholder in sentence:
                sentence = sentence.replace(placeholder, content)
        restored_sentences.append(sentence.strip())
    return restored_sentences


def split_large_code_blocks(code_block: str, max_lines: int = 1000) -> List[str]:
    """
    将超大的代码块按行数分割为多个较小的代码片段。
    """
    lines = code_block.split('\n')
    return ['\n'.join(lines[i:i + max_lines]) for i in range(0, len(lines), max_lines)]


def split_markdown_list(text: str) -> List[str]:
    """
    识别并单独处理Markdown中的列表项。
    返回处理后的句子列表。
    """
    sentences = []
    lines = text.split('\n')
    list_pattern = re.compile(r'^(\s*)([-*+]|\d+\.)\s+')

    current_list = deque()
    for line in lines:
        if list_pattern.match(line):
            current_list.append(line.strip())
        else:
            if current_list:
                sentences.extend(list(current_list))
                current_list.clear()
            if line.strip():
                sentences.append(line.strip())
    if current_list:
        sentences.extend(list(current_list))
    return sentences


def split_markdown_blockquotes(text: str) -> List[str]:
    """
    识别并单独处理Markdown中的引用块。
    返回处理后的句子列表。
    """
    sentences = []
    lines = text.split('\n')
    blockquote_pattern = re.compile(r'^>\s+')

    current_quote = deque()
    for line in lines:
        if blockquote_pattern.match(line):
            current_quote.append(line.strip())
        else:
            if current_quote:
                quote = ' '.join(current_quote)
                sentences.append(quote)
                current_quote.clear()
            if line.strip():
                sentences.append(line.strip())
    if current_quote:
        quote = ' '.join(current_quote)
        sentences.append(quote)
    return sentences


def split_markdown_headers(text: str) -> List[str]:
    """
    识别并单独处理 Markdown 标题。
    返回处理后的句子列表。
    """
    lines = text.split('\n')
    sentences = []
    header_pattern = re.compile(r'^(#{1,6})\s+')

    for line in lines:
        line = line.strip()
        if header_pattern.match(line):
            sentences.append(line)
        elif line:
            sentences.append(line)
    return sentences


def split_markdown_tables(text: str) -> List[str]:
    """
    识别并单独处理Markdown中的表格。
    返回处理后的句子列表。
    """
    sentences = []
    lines = text.split('\n')
    table_pattern = re.compile(r'^\|.*\|$')
    current_table = deque()

    for line in lines:
        if table_pattern.match(line):
            current_table.append(line.strip())
        else:
            if current_table:
                table = '\n'.join(current_table)
                sentences.append(table)
                current_table.clear()
            if line.strip():
                sentences.append(line.strip())
    if current_table:
        table = '\n'.join(current_table)
        sentences.append(table)
    return sentences


def split_sentences(text: str) -> List[str]:
    # Step 1: 保护Markdown中的特殊元素
    text, protected = protect_markdown_elements(text)

    # Step 2: 按照占位符分割文本
    # 此时，文本中已替换了代码块、行内代码、表格为占位符

    # Step 3: 逐行处理列表和引用块
    # 首先处理列表
    list_sentences = split_markdown_list(text)

    intermediate_sentences = []
    for segment in list_sentences:
        # 处理引用块
        if segment.startswith('>'):
            quote_sentences = split_markdown_blockquotes(segment)
            intermediate_sentences.extend(quote_sentences)
        else:
            intermediate_sentences.append(segment)

    # Step 4: 处理表格
    final_sentences = []
    for segment in intermediate_sentences:
        if segment.startswith('|') and segment.endswith('|'):
            final_sentences.append(segment)
        else:
            # 处理Markdown标题
            header_sentences = split_markdown_headers(segment)
            for header in header_sentences:
                if not header:
                    continue
                # 判断是否为中文或英文
                if re.search(r'[\u4e00-\u9fff]', header):
                    # 中文或混合文本
                    sentences = split_chinese(header)
                    final_sentences.extend(sentences)
                else:
                    # 英文或其他文本
                    sentences = split_english(header)
                    final_sentences.extend(sentences)

    # Step 5: 处理被保护的内容
    # 这里不需要恢复代码块和表格，因为它们作为独立的句子已经存在

    # Step 6: 进一步处理代码块占位符
    # 如果有超大的代码块，则需要进一步分割
    processed_sentences = []
    for sentence in final_sentences:
        code_block_match = re.match(r'^\{\{CODE_BLOCK_(\d+)$', sentence)
        table_match = re.match(r'^\{\{TABLE_(\d+)$', sentence)
        if code_block_match:
            code_block = protected.get(sentence, '')
            # 判断代码块是否过大，进行分割
            code_lines = code_block.count('\n') + 1
            if code_lines > 1000:
                split_blocks = split_large_code_blocks(code_block, max_lines=1000)
                for split_block in split_blocks:
                    processed_sentences.append(split_block)
            else:
                processed_sentences.append(code_block)
        elif table_match:
            table = protected.get(sentence, '')
            processed_sentences.append(table)
        else:
            processed_sentences.append(sentence)

    # Step 7: 处理行内代码占位符
    # 将行内代码占位符替换回原始内容
    final_restored_sentences = restore_protected_elements(processed_sentences, protected)

    # Step 8: 移除可能的多余空白字符
    final_restored_sentences = [s for s in final_restored_sentences if s]

    return final_restored_sentences


def split_english(text: str) -> List[str]:
    """
    使用 spaCy 对英文文本进行分句。
    """
    doc = nlp_en(text)
    return [sent.text.strip() for sent in doc.sents]


def split_chinese(text: str) -> List[str]:
    """
    使用正则表达式对中文文本进行分句。
    仅根据中文句子结束标点（。！？；）进行分割，不进行进一步拆分。
    """
    # 定义中文句子结束标点
    chinese_punct = '。！？；'
    # 使用正则表达式按标点分割，保留标点
    pattern = re.compile(f'[^{"".join(chinese_punct)}]*[{chinese_punct}]')
    sentences = pattern.findall(text)
    # 处理可能的剩余部分
    remaining = pattern.sub('', text).strip()
    if remaining:
        sentences.append(remaining)
    # 过滤空句子并去除多余空白字符
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences
