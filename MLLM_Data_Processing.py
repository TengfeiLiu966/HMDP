#########################################################MLLM-Data-Processing
import os
import json
import numpy as np
from sklearn import metrics
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type
)
from swift.utils import seed_everything
import torch
import re
import random
from tqdm import tqdm

# 模型初始化
model_type = 'llava1_6-mistral-7b-instruct'
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16, model_kwargs={'device_map': 'auto'})

# 数据文件路径
json_file = './MMaterials/exMMaterials_train.json'
image_folder = './MMaterials/Images'

# 输出文件路径
formatted_json_file = './MMaterials/llava_finetune_data_train_0.3.json'

# 定义类别映射
label_mapping = {
    "1000000": 0,
    "0100000": 1,
    "0010000": 2,
    "0001000": 3,
    "0000100": 4,
    "0000010": 5,
    "0000001": 6
}

# 随机选择忽略的样本（30%）
with open(json_file, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

num_ignore_images = int(total_lines * 0.3)
ignore_image_lines = set(random.sample(range(1, total_lines + 1), num_ignore_images))
print(f"Lines to ignore images: {sorted(ignore_image_lines)}")

# 初始化结果列表
formatted_data = []

# 逐行读取 JSON 文件并重新组织
with open(json_file, 'r', encoding='utf-8') as f:
    for idx, line in tqdm(enumerate(f, start=1), total=total_lines):
        # 去掉首尾空格
        line = line.strip()
        # 按 \t 分割标签和 JSON 内容
        parts = line.split('\t')
        if len(parts) != 2:
            print(f"Invalid line format: {line}")
            continue

        # 提取标签和 JSON 数据部分
        raw_label = parts[0]  # 标签部分
        json_content = parts[1]  # JSON 内容部分

        # 解析 JSON 数据
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e} in line: {line}")
            continue

        # 将原始标签映射为类别索引
        if raw_label not in label_mapping:
            print(f"Label {raw_label} not found in mapping. Skipping...")
            continue

        label = label_mapping[raw_label]
        response = str(label)  # 目标响应为类别索引

        # 如果当前行号在 ignore_image_lines 中，忽略图像
        if idx in ignore_image_lines:
            images = ['./placeholder_image.jpg']  # 替换为本地的占位图像路径
            print(f"Ignoring images for line {idx}")
        else:
            # 正常处理图像路径
            sample_folder = os.path.join(image_folder, data['title'])
            if not os.path.exists(sample_folder):
                print(f"Sample folder not found: {sample_folder}")
                continue

            images = [
                os.path.join(sample_folder, img) for img in os.listdir(sample_folder)
                if img.lower().endswith(('.png', '.jpg', '.jpeg'))
            ][:9]
            if not images:
                print(f"No images found in folder: {sample_folder}")
                continue

        # 获取前 2048 个 tokens 的文本内容
        text_content = ' '.join(value for key, value in data.items() if key != 'title' and key != 'label')
        tokenized_text = tokenizer.tokenize(text_content)
        truncated_text = tokenizer.convert_tokens_to_string(tokenized_text[:1500])
        # 构建 query
        if images:
            query = (
                "Please combine the following text and associated images to determine which category the text belongs to. "
                "Only return the category number (0, 1, 2, 3, 4, 5, or 6) without any explanation or additional text:\n\n"
                "- 0: Composite materials\n"
                "- 1: Battery separators\n"
                "- 2: Energy storage materials\n"
                "- 3: Graphene\n"
                "- 4: Nanomaterials\n"
                "- 5: Silicon carbide\n"
                "- 6: Titanium alloys\n\n"
                f"The text is: {truncated_text}"
            )
        else:
            query = (
                "Please determine which category the following text belongs to without using any images. "
                "Only return the category number (0, 1, 2, 3, 4, 5, or 6) without any explanation or additional text:\n\n"
                "- 0: Composite materials\n"
                "- 1: Battery separators\n"
                "- 2: Energy storage materials\n"
                "- 3: Graphene\n"
                "- 4: Nanomaterials\n"
                "- 5: Silicon carbide\n"
                "- 6: Titanium alloys\n\n"
                f"The text is: {truncated_text}"
            )

        # 构建输出格式
        formatted_data.append({
            "query": query,
            "response": response,
            "images": images
        })

# 保存到文件
with open(formatted_json_file, 'w', encoding='utf-8') as f:
    for entry in formatted_data:
        f.write(json.dumps(entry) + '\n')

print(f"Formatted data saved to {formatted_json_file}")
