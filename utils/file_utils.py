import csv
import os, logging
from typing import List, Dict, Any, Union


def load_prompts(file_path, encoding='utf-8'):
    try:
        with open(file_path, "r", newline="", encoding=encoding) as f:
            reader = csv.DictReader(f)
            print(f"成功读取 {file_path}\n")
            return [row for row in reader]
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return []


import json

def load_json_prompts(file_path, encoding='utf-8'):
    data = []
    try:
        with open(file_path, "r", encoding=encoding) as f:
            # 逐行读取文件
            for i, line in enumerate(f, 1):
                line = line.strip() # 去除首尾空白符
                if line: # 跳过空行
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"警告: 第 {i} 行解析失败，已跳过 - {e}")
                        
        print(f"成功读取 {file_path}，共加载 {len(data)} 条数据\n")
        return data
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return []
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []


def save_to_csv(
    data_list: List[Dict[str, Any]], 
    output_dir: str, 
    filename: str, 
    encoding: str = 'utf-8-sig',  # 默认改用 sig，解决 Excel 中文乱码
    fill_missing: Any = ''        # 当某个字典缺少字段时，默认填什么
) -> bool:
    """
    将字典列表保存为 CSV 文件，具备自动补全表头、处理路径、防乱码等功能。
    """
    
    # 1. 基础数据校验
    if not data_list:
        logging.warning("⚠️ 数据列表为空，跳过写入。")
        return False
    
    if not isinstance(data_list, list) or not isinstance(data_list[0], dict):
        logging.error("❌ 数据格式错误：必须是字典列表 (List[Dict])。")
        return False

    try:
        # 2. 智能处理文件名和路径
        # 防止出现 .csv.csv 的情况
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
            
        full_path = os.path.join(output_dir, filename)
        
        # 确保目录存在
        dir_name = os.path.dirname(full_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True) # exist_ok=True 防止并发创建时的报错

        # 3. 动态获取所有可能的字段名 (鲁棒性核心)
        # 很多时候 dict_list 里每个 dict 的 key 可能不一样，这里取并集
        # 使用 dict.fromkeys 保持插入顺序 (Python 3.7+)，如果不在意顺序可用 set()
        all_keys = {} 
        for d in data_list:
            all_keys.update(d)
        fieldnames = list(all_keys.keys())

        # 4. 写入文件
        # newline='' 是为了防止 Windows 下出现多余空行
        with open(full_path, 'w', newline='', encoding=encoding) as csvfile:
            writer = csv.DictWriter(
                csvfile, 
                fieldnames=fieldnames, 
                restval=fill_missing,  # 如果某个 dict 缺少 key，自动填这个值
                extrasaction='ignore'  # 如果 dict 有 header 没涵盖的 key，忽略而不报错 (虽然上面我们已经取了全集，加上这个是双保险)
            )
            
            writer.writeheader()
            writer.writerows(data_list)
            
        logging.info(f"✅ 成功写入: {os.path.abspath(full_path)} (共 {len(data_list)} 条)")
        return True

    except PermissionError:
        logging.error(f"❌ 写入失败：文件 '{filename}' 正被其他程序(如Excel)占用，请关闭后重试。")
        return False
    except OSError as e:
        logging.error(f"❌ 操作系统错误 (路径或权限问题): {e}")
        return False
    except Exception as e:
        logging.error(f"❌ 未知错误: {e}")
        return False