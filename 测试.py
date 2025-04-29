from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig, AutoModel, BitsAndBytesConfig,DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from transformers.generation.utils import GenerationConfig
import json
import openpyxl
from openpyxl import load_workbook
import openai
import random
import time
import torch
import sys
import traceback
import os
import pandas as pd
from tqdm import tqdm
import re

def predict(user_content, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512,temperature=0.2)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    return response

# qwen2.5-72b
openai.api_key = 'sk-GB5jsBV0Gyetp5tJA76c8a5f91794295A07a1bAf8c68888e'
openai.api_base = 'http://10.8.4.23:36435/v1'

product_PROMPT_TEMPLATE = """
汽车产品力维度是一种对汽车的评价体系分类，其包含[操控/动力性能/品质/便利性/空间/舒适性/智能座舱/智能驾驶/NVH/气味性/安全性/续航/经济性/性价比/品牌/感知质量/配置/改装/场景化功能/声音品质]共20个分类。
操控定义：
方向盘操控感受、刹车点头、急刹等，如
1）操控感
2）方向盘：转向手感
3）刹车：点头/软硬
4）变速箱换挡平顺性
5）轮胎表现：抓地力
6）转弯半径；
动力性能定义：
1. 动力加速性能；
品质定义：
1.品质缺陷类问题，比如起火，故障，失灵，断了，坏了等
座椅按摩、加热等功能故障
2.发烫：如娱乐屏发烫、流媒体后视镜发烫；
便利性定义：
1. 若吐槽车身太大，不方便停车，转弯等，先放便利性
2.有这个配置带来了便利性体验，暂时先放便利性，如有电动踏板，上下车很方便，就放便利性
空间定义：
1. 车内空间
2. 车身尺寸
3. 储物相关全放空间，包括前备箱空间大小，好不好用、储物数量、包括杯托个数，杯托好不好用等
4. 乘坐空间
舒适性定义：
环境、动态驾驶、乘坐等带来的舒适性感受。如：
1）座椅舒适性：座椅宽、窄、软、硬，靠背过直等
2）隐私：如一些隐私玻璃
3）遮阳帘：如有遮阳帘，可以防晒；遮阳帘太薄，不防晒
3）乘坐舒适性，如有用户说坐起来很舒服，不晕车
4）空调等
5）方向盘/座椅，暴晒后过热或冬天过冷冰等
6）加热不够热，空调制冷不够冷、
7）悬挂减震，如过坎不颠簸等
8）动能回收
9）底盘调校：软硬
智能座舱定义：
1.座舱智能化设备，包括外观与交互等，语音、界面逻辑
2. 流媒体按缺陷类型分（如黑边宽放PQ，发热放品质，不清晰放智能座舱，流媒体广角不够广，放安全）
3. HUD
智能驾驶定义：
智驾相关的，如雷达不灵敏、自动泊车,NOA、高速领航等
NVH定义：
1.风噪、胎噪，驾驶过程中有的各种声音
2.车内漏风
3.各种异响
4.静谧性：车内隔音
气味性定义：
空调发出糊味、塑料味、香氛的味道
安全性定义：
1. 碰撞安全、钢性底盘等
2. 视野类：如三角窗不挡视野；后视镜视野小；流媒体广角不够广
3. 车灯不够亮
续航定义：
车本身的续航能力
经济性下还有两个产品力维度，分别定义为：
1.能耗。掉电快，电耗，如空调开着特费电
2.油耗。百公里油耗，如豹5百公里14个
3.维护保养。维护保养贵，怕装车后费钱，如像素大灯撞击后，用户觉得维修贵；保险贵
性价比定义：
1.价格、用料带来的高性价比
2.强调配置的丰富度带来的性价比
3. 便宜
品牌定义：
售后服务、门店服务、口碑、保值率、优惠、补贴、发布会效果等、车型定位
感知质量定义：
用户所感知的视听触嗅用五感相关体验，包括造型、异响，触感，美观性，精致感，耐用性，牢固感等。
配置定义：
1. 车漆颜色
2. 内饰颜色
3. 功能类：如缺少方向盘/座椅加热
4. 无法自主选择内外饰颜色搭配
5. 30w车没有电动踏板
6. 有没有前备箱
7. 没有遮阳帘
改装定义：
改装的难易度
场景化功能定义：
联动功能；影院模式；宠物模式；露营模式等跟场景相关的
声音品质定义：
多媒体类：音响效果、引擎模拟音、喇叭音、车机启动音、品牌音
请判断如下关于{brand}用户评论对应的评价对象和产品力维度（需一一对应）、如有多个对象和维度，你需要将其全部都输出：
评论：{comment}

请严格按照以下格式响应，请勿输出分析过程：
{{
    "评价对象": [对象1, ...],
    "产品力维度": [维度1, ...]
}}
"""
PROMPT_TEMPLATE = product_PROMPT_TEMPLATE

if __name__ == '__main__':
    # 加载deepseek-7b-awq
    model_name_or_path ='qwen2-7b-awq' 
    checkpoint_path = 'output\qlora产品力汇总\checkpoint-400'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    # model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
    )

    # 加载零部件数据集
    file_path = 'test.xlsx'  
    df = pd.read_excel(file_path,  sheet_name="Sheet1", engine='openpyxl')
    # filtered_df = df['观点', '评价对象']
    jsons_list = []
    for index, row in df.iterrows():
        user_content = row.get('观点')
        # duixiang = row.get('评价对象')
        chexing = row.get('车型')
        messages = [
            {"role": "system", "content": "你是一个专业的汽车领域评论分析助手"},
            {"role": "user", "content": PROMPT_TEMPLATE.format(brand = chexing,comment=user_content)}
        ]
        try:
            text  = predict(messages, model, tokenizer)
            text = text.replace('“', '"').replace('”', '"')
            matches = re.findall(r'\[(.*?)\]', text)
            extracted_lists = [json.loads(f'[{m}]') if m else [] for m in matches]
            # text = text.replace('：', ':')
            # result = ''
            # match = re.search(r'"产品力":\s*(.+)', text)
            # if match:
            #     result = match.group(1).strip()
            # else:
            #     print("没匹配到")
            json_data = {
                '车型':chexing,
                '评论':user_content,
                '评价对象':extracted_lists[0],
                '产品力维度':extracted_lists[1]
            }
            jsons_list.append(json_data)
            # df['产品力_模型输出'] = result
            # print(result)
        except Exception as e:
            traceback.print_exc()
    # df.to_excel('ModelY_产品力_微调输出.xlsx', index=False) 
    # 指定保存的文件路径
    file_path = "汇总_评价对象_产品力_微调输出.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(jsons_list, f, ensure_ascii=False, indent=4)