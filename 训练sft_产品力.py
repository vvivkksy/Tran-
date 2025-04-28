from datasets import Dataset,load_from_disk
import pandas as pd
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig, AutoModel, BitsAndBytesConfig,DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from transformers.generation.utils import GenerationConfig
import torch.nn as nn
from torchkeras.chat import ChatLLM
from sklearn.model_selection import train_test_split
from datetime import datetime

def standard_pre_process(test_ratio,eval_ratio,version,corpus_path):
    # 训练/验证/测试集jsonl路径
    train_temp_dataset_path = "temp_dataset/train_%s.jsonl"%(version)
    eval_temp_dataset_path = "temp_dataset/eval_%s.jsonl"%(version)
    test_temp_dataset_path = "temp_dataset/test_%s.jsonl"%(version)
    # --
    with open(train_temp_dataset_path, 'w') as f_train, open(eval_temp_dataset_path, 'w') as f_eval, open(test_temp_dataset_path, 'w') as f_test:
        pass 
    # 随机种子
    random_state = 44
    df_full = pd.read_excel(corpus_path, sheet_name='Sheet1', usecols='A:C', names=['topic', 'opinion', 'label'])
    df_full = df_full.sample(frac=1).reset_index(drop=True)
    train_val, df_test = train_test_split(df_full, test_size=test_ratio, random_state=random_state)
    df_train, df_eval = train_test_split(train_val, test_size=eval_ratio/(1-eval_ratio), random_state=random_state)
    train_data_list = df_train.values.tolist()
    eval_data_list = df_eval.values.tolist()
    test_data_list = df_test.values.tolist()
    with open(train_temp_dataset_path, 'a', encoding="utf-8") as f_train, open(eval_temp_dataset_path, 'a', encoding="utf-8") as f_eval, open(test_temp_dataset_path, 'a', encoding="utf-8") as f_test:
        for data in train_data_list:
            item = {
                "topic": data[0],
                "input": data[1],
                "label": data[2],
            }
            f_train.write(json.dumps(item, ensure_ascii=False) + '\n')

        for data in eval_data_list:
            item = {
                "topic": data[0],
                "input": data[1],
                "label": data[2],
            }
            f_eval.write(json.dumps(item, ensure_ascii=False) + '\n')

        for data in test_data_list:
            item = {
                "topic": data[0],
                "input": data[1],
                "label": data[2],
            }
            f_test.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("训练集、验证集、测试集的jsonl文件已写入！")

def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = WQLinear_GEMM
    # cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def process_func(example):
    MAX_LENGTH = 1024
    input_ids, attention_mask, labels = [], [], [] 
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n你是一名汽车舆情分析者，给你一条用户评论和对应车型，请你总结对该车型的评价对象和产品力维度。\n汽车产品力维度是一种对汽车的评价体系分类，其包含[操控/动力性能/品质/便利性/空间/舒适性/智能座舱/智能驾驶/NVH/气味性/安全性/续航/经济性/性价比/品牌/感知质量/配置/改装/场景化功能/声音品质]共20个分类。<|eot_id|><|start_header_id|>user<|end_header_id|>\n{example['topic']}：{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"评价对象：{example['label1']}，产品力维度：{example['label2']}", add_special_tokens=False)
    input_ids = (
        instruction["input_ids"] 
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = (
        [-100] * len(instruction["input_ids"])
        + response["input_ids"]
        + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH: 
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

if __name__ == "__main__":
    # 参数：版本/对应数据路径
    version = '产品力汇总' 
    version_path = 'b_modified.xlsx'
    # 参数：训练/验证/测试集比例
    test_ratio = 0.15
    eval_ratio = 0.15   
    # 训练/验证/测试集读取：excel->jsonl
    # standard_pre_process(test_ratio,eval_ratio,version,corpus_path=version_path)
    # 加载预训练模型
    model_name_or_path ='qwen2-7b-awq'   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True
    )
    torch.cuda.empty_cache() 
    model.gradient_checkpointing_enable()
    # 测试预训练模型
    # model.generation_config = GenerationConfig.from_pretrained(model_name_or_path) 
    # llm = ChatLLM(model,tokenizer)
    # print(llm.chat(messages=llm.build_messages(query='世界上第二高的山峰是哪一座？')))

    # 训练集tokenize
    df = pd.read_json("temp_dataset/train_%s.jsonl"%(version),lines=True)
    ds = Dataset.from_pandas(df)
    disk_data = ds.map(process_func, remove_columns=ds.column_names)
    tokenized_id_path = 'dataset/train-dataset'
    disk_data.save_to_disk(tokenized_id_path)
    tokenized_id = load_from_disk(tokenized_id_path)
    # 验证集tokenize
    df = pd.read_json("temp_dataset/eval_%s.jsonl"%(version),lines=True)
    ds = Dataset.from_pandas(df)
    disk_data = ds.map(process_func, remove_columns=ds.column_names)
    tokenized_id_path_eval = 'dataset/eval-dataset'
    disk_data.save_to_disk(tokenized_id_path_eval)
    tokenized_id_eval = load_from_disk(tokenized_id_path_eval)
    # 设置训练配置
    output_path = 'output/sft%s'%(version)
    args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=200, 
        learning_rate=1e-5,
        save_on_each_node=True,
        gradient_checkpointing=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        eval_dataset=tokenized_id_eval,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train() 