import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import json
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
from PIL import Image
import argparse

PROMPT1 = """You are given a table image, a question and a sub-table.
First, let's think step by step based on the given information.
Then provide the final concise answer in the JSON format {{\"answer\": \"<YOUR ANSWER>\"}}.

Output in the following format:
Reasoning: "think step by step to answer the question"
{{\"answer\": \"<YOUR ANSWER>\"}}

Sub-table:
{subtable}

Question: 
{question}
"""

PROMPT2 = """You are given a table image, a question and a sub-table.
First, let's think step by step based on the given information.
Then provide the final concise answers in a JSON format {{\"answer\": [<a list of answer strings>]}}.

Output in the following format:
Reasoning: "think step by step to answer the question"
{{\"answer\": [<a list of answer strings>]}}

Sub-table:
{subtable}

Question: 
{question}
"""


PROMPT3 = """You are given a table image, a task and a sub-table.
First, let's think step by step based on the given information.
Then provide the final concise answer in the JSON format {{\"answer\": \"<YOUR ANSWER>\"}}.

Output in the following format:
Reasoning: "think step by step to complete the task"
{{\"answer\": \"<YOUR ANSWER>\"}}

Sub-table:
{subtable}

Task: 
{task}
"""



def create_prompt_qwen3_vl(data, img_path, stage2_file, model_type):
    stage2_data = {}
    with open(stage2_file, 'r') as f:
        for line in f:
            d = json.loads(line)
            stage2_data[d['item_id']] = d['prediction']
    prompts = []
    for sample in tqdm(data):
        image_file = os.path.join(img_path, sample['image_id']) + '.jpg'
        image = Image.open(image_file).convert("RGB")
        subtable = stage2_data[sample['item_id']].strip().split('Sub-table:')[-1].strip()
        if sample['dataset_name'] in ['AIT-QA']:
            qs = PROMPT1.format(subtable=subtable, question=sample['original_query'])
        elif sample['dataset_name'] in ['TabMCQ']:
            qs = PROMPT1.format(subtable=subtable, question=sample['input'])
        elif sample['dataset_name'] in ['HiTab', 'WTQ', 'TAT-QA']:
            qs = PROMPT2.format(subtable=subtable, question=sample['original_query'])
        else:
            original_input = sample['input']
            qs = PROMPT3.format(subtable=subtable, task=original_input)

        if model_type == 'qwen3vl':
            chat_prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{qs}<|im_end|>\n<|im_start|>assistant\n"
        elif model_type == 'gemma3n':
            chat_prompt = f"<start_of_turn>user\n<image_soft_token>{qs}<end_of_turn>\n<start_of_turn>model\n"

        prompts.append(
            {
                "prompt": chat_prompt,
                "multi_modal_data": {"image": image}
            }
        )
    return prompts

def main(llm, args):
    eval_file_path = args.eval_file_path
    img_path = args.img_path

    answers_file = args.answers_file
    answers_dir = os.path.dirname(answers_file)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)

    final_test_data = json.load(open(eval_file_path))
    ans_file = open(answers_file, "w")

    stage2_file = args.stage2_file

    prompts = create_prompt_qwen3_vl(final_test_data, img_path, stage2_file, args.model_type)
    print(f"Loaded {len(prompts)} data for generation")
    print(f"Example data: {prompts[:5]}")

    gen_params = [SamplingParams(temperature=float(args.temperature), max_tokens=4096) for _ in range(len(prompts))]
    outputs = llm.generate(prompts, gen_params, use_tqdm=True)

    gen_res = []

    for o in outputs:
        generated_text = o.outputs[0].text
        gen_res.append(generated_text)

    new_res = gen_res

    assert len(new_res) == len(final_test_data) == len(prompts), f"Length of results and data should be equal. Got {len(new_res)} and {len(final_test_data)}"

    for i in range(len(final_test_data)):
        record = final_test_data[i]
        record['prediction'] = new_res[i]
        record['prompt'] = prompts[i]['prompt']
        ans_file.write(json.dumps(record) + '\n')
        ans_file.flush()
    ans_file.close()

    print("Results saved to:", answers_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--temperature', type=str, default='0')
    parser.add_argument('--eval_file_path', type=str, required=True, help='Path to the evaluation data file')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the directory containing images')
    parser.add_argument('--answers_file', type=str, required=True, help='Path to the output file where answers will be saved')
    parser.add_argument('--stage2_file', type=str, required=True, help='Path to the stage 2 output file')
    parser.add_argument('--model_type', type=str, choices=['qwen3vl', 'gemma3n'], required=True, help='model type')

    args = parser.parse_args()

    print("Loading model from:", args.model_path)

    world_size = torch.cuda.device_count()
    if args.model_type == 'qwen3vl':
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=world_size,
            gpu_memory_utilization=0.6,
            max_model_len=86256
        )
    elif args.model_type == 'gemma3n':
        llm = LLM(
            model=args.model_path,
            max_model_len=32768,
            enforce_eager=True,
            tensor_parallel_size=world_size,
        )
    else:
        raise NotImplementedError('Not Implemented Model Type.')

    main(llm, args)
