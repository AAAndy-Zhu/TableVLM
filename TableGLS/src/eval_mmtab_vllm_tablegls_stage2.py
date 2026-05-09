import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import json
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
from PIL import Image
import argparse

PROMPT1 = """You are given a table image, a question and a reasoning plan with target rows and columns.
First, evaluate whether the given reasoning plan is correct and sufficient for answering the question. If the plan is incorrect or incomplete, revise it to obtain a correct reasoning plan.
Then, based on the correct reasoning plan, extract the sub-table that is necessary to answer the question.

Output strictly in the following format:
Plan Evaluation: "brief explanation of your judgment"
Sub-table:
Row m Column n: [Content]
...

Reasoning Plan:
{reasoning_plan}

Question: 
{question}
"""

PROMPT2 = """You are given a table image, a task and a reasoning plan with target rows and columns.
First, evaluate whether the given reasoning plan is correct and sufficient for completing the given task. If the plan is incorrect or incomplete, revise it to obtain a correct reasoning plan.
Then, based on the correct reasoning plan, extract the sub-table that is necessary to answer the question.

Output strictly in the following format:
Plan Evaluation: "brief explanation of your judgment"
Sub-table:
Row m Column n: [Content]
...

Reasoning Plan:
{reasoning_plan}

Task: 
{task}
"""

PROMPT1_hal = """You are given a table image, a question, and a reasoning plan with target rows and columns.

First, critically evaluate whether the given reasoning plan is correct and sufficient for answering the question.
- Verify whether the selected rows and columns are relevant and complete.
- Check for potential missing conditions, incorrect assumptions, or irrelevant selections.
If the plan is incorrect or incomplete, revise it to obtain a correct and sufficient reasoning plan.

Then, BEFORE extracting the sub-table, perform an evidence-grounding check:
- Ensure that every selected row and column can be explicitly located in the table image.
- Only extract information that is directly supported by visible table content.
- Do NOT infer, guess, or hallucinate any content that is not clearly present in the table.

If the evidence is insufficient or uncertain, explicitly state the uncertainty instead of making assumptions.

Finally, based only on verified and grounded evidence, extract the minimal sub-table necessary to answer the question.

Output strictly in the following format:
Plan Evaluation: "brief explanation of your judgment, including any corrections"
Evidence Check: "confirm that extracted content is grounded in the table or note any uncertainty"
Sub-table:
Row m Column n: [Content]
...

Reasoning Plan:
{reasoning_plan}

Question:
{question}
"""

PROMPT2_hal = """You are given a table image, a task, and a reasoning plan with target rows and columns.

First, critically evaluate whether the given reasoning plan is correct and sufficient for completing the given task.
- Verify whether the selected rows and columns are relevant and complete.
- Check for potential missing conditions, incorrect assumptions, or irrelevant selections.
If the plan is incorrect or incomplete, revise it to obtain a correct and sufficient reasoning plan.

Then, BEFORE extracting the sub-table, perform an evidence-grounding check:
- Ensure that every selected row and column can be explicitly located in the table image.
- Only extract information that is directly supported by visible table content.
- Do NOT infer, guess, or hallucinate any content that is not clearly present in the table.

If the evidence is insufficient or uncertain, explicitly state the uncertainty instead of making assumptions.

Finally, based only on verified and grounded evidence, extract the minimal sub-table necessary to answer the question.

Output strictly in the following format:
Plan Evaluation: "brief explanation of your judgment, including any corrections"
Evidence Check: "confirm that extracted content is grounded in the table or note any uncertainty"
Sub-table:
Row m Column n: [Content]
...

Reasoning Plan:
{reasoning_plan}

Task: 
{task}
"""


def create_prompt_qwen3_vl(data, img_path, stage1_file, model_type):
    # assert processor is not None, "processor should not be None when using Qwen3-VL template."
    stage1_data = {}
    with open(stage1_file, 'r') as f:
        for line in f:
            d = json.loads(line)
            stage1_data[d['item_id']] = d['prediction']
    
    prompts = []
    for sample in tqdm(data):
        image_file = os.path.join(img_path, sample['image_id']) + '.jpg'
        image = Image.open(image_file).convert("RGB")
        stage1_output = stage1_data[sample['item_id']]

        prediction = stage1_output.replace('```json', '').replace('```', '').strip()
        if prediction.endswith('}'):
            prediction = prediction
        else:
            prediction = "N/A"
        if sample['dataset_name'] in ['AIT-QA', 'TabMCQ']:
            qs = PROMPT1.format(reasoning_plan=prediction, question=sample['original_query'])
        elif sample['dataset_name'] in ['HiTab', 'WTQ', 'TAT-QA']:
            qs = PROMPT1.format(reasoning_plan=prediction, question=sample['original_query'])
        else:
            original_input = sample['input'].replace('e.g.,', 'for example,')
            qs = PROMPT2.format(reasoning_plan=prediction, task=original_input)

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

    stage1_file = args.stage1_file

    prompts = create_prompt_qwen3_vl(final_test_data, img_path, stage1_file, args.model_type)
    print(f"Loaded {len(prompts)} data for generation")
    print(f"Example data: {prompts[:5]}")

    gen_params = [SamplingParams(temperature=float(args.temperature), max_tokens=1024) for _ in range(len(prompts))]
    outputs = llm.generate(prompts, gen_params, use_tqdm=True)

    gen_res = []

    for o in outputs:
        generated_text = o.outputs[0].text
        gen_res.append(generated_text)

    new_res = gen_res

    assert len(new_res) == len(final_test_data), f"Length of results and data should be equal. Got {len(new_res)} and {len(final_test_data)}"

    for i in range(len(final_test_data)):
        record = final_test_data[i]
        record['prediction'] = new_res[i]
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
    parser.add_argument('--stage1_file', type=str, required=True, help='Path to the stage 1 output file')
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
