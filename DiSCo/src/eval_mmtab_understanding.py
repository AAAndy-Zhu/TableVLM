import os
import json
from tqdm import tqdm

# transformers 4.57.1


def normalize_model_type(model_type):
    aliases = {
        'qwen3-vl': 'qwen3vl',
        'llava-v1.6': 'llava_v1_6',
        'llava_v1.6': 'llava_v1_6',
    }
    return aliases.get(model_type.lower(), model_type.lower())


def load_model_and_processor(model_name, model_type):
    import torch

    if model_type == 'qwen3vl':
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        ).eval()
        processor = AutoProcessor.from_pretrained(model_name)
    elif model_type == 'gemma3':
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, device_map="auto"
        ).eval()
        processor = AutoProcessor.from_pretrained(model_name)
    elif model_type == 'gemma3n':
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration

        model = Gemma3nForConditionalGeneration.from_pretrained(
            model_name, device_map="auto"
        ).eval()
        processor = AutoProcessor.from_pretrained(model_name)
    elif model_type == 'llava_v1_6':
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

        processor = LlavaNextProcessor.from_pretrained(model_name)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise NotImplementedError(f"Unsupported model_type: {model_type}")
    return model, processor


def generate_one(model, processor, model_type, qs, image_file, max_new_tokens):
    import torch

    if model_type in ['qwen3vl', 'gemma3', 'gemma3n']:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_file,
                    },
                    {"type": "text", "text": qs},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        if model_type == 'gemma3':
            inputs = inputs.to(model.device, dtype=torch.bfloat16)
        else:
            inputs = inputs.to(model.device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        if model_type == 'qwen3vl':
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            return processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        generation = generated_ids[0][input_len:]
        return processor.decode(generation, skip_special_tokens=True)

    if model_type == 'llava_v1_6':
        from PIL import Image

        image = Image.open(image_file)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": qs},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=text, return_tensors="pt")
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    raise NotImplementedError(f"Unsupported model_type: {model_type}")


def main(args):
    model_name = args.model_path
    model_type = normalize_model_type(args.model_type)
    model, processor = load_model_and_processor(model_name, model_type)

    answers_file = args.answers_file
    answers_dir = os.path.dirname(answers_file)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)

    eval_file_path = args.eval_file_path
    img_path = args.img_path

    final_test_data = json.load(open(eval_file_path))

    ans_file = open(answers_file, "w")

    print(model_name)

    for data in tqdm(final_test_data):
        qs = data['input']

        image_file = os.path.join(img_path, data['image_id']) + '.jpg'
        output_text = generate_one(
            model,
            processor,
            model_type,
            qs,
            image_file,
            args.max_new_tokens,
        )

        data['prediction'] = output_text
        ans_file.write(json.dumps(data) + '\n')
        ans_file.flush()
    ans_file.close()

    # 20512

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['qwen3vl', 'qwen3-vl', 'gemma3', 'gemma3n', 'llava_v1_6', 'llava-v1.6', 'llava_v1.6'],
        required=True,
        help='Model type for inference.',
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument('--eval_file_path', type=str, required=True, help='Path to the evaluation data file')
    parser.add_argument('--img_path', type=str, required=True, help='Path to the directory containing images')
    parser.add_argument('--answers_file', type=str, required=True, help='Path to the output file where answers will be saved')
    args = parser.parse_args()

    main(args)
