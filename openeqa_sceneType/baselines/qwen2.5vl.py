# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional, Literal
import re
import sys
import numpy as np

import tqdm
import cv2
from tenacity import retry, stop_after_attempt, wait_random_exponential

from openeqa_sceneType.utils.scene_types import open_ai_scene_types

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

def prepare_messages(content: str, reasoning: bool = False):
    messages = []
    messages.append({"role": "user", "content": content})
    return messages

def prepare_vision_messages(
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    image_size: Optional[int] = 512,
):
    messages = []
   
    if image_paths is None:
        image_paths = []

    content = []

    for path in image_paths:
        frame = cv2.imread(path)
        if image_size:
            factor = image_size / max(frame.shape[:2])
            frame = cv2.resize(frame, dsize=None, fx=factor, fy=factor)
        _, buffer = cv2.imencode(".png", frame)
        frame = base64.b64encode(buffer).decode("utf-8")
        content.append(
            {
                "image": f"data:image/png;base64,{frame}",
                "type": "image",
            }
        )

    text = []
    if prefix:
        text.append(prefix)
    if suffix:
        text.append(suffix)
    
    text = "\n\n".join(text)

    content.append({"text": text, "type": "text"})
    messages.append({"role": "user", "content": content})

    return messages

DEFAULT_DATA_DIR: Path = Path("./") / "prompts_sceneType"

PROMPT_NAME_TO_PATH = {
    "blind": DEFAULT_DATA_DIR / Path("blind.txt"),
    "blind_not_step_by_step": DEFAULT_DATA_DIR / Path("blind_not_step_by_step.txt"),
    "vision": DEFAULT_DATA_DIR / Path("vision.txt"),
    "vision_not_step_by_step": DEFAULT_DATA_DIR / Path("vision_not_step_by_step.txt"),
    "vision_and_text_prefix": DEFAULT_DATA_DIR / Path("vision_and_text_prefix.txt"),
    "vision_and_text_prefix_not_step_by_step": DEFAULT_DATA_DIR / Path("vision_and_text_prefix_not_step_by_step.txt"),
    "vision_and_text_suffix": DEFAULT_DATA_DIR / Path("vision_and_text_suffix.txt"),
}

def load_prompt(name: str):
    if name not in PROMPT_NAME_TO_PATH:
        raise ValueError("invalid prompt: {}".format(name))
    path = PROMPT_NAME_TO_PATH[name]
    with path.open("r") as f:
        return f.read().strip()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/open-eqa-v0_sceneType.json",
        help="path to EQA dataset (default: data/open-eqa-v0_sceneType.json)",
    )
    parser.add_argument(
        "--num-q-and-a",
        type=int,
        default=14,
        help="number of using questions and answers (default: 14)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="GPT model (default: gpt-4-0613)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="blind",
        help="prompt (default: blind)",
    )
    parser.add_argument(
        "--image-directory",
        type=str,
        default="data/scene_images/",
        help="path image layouts (default: data/scene_images/)",
    )
    parser.add_argument(
        "--num-layouts",
        type=int,
        default=1,
        help="num layouts in gpt4v (default: 1)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="image size (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="gpt seed (default: 1234)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="gpt temperature (default: 0.2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="gpt maximum tokens (default: 4000)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/results",
        help="output directory (default: data/results)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="continue running on API errors (default: false)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only process the first 5 questions",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (
        args.dataset.stem + "-{}-{}-{}.json".format(re.sub(r".*/", "", args.model), args.prompt, args.seed)
    )
    return args


def parse_output(output: str) -> str:
    start_idx = output.find("A:")
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return output[start_idx:].replace("A:", "").strip()
    return output[start_idx:end_idx].replace("A:", "").strip()


def ask_category(
    questions_and_answers: str,
    model,
    processor,
    seed: int = 1234,
    max_tokens: int = 128,
    temperature: float = 0.2,
    prompt: str = "blind",
    force: bool = False,
    image_paths: List = None,
    image_size: int = None,
) -> Optional[str]:
    try:
        if "vision" in prompt:
            suffix = None
            if "text" in prompt:
                if "not_step_by_step" in prompt:
                    prefix = load_prompt("vision_and_text_prefix_not_step_by_step")
                else:
                    prefix = load_prompt("vision_and_text_prefix")
                suffix = load_prompt("vision_and_text_suffix")
                suffix = suffix.format(questions_and_answers=questions_and_answers)            
            else:
                if "not_step_by_step" in prompt:
                    prefix = load_prompt("vision_not_step_by_step")
                else:
                    prefix = load_prompt("vision")

            messages = prepare_vision_messages(
                prefix=prefix, suffix=suffix, image_paths=image_paths, image_size=image_size
            )
        else:
            prompt = load_prompt(prompt)
            messages = prepare_messages(prompt.format(questions_and_answers=questions_and_answers))

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        torch.
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output[0]
    except Exception as e:
        if not force:
            traceback.print_exc()
            raise e


def main(args: argparse.Namespace):
    # load dataset
    dataset = {}
    for item in json.load(args.dataset.open("r")):
        if "sceneType" in item:
            episode_history = item["episode_history"]
            if not episode_history in dataset:
                dataset[episode_history] = {"questions":[], "answers":[], "sceneType": item["sceneType"]}
            dataset[episode_history]["questions"].append(item["question"])
            dataset[episode_history]["answers"].append(item["answer"])
    print("found {:,} episode histories".format(len(dataset)))
    sys.stdout.flush()

    # load results
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
        sys.stdout.flush()
    completed = [item["episode_history"] for item in results]

    # process data
    print("model:{}".format(args.model))
    sys.stdout.flush()
    
    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # default processor
    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    
    freq = {}
    for idx, (episode_history, item) in enumerate(tqdm.tqdm(dataset.items())):
        if args.dry_run and idx >= 5:
            break

        # skip completed questions
        if episode_history in completed:
            continue  # skip existing

        # extract scene paths
        paths = None
        if "vision" in args.prompt:
            # folder = args.layouts_directory / episode_history
            # layouts = sorted(folder.glob("*-rgb.png"))
            # indices = np.round(np.linspace(0, len(layouts) - 1, args.num_layouts)).astype(int)
            # paths = [str(layouts[i]) for i in indices]
            paths = [os.path.join(args.image_directory, f"{episode_history}.png")]

        # get Q&A
        questions_and_answers = []
        for question, answer in zip(item["questions"], item["answers"]):
            questions_and_answers.append(f"Q: {question}\nA: {answer}")
            if len(questions_and_answers) == args.num_q_and_a:
                break
        questions_and_answers = "\n\n".join(questions_and_answers)
        
        output = ask_category(
            questions_and_answers=questions_and_answers,
            model=model,
            processor=processor,
            seed=args.seed,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            prompt=args.prompt,
            force=args.force,
            image_paths=paths,
            image_size=args.image_size,
        )
        lower_output = output.lower()
        if "the category is" in lower_output:
            lower_output = re.sub(".*the category is", "", lower_output, flags=re.MULTILINE | re.DOTALL)
            words = [re.sub("\W+", "", word) for word in lower_output.split()]
            words = [word for word in words if word and word != "or"]
        else:
            words = [re.sub("\W+", "", word) for word in lower_output.split()]
            words = [word for word in words if word and word != "or"]
        estimated_sceneType = None
        for i in range(len(words)):
            for j in range(1, 4):
                if i + j == len(words) + 1:
                    break
                scene_type_key = tuple(words[i:i+j])
                if scene_type_key in open_ai_scene_types:
                    estimated_sceneType = open_ai_scene_types[scene_type_key]
                    break
            if estimated_sceneType:
                break
        
        # count sceneTypes
        if not item["sceneType"] in freq:
            freq[item["sceneType"]] = [0, 0]
        freq[item["sceneType"]][0] += 1
        if item["sceneType"] == estimated_sceneType:
            freq[item["sceneType"]][1] += 1

        # store results
        results.append({
            "episode_history": episode_history,
            "output": output,
            "estimated_sceneType": estimated_sceneType,
            "sceneType": item["sceneType"]
        })
        json.dump(results, args.output_path.open("w"), indent=2)

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))
    sys.stdout.flush()
    
    for k,v in freq.items():
        print(k, v)
    sys.stdout.flush()

if __name__ == "__main__":
    main(parse_args())
