# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional, Literal

import numpy as np
import tqdm
import re

from openeqa_sceneType.utils.openai_utils import (
    call_openai_api,
    prepare_openai_vision_messages,
    set_openai_key,
)
from openeqa_sceneType.utils.prompt_utils import load_prompt
from openeqa_sceneType.utils.scene_types import open_ai_scene_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/open-eqa-v0_sceneType.json",
        help="path to EQA dataset (default: data/open-eqa-v0_sceneType.json)",
    )
    parser.add_argument(
        "--datatype",
        type=str,
        default="vision",
        help="vision or vision_and_text (default: vision)",
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
        "--frames-directory",
        type=Path,
        default="data/frames/",
        help="path image frames (default: data/frames/)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=50,
        help="num frames in gpt4v (default: 50)",
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
        default=128,
        help="gpt maximum tokens (default: 128)",
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
        args.dataset.stem + "-{}-{}-{}.json".format(args.model, args.seed, args.datatype)
    )
    return args


def ask_category(
    image_paths: List,
    image_size: int = 512,
    questions_and_answers: str = None,
    openai_key: Optional[str] = None,
    openai_model: Literal["gpt-4o", "gpt-4o-mini"] = "gpt-4o",
    openai_seed: int = 1234,
    openai_max_tokens: int = 128,
    openai_temperature: float = 0.2,
    force: bool = False,
) -> Optional[str]:
    try:
        set_openai_key(key=openai_key)

        suffix = None
        if questions_and_answers:
            prefix = load_prompt("vision_and_text_prefix")
            suffix = load_prompt("vision_and_text_suffix")
            suffix = suffix.format(questions_and_answers=questions_and_answers)            
        else:
            prefix = load_prompt("vision")

        messages = prepare_openai_vision_messages(
            prefix=prefix, suffix=suffix, image_paths=image_paths, image_size=image_size
        )
        output = call_openai_api(
            messages=messages,
            model=openai_model,
            seed=openai_seed,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
        )
        return output
    except Exception as e:
        if not force:
            traceback.print_exc()
            raise e


def main(args: argparse.Namespace):
    # check for openai api key
    assert "OPENAI_API_KEY" in os.environ

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

    # load results
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["episode_history"] for item in results]

    # process data
    freq = {}
    for idx, (episode_history, item) in enumerate(tqdm.tqdm(dataset.items())):
        if args.dry_run and idx >= 5:
            break

        # skip completed questions
        if episode_history in completed:
            continue  # skip existing

        # extract scene paths
        folder = args.frames_directory / episode_history
        frames = sorted(folder.glob("*-rgb.png"))
        indices = np.round(np.linspace(0, len(frames) - 1, args.num_frames)).astype(int)
        paths = [str(frames[i]) for i in indices]

        # generate answer
        questions_and_answers = None
        if args.datatype == "vision_and_text":
            questions_and_answers = []
            for i,(question, answer) in enumerate(zip(item["questions"], item["answers"])):
                questions_and_answers.append(f"Q: {question}\nA: {answer}")
                if len(questions_and_answers) == args.num_q_and_a:
                    break
            questions_and_answers = "\n\n".join(questions_and_answers)
            
        output = ask_category(
            image_paths=paths,
            image_size=args.image_size,
            questions_and_answers=questions_and_answers,
            openai_model=args.model,
            openai_seed=args.seed,
            openai_max_tokens=args.max_tokens,
            openai_temperature=args.temperature,
            force=args.force,
        )
        estimated_sceneType = None
        for word in output.split():
            word = re.sub("\W+", "", word)
            if word.lower() in open_ai_scene_types:
                estimated_sceneType = open_ai_scene_types[word.lower()]
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
    
    for k,v in freq.items():
        print(k, v)

if __name__ == "__main__":
    main(parse_args())
