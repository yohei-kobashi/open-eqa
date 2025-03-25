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

import tqdm

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
        "--prompt",
        type=str,
        default="blind",
        help="prompt (default: blind)",
    )
    parser.add_argument(
        "--num-q-and-a",
        type=int,
        default=14,
        help="number of using questions and answers (default: 14)",
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
        "--output-directory",
        type=Path,
        default="data/results",
        help="output directory (default: data/results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only process the first 5 questions",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (
        args.dataset.stem + "-human-{}.json".format(args.prompt)
    )
    return args


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

    # load results
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
        sys.stdout.flush()
    completed = [item["episode_history"] for item in results]
        
    freq = {}
    for idx, (episode_history, item) in enumerate(tqdm.tqdm(dataset.items())):
        if args.dry_run and idx >= 5:
            break

        # skip completed questions
        if episode_history in completed:
            continue  # skip existing

        print("--------------------------------------------------")
        if "blind" in args.prompt:
            # get Q&A
            questions_and_answers = []
            for question, answer in zip(item["questions"], item["answers"]):
                questions_and_answers.append(f"Q: {question}\nA: {answer}")
                if len(questions_and_answers) == args.num_q_and_a:
                    break
            questions_and_answers = "\n\n".join(questions_and_answers)
            print(questions_and_answers)
        if "vision" in args.prompt:
            print("img:", os.path.join(args.image_directory, f"{episode_history}.png"))
            
        print(["Apartment", "Bathroom", "Bedroom / Hotel", "Bookstore / Library", "Classroom", "Conference Room", "Copy / Mail Room", "Kitchen", "Laundry Room", "Living room / Lounge", "Lobby", "Office", "Storage / Basement / Garage"])
        output = input("Please input the reasoning...:").strip()
        estimated_sceneType = input("Please input the scene type:").strip()
        
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
    
    for k in ["Apartment", "Bathroom", "Bedroom / Hotel", "Bookstore / Library", "Classroom", "Conference Room", "Copy / Mail Room", "Kitchen", "Laundry Room", "Living room / Lounge", "Lobby", "Office", "Storage / Basement / Garage"]:
        print(k, freq[k])

if __name__ == "__main__":
    main(parse_args())
