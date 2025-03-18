# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

DEFAULT_DATA_DIR: Path = Path(__file__).parent.parent.parent.resolve() / "prompts_sceneType"

PROMPT_NAME_TO_PATH = {
    "blind": DEFAULT_DATA_DIR / Path("blind.txt"),
    "blind_not_step_by_step": DEFAULT_DATA_DIR / Path("blind_not_step_by_step.txt"),
    "vision": DEFAULT_DATA_DIR / Path("vision.txt"),
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
