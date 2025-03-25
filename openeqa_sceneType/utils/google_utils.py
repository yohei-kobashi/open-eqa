# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import traceback
from typing import Any, List, Optional, Union

import google as genai
from google.genai import types
import PIL.Image
from tenacity import retry, stop_after_attempt, wait_random_exponential

def prepare_google_messages(content: str, reasoning: bool = False):
    messages = []
    messages.append(content)
    return messages

def prepare_google_vision_messages(
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    image_size: Optional[int] = 512,
):
    messages = []
   
    if image_paths is None:
        image_paths = []

    for path in image_paths:
        frame = PIL.Image.open(path)
        if image_size:
            factor = image_size / max(frame.size)  # Pillowは(width, height)なのでmaxを取る
            new_size = (int(frame.width * factor), int(frame.height * factor))
            frame = frame.resize(new_size, PIL.Image.LANCZOS)
        messages.append(frame)

    text = []
    if prefix:
        text.append(prefix)
    if suffix:
        text.append(suffix)
    
    text = "\n\n".join(text)

    messages.append(text)

    return messages


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_google_api(
    messages: List,
    model: str = "gemini-pro",
    seed: Optional[int] = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
) -> str:
    try:
        assert "GOOGLE_API_KEY" in os.environ
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        response = client.models.generate_content(
            model=model,
            contents=messages,
            config=types.GenerateContentConfig(
                temperature=temperature,
                seed=seed,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text
    except Exception as e:
        print(f"{type(e).__name__}: {e}")
        raise e


if __name__ == "__main__":

    input = "What color are apples?"
    print("input: {}".format(input))

    output = call_google_api(input)
    print("output: {}".format(output))
