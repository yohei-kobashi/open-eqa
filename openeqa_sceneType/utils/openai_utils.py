# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import base64
import os
from typing import List, Optional, Literal

import cv2
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential


def set_openai_key(key: Optional[str] = None):
    if key is None:
        assert "OPENAI_API_KEY" in os.environ
        key = os.environ["OPENAI_API_KEY"]
    openai.api_key = key

def set_deepinfra_key(key: Optional[str] = None):
    if key is None:
        assert "DEEPINFRA_API_KEY" in os.environ
        key = os.environ["DEEPINFRA_API_KEY"]
    openai.api_key = key
    openai.base_url = "https://api.deepinfra.com/v1/openai"

def set_together_key(key: Optional[str] = None):
    if key is None:
        assert "TOGETHER_API_KEY" in os.environ
        key = os.environ["TOGETHER_API_KEY"]
    openai.api_key = key
    openai.base_url = "https://api.together.xyz/v1"

def prepare_openai_messages(content: str, reasoning: bool = False):
    messages = []
    messages.append({"role": "user", "content": content})
    return messages


def prepare_openai_vision_messages(
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
                "image_url": {"url": f"data:image/png;base64,{frame}"},
                "type": "image_url",
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


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_openai_api(
    messages: list,
    model: str = "gpt-4o",
    seed: Optional[int] = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
):
    client = openai.OpenAI(api_key=openai.api_key, base_url=openai.base_url)
    params = {"model": model, "messages": messages, "seed": seed}
    if model != "o1" and model != "o3-mini":
        params.update({"max_tokens": max_tokens, "temperature": temperature})
    completion = client.chat.completions.create(**params)
    if verbose:
        print("openai api response: {}".format(completion))
    assert len(completion.choices) == 1
    return completion.choices[0].message.content


if __name__ == "__main__":
    set_openai_key(key=None)

    messages = prepare_openai_messages("What color are apples?")
    print("input:", messages)

    model = "gpt-4o"
    output = call_openai_api(messages, model=model, max_tokens=512, temperature=1.0)
    print("output: {}".format(output))
