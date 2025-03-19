# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import traceback
from typing import Any, List, Optional, Union

import google as genai
from google.genai import types
from PIL.Image import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_google_api(
    message: Union[str, List[Union[Any, Image]]],
    model: str = "gemini-pro",  # gemini-pro, gemini-pro-vision
) -> str:
    try:
        assert "GOOGLE_API_KEY" in os.environ
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        response = client.models.generate_content(
            model=model,
            contents=types.Part.from_text(text='Why is the sky blue?'),
            config=types.GenerateContentConfig(
                temperature=0,
                top_p=0.95,
                top_k=20,
                candidate_count=1,
                seed=5,
                max_output_tokens=100,
                stop_sequences=['STOP!'],
                presence_penalty=0.0,
                frequency_penalty=0.0,
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
