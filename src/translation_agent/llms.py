"""
This module contains example implementations of the LLMInterface for different language models.

Classes:
    OpenAIModel: An example of a translation model using OpenAI's API.
"""

from typing import Optional

import tiktoken
from openai import OpenAI

from .interfaces import LLMInterface


class OpenAIModel(LLMInterface):
    """An example of a translation model using OpenAI's API."""

    def __init__(
        self,
        client: OpenAI,
        model_name: Optional[str] = "gpt-4-turbo",
        temperature: int = 0.3,
    ) -> None:
        """
        Initializes the OpenAI translation model.

        Args:
            client (openai.Client): The OpenAI API client.
            model_name (str): The name of the OpenAI model to use.
            temperature (float): The temperature parameter for the model.
        """
        self.client = client
        self.model_name = model_name
        self.temperature = temperature

    def get_completion(self, prompt: str, sysmsg: Optional[str] = "") -> str:
        """
        Gets the completion for a given prompt and system message.

        Args:
            prompt (str): The user's prompt.
            sysmsg (Optional[str]): The system message to model if needed.

        Returns:
            str: The completion of the prompt.
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            top_p=1,
            messages=[
                {"role": "system", "content": sysmsg},
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content

    def num_tokens_in_string(self, input_str: str) -> int:
        """
        Calculates the number of tokens in a given string.

        Args:
            input_str (str): The input string.

        Returns:
            int: The number of tokens in the input string.
        """
        encoding = tiktoken.encoding_for_model(self.model_name)

        return len(encoding.encode(input_str))

    def get_max_tokens(self) -> int:
        """
        Returns:
            int: The maximum number of tokens allowed by the model.
        """

        return 4096
