"""
This module contains example implementations of the SplitterInterface for different text splitters.

Classes:
    UniformTextSplitter: Splits text into equal parts based on the number of chunks.
    NewLineSplitter: Splits text by newline character.
    SentenceSplitter: Splits text into sentences.
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tiktoken import encoding_for_model

from .interfaces import SplitterInterface


class UniformTextSplitter(SplitterInterface):
    """Splits text into equal parts based on the number of chunks."""

    def __init__(self, num_chunks: int) -> None:
        """
        Initializes the UniformTextSplitter.

        Args:
            num_chunks (int): The number of chunks to split the text into.
        """
        self.num_chunks = num_chunks

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into equal parts

        Args"
            text (str): The text to split.
        """
        chunk_size = len(text) // self.num_chunks
        return [
            text[i : i + chunk_size] for i in range(0, len(text), chunk_size)
        ]


class NewLineSplitter(SplitterInterface):
    """Splits text by newline character."""

    def split_text(self, text: str) -> List[str]:
        """splits text by newline character"""
        return text.split("\n")


class SentenceSplitter(SplitterInterface):
    """Splits text into sentences."""

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into sentences.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: A list of sentences.
        """
        return text.split(".")


class TikTokenMaxTokenSplitter(SplitterInterface):
    """Splits text into chunks based on TikToken encoding token counts and max tokens per chunk."""

    def __init__(self, model_name, max_tokens_per_chunk) -> None:
        """
        Initializes the TikTokenEncoderSplitter.

        Args:
            encoding (tiktoken.Encoding): The TikToken encoding to use.
        """
        self.model_name = model_name
        self.max_tokens_per_chunk = max_tokens_per_chunk

    def calculate_chunk_size(self, token_count, token_limit):
        """
        Calculate the chunk size based on the token count and token limit.

        Args:
            token_count (int): The total number of tokens.
            token_limit (int): The maximum number of tokens allowed per chunk.

        Returns:
            int: The calculated chunk size.

        Description:
            This function calculates the chunk size based on the given token count and token limit.
            If the token count is less than or equal to the token limit, the function returns the
            token count as the chunk size. Otherwise, it calculates the number of chunks needed to
            accommodate all the tokens within the token limit. The chunk size is determined by
            dividing the token limit by the number of chunks. If there are remaining tokens after
            dividing the token count by the token limit, the chunk size is adjusted by adding the
            remaining tokens divided by the number of chunks.

        Example:
            >>> calculate_chunk_size(1000, 500)
            500
            >>> calculate_chunk_size(1530, 500)
            389
            >>> calculate_chunk_size(2242, 500)
            496
        """

        if token_count <= token_limit:
            return token_count

        num_chunks = (token_count + token_limit - 1) // token_limit
        chunk_size = token_count // num_chunks

        remaining_tokens = token_count % token_limit
        if remaining_tokens > 0:
            chunk_size += remaining_tokens // num_chunks

        return chunk_size

    def num_tokens_in_text(self, text: str) -> int:
        """
        Calculate the number of tokens in a given string using a specified encoding.

        Args:
            str (str): The input string to be tokenized.

        Returns:
            int: The number of tokens in the input string.

        Example:
            >>> text = "Hello, how are you?"
            >>> num_tokens = num_tokens_in_string(text)
            >>> print(num_tokens)
            5
        """
        encoding = encoding_for_model(self.model_name)
        return len(encoding.encode(text))

    def split_text(self, text: str) -> List[str]:
        num_tokens_in_string = self.num_tokens_in_text(text)

        chunk_size = self.calculate_chunk_size(
            num_tokens_in_string, self.max_tokens_per_chunk
        )

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=chunk_size,
            chunk_overlap=0,
        )

        return text_splitter.split_text(text)
