"""
This module contains classes and functions which defined the interfaces for classes used
to initialize the translation agent.

Classes:
    LLMInterface: Should be implemented to allow the translation agent to interact
        with the language model.
    SplitterInterface: Should be implemented to apply a text splitting strategy to the input text.

"""

from abc import ABC
from abc import abstractmethod
from typing import List


class LLMInterface(ABC):
    """
    Abstract base class for Language Models.

    This class defines the interface that should be implemented by each translation model.
    """

    @abstractmethod
    def get_completion(self, prompt: str, sysmsg: str) -> str:
        """
        Gets the completion for a given prompt and system message.

        Args:
            prompt (str): The user's prompt.
            sysmsg (str): The system message to provide additional context.

        Returns:
            str: The generated completion.
        """

        # get class name
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"{cls_name} must re-implement the get_completion method."
        )

    @abstractmethod
    def num_tokens_in_string(self, input_str: str) -> int:
        """
        Calculates the number of tokens in a given string.

        Args:
            input_str (str): The input string to count tokens from.

        Returns:
            int: The number of tokens in the input string.
        """
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"{cls_name} must re-implement the num_tokens_in_string method."
        )

    @abstractmethod
    def get_max_tokens(self) -> int:
        """
        Returns the maximum number of tokens allowed for this translation agent.

        Returns:
            int: The maximum number of tokens allowed.
        """
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"{cls_name} must re-implement the get_max_tokens method."
        )


class SplitterInterface(ABC):
    """
    Abstract base class for Text Splitters.

    This class defines the interface that should be implemented by each TextSplitter class.
    """

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Splits the given text into a list of strings.

        Args:
            text (str): The text to be split.

        Returns:
            List[str]: A list of strings obtained by splitting the text.
        """

        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f"{cls_name} must re-implement the split_text method."
        )
