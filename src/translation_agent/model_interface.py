from abc import ABC
from abc import abstractmethod


class TranslationModelInterface(ABC):
    """Interface for translation model classes."""

    @abstractmethod
    def get_completion(self, prompt: str, system_message: str) -> str:
        """
        Returns the completion of a prompt, accepts a system message.
        This method should be implemented by each translation model.

        Args:
            prompt (str): The user's prompt or query.

        Returns:
            str: The completion of the prompt.

        """
        raise NotImplementedError(
            "Your model must re-implement the get_completion method."
        )

    @abstractmethod
    def num_tokens_in_string(self, input_str: str) -> int:
        """
        Gets the number of tokens in a string.
        This method should be implemented by each translation model.

        Args:
            input_str (str): The input string.

        Returns:
            int: The number of tokens in the input string.
        """
        raise NotImplementedError(
            "Your model must re-implement the num_tokens_in_string method."
        )
