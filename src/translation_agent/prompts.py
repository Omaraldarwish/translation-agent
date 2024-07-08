"""
This module contains classes and functions related to prompts used by the translation agent.

Classes:
    Default: Default dictionary class that maps missing keys to the key itself.
    PromptTemplate: Class representing a prompt template with formatted prompt and system message.
    PromptConfig: Class representing a prompt configuration object for a translation agent.

Variables:
    PROMPT_STORE_DIR: Directory path to where pre-defined prompt templates are stored.

    BASE_TRANSLATION_PROMPT: Base translation prompt template.
    BASE_REFLECTION_PROMPT: Base reflection prompt template.
    BASE_IMPROVEMENT_PROMPT: Base improvement prompt template.

    BASE_PROMPT_CONFIG: Base prompt configuration object.

"""

import os
import string
from typing import ClassVar
from typing import Dict
from typing import Optional
from typing import Tuple
from warnings import warn


class Default(dict):
    """Default dictionary class. Maps missing keys to the key itself."""

    def __missing__(self, key):
        return key


class PromptTemplate:
    """Prompt template class."""

    def __init__(
        self,
        prompt: str,
        sysmsg: str,
        placeholder_kwargs: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initializes a prompt template object.

        Args:
            prompt (str): The prompt template string.
            sysmsg (str): The system message template string.
            placeholder_kwargs (Optional[Dict[str,str]]): A dictionary containing keywords to be
                used as for formatting both the prompt and sysmsg strings.

        Returns:
            PromptTemplate: A prompt template object.
        """

        # save the prompt and sysmsg strings
        self._prompt = prompt
        self._sysmsg = sysmsg

        # extract placeholders from the prompt and sysmsg strings
        self._prompt_placeholders = self.get_string_placeholders(prompt)
        self._sysmsg_placeholders = self.get_string_placeholders(sysmsg)

        # filter the placeholder_kwargs
        placeholder_kwargs = placeholder_kwargs or {}
        self.prompt_kwargs = {
            k: v
            for k, v in placeholder_kwargs
            if k in self._prompt_placeholders
        }
        self.sysmsg_kwargs = {
            k: v
            for k, v in placeholder_kwargs
            if k in self._sysmsg_placeholders
        }

    @staticmethod
    def get_string_placeholders(_str: str) -> Tuple[str]:
        """
        Extracts placeholders from a given string template.

        Args:
            _str (str): The input string.

        Returns:
            Tuple[str]: A tuple containing the extracted string parameters.
        """
        formatter = string.Formatter()
        return tuple(
            f for _, f, _, _ in formatter.parse(_str) if f is not None
        )

    @staticmethod
    def placeholders_in_dict(_str: str, _kwargs: dict) -> bool:
        """
        Checks if all string placeholders in the given string `_str` are present as keys
        in the `_kwargs` dictionary.

        Args:
            _str (str): The template-string containing the placeholders to check.
            _kwargs (dict): The dictionary containing the key-value pairs.

        Returns:
            bool: True if all placeholders are present in `_kwargs`, False otherwise.

        Raises:
            ValueError: If `_str` contains placeholders that are not present in `_kwargs`.

        Warnings:
            Issues warning if `_kwargs` contains placeholder keys that are not present in `_str`

        """

        str_set = set(PromptTemplate.get_string_placeholders(_str))
        kwargs_set = set(_kwargs.keys())

        if not str_set.issubset(kwargs_set):
            raise ValueError(
                f"Prompt requires missing parameters: {str_set - kwargs_set}"
            )

        if not kwargs_set.issubset(str_set):
            warn(
                f"String-template has extra parameters: {kwargs_set - str_set}",
                stacklevel=2,
            )
            return False

        return True

    @staticmethod
    def get_formatted(
        to_format, format_kwargs: Optional[dict[str, str]] = None
    ) -> str:
        """
        Formats the given string-tempalte `to_format` using the provided keyword arguments
        `format_kwargs`.

        Args:
            to_format (str): The string to be formatted.
            format_kwargs (Optional[dict[str, str]]): keyword arguments used for formatting.

        Returns:
            str: The formatted string.

        """
        format_kwargs = format_kwargs or {}
        _ = PromptTemplate.placeholders_in_dict(to_format, format_kwargs)

        return to_format.format(**format_kwargs)

    def get_formatted_prompt(self, **kwargs) -> str:
        """
        Returns the formatted prompt using the provided keyword arguments.

        Args:
            **kwargs: keyword arguments used for formatting the prompt.

        Returns:
            str: The formatted prompt string.

        Warns:
            If the provided keyword arguments override the prompt_kwargs, a warning is issued.
        """
        # check if kwargs override self.prompt_kwargs
        _intersect = set(self.prompt_kwargs.keys()).intersection(
            set(kwargs.keys())
        )
        if _intersect:
            warn(
                f"The following template placeholders were overridden: {_intersect}.",
                stacklevel=2,
            )

        _kwargs = self.prompt_kwargs | kwargs
        return self.get_formatted(self.prompt, _kwargs)

    def get_formatted_sysmsg(self, **kwargs) -> str:
        """
        Returns the formatted system message using the provided keyword arguments.

        Args:
            **kwargs: keyword arguments used for formatting the system message.

        Returns:
            str: The formatted system message string.

        Warns:
            If the provided keyword arguments override the sysmsg_kwargs, a warning is issued.
        """
        _intersect = set(self.sysmsg_kwargs.keys()).intersection(
            set(kwargs.keys())
        )
        if _intersect:
            warn(
                f"The following template placeholders were overrided: {_intersect}.",
                stacklevel=2,
            )

        _kwargs = self.sysmsg_kwargs | kwargs
        return self.get_formatted(self.sysmsg, _kwargs)

    @property
    def prompt(self) -> str:
        """Returns the prompt template-string."""
        return self._prompt

    @property
    def sysmsg(self) -> str:
        """Returns the system message template-string."""
        return self._sysmsg

    @property
    def prompt_placehodlers(self) -> tuple:
        """retruns the list of placeholders in the prompt string."""
        return self._prompt_placeholders

    @property
    def sysmsg_params(self) -> tuple:
        """Returns the list of placeholders in the system message string."""
        return self._sysmsg_placeholders


class PromptConfig:
    """Prompt configuration class."""

    # Required prompt placeholders for each prompt type
    REQUIRED_PROMPT_PLACEHOLDERS: ClassVar[Dict[str, tuple]] = {
        "translation": ("source_lang", "target_lang", "source_chunk"),
        "reflection": (
            "source_lang",
            "target_lang",
            "source_chunk",
            "translated_chunk",
        ),
        "improvement": (
            "source_lang",
            "target_lang",
            "source_chunk",
            "translated_chunk",
            "reflection_chunk",
        ),
    }

    def __init__(
        self,
        tranlsation_prompt: PromptTemplate,
        reflection_prompt: PromptTemplate,
        improvement_prompt: PromptTemplate,
    ) -> None:
        """
        Initializes a prompt configuration object for a translation agent.

        Args:
            tranlsation_prompt (PromptTemplate): Translation prompt template.
                must contain the following placeholders in the prompt string:
                    - source_lang
                    - target_lang
                    - source_tagged
                    - source_chunk
            reflection_prompt (PromptTemplate): Reflection prompt template.
                must contain the following placeholders in the prompt string:
                    - source_lang
                    - target_lang
                    - source_tagged
                    - source_chunk
                    - translated_chunk
            improvement_prompt (PromptTemplate): Improvement prompt template.
                must contain the following placeholders in the prompt string:
                    - source_lang
                    - target_lang
                    - source_tagged
                    - source_chunk
                    - translated_chunk
                    - reflection_chunk

        Returns:
            PromptConfig: A prompt configuration object.
        """

        self._translation_prompt = tranlsation_prompt
        self._reflection_prompt = reflection_prompt
        self._improvement_prompt = improvement_prompt

        # Check if the required parameters are present in the prompt templates
        for (
            prompt_type,
            required_placeholders,
        ) in self.REQUIRED_PROMPT_PLACEHOLDERS.items():
            prompt = getattr(self, f"_{prompt_type}_prompt")
            for placeholder in required_placeholders:
                if placeholder not in prompt.prompt_placehodlers:
                    raise ValueError(
                        f"{prompt_type} prompt must contain the placeholder '{placeholder}'."
                    )

    @property
    def translation_prompt(self) -> PromptTemplate:
        """Returns the translation prompt template."""
        return self._translation_prompt

    @property
    def reflection_prompt(self) -> PromptTemplate:
        """Returns the reflection prompt template."""
        return self._reflection_prompt

    @property
    def improvement_prompt(self) -> PromptTemplate:
        """Returns the improvement prompt template."""
        return self._improvement_prompt


# Load the prompt templates from the prompt_store directory
# --------------------------------------------------------------------------------------------------
PROMPT_STORE_DIR = os.path.join(os.path.dirname(__file__), "prompt_store")

with open(
    os.path.join(PROMPT_STORE_DIR, "base_sys_msg.txt"), encoding="utf-8"
) as f:
    _base_sysmsg = f.read()

with open(
    os.path.join(PROMPT_STORE_DIR, "base_translation_prompt.txt"),
    encoding="utf-8",
) as f:
    _base_translation_prompt = f.read()

with open(
    os.path.join(PROMPT_STORE_DIR, "base_reflection_prompt.txt"),
    encoding="utf-8",
) as f:
    _base_reflection_prompt = f.read()

with open(
    os.path.join(PROMPT_STORE_DIR, "base_improvement_prompt.txt"),
    encoding="utf-8",
) as f:
    _base_improvement_prompt = f.read()
# --------------------------------------------------------------------------------------------------

# Initialize the base prompt templates
# --------------------------------------------------------------------------------------------------
BASE_TRANSLATION_PROMPT = PromptTemplate(
    prompt=_base_translation_prompt,
    sysmsg=_base_sysmsg,
)


BASE_REFLECTION_PROMPT = PromptTemplate(
    prompt=_base_reflection_prompt,
    sysmsg=_base_sysmsg,
)

BASE_IMPROVEMENT_PROMPT = PromptTemplate(
    prompt=_base_improvement_prompt,
    sysmsg=_base_sysmsg,
)

BASE_PROMPT_CONFIG = PromptConfig(
    tranlsation_prompt=BASE_TRANSLATION_PROMPT,
    reflection_prompt=BASE_REFLECTION_PROMPT,
    improvement_prompt=BASE_IMPROVEMENT_PROMPT,
)
# --------------------------------------------------------------------------------------------------
