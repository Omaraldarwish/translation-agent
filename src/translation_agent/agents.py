"""
This module contains example implementations of translation agents. A translation agent is a class
that uses a language model to translate text from one language to another. The typical workflow of
a translation agent involves the following steps:
    - Split the source text into chunks.
    - Translate each chunk using the language model.
    - Reflect on the translation and generate feedback.
    - Improve the translation based on the feedback.
    - Combine the improved chunks into the final translated text.

The translation agent can be customized by providing different prompts for translation, reflection,
and improvement. The prompts are templates that can be formatted with the source text, translated
text, reflection text to generate the input for the language model, and custom default arguments.

Classes:
    - SimpleTranslationAgent: example class for a translation agent.

"""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import List
from typing import Optional
from warnings import warn

from icecream import ic

from .interfaces import LLMInterface
from .interfaces import SplitterInterface
from .prompts import PromptConfig
from .prompts import PromptTemplate


class SimpleTranslationAgent:
    """Base class for a translation agent"""

    def __init__(
        self,
        llm: LLMInterface,
        text_splitter: SplitterInterface,
        prompts: PromptConfig,
        source_lang: str,
        target_lang: str,
        verbose: bool = False,
        source_text_tag: Optional[str] = "TRANSLATE_THIS",
        max_workers: Optional[int] = 10,
    ):
        """
        Initializes the translation agent.

        Args:
            llm (LLMInterface): language model interface instance.
            text_splitter (SplitterInterface): text splitter interface instance.
            prompts (PromptConfig): prompt configuration instance.
            source_lang (str): Source language for translation.
            target_lang (str): Target language for translation.
            verbose (bool, optional): Verbose mode. Defaults to False.
            source_text_tag (str, optional): str to be used for source text in prompt tagging.
                Defaults to 'TRANSLATE_THIS'.
            max_workers (int, optional): Maximum number of workers for parallel processing.

        Raises:
            TypeError: If llm is not an instance of LLMInterface.
            TypeError: If text_splitter is not an instance of SplitterInterface.
            TypeError: If prompts is not an instance of PromptConfig.
        """
        # Check if llm is an instance of LLMInterface
        if not isinstance(llm, LLMInterface):
            raise TypeError("llm must be an instance of LLMInterface.")

        # Check if text_splitter is an instance of SplitterInterface
        if not isinstance(text_splitter, SplitterInterface):
            raise TypeError(
                "text_splitter must be an instance of SplitterInterface."
            )

        # Check if prompts is an instance of PromptConfig
        if not isinstance(prompts, PromptConfig):
            raise TypeError("prompts must be an instance of PromptConfig.")

        # store attributes
        self.llm = llm
        self.text_splitter = text_splitter
        self.prompts = prompts
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.verbose = verbose
        self.source_text_tag = source_text_tag
        self.default_kwargs = {
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
        }
        self.max_workers = max_workers
        # buffer
        self._buffer = None

    @staticmethod
    def tag_chunk(chunk: str, tag: str = "TRANSLATE_THIS") -> str:
        """
        Tags a given chunk of text with a specified tag.

        Args:
            chunk (str): The text chunk to be tagged.
            tag (str, optional): The tag to be used. Defaults to 'TRANSLATE_THIS'.

        Returns:
            str: The tagged text chunk.

        Example:
            >>> tag_chunk("Hello, world!", "TAG")
            '<TAG>Hello, world!</TAG>'
        """
        return f"<{tag}>{chunk}</{tag}>"

    def get_tagged_source_text(self, chunk_idx: int) -> dict:
        """
        Returns the tagged source text for a specific chunk index.

        Args:
            chunk_idx (int): The index of the chunk to tag.

        Returns:
            dict: A dictionary containing the tagged source chunk and the tagged source text.
                - 'source_chunk': The original source chunk at the specified index.
                - 'source_tagged': The tagged source text with the specified chunk tagged.

        Raises:
            ValueError: If the buffer is empty.
            ValueError: If the 'source_chunks' key is not found in the buffer.
        """

        if not self._buffer:
            raise ValueError("Buffer is empty. Run agentic_translate first.")

        if "source_chunks" not in self._buffer:
            raise ValueError(
                "Source chunks not found in buffer. Run agentic_translate first."
            )

        chnks = self._buffer["source_chunks"]
        tagged = "".join(
            chnks[:chunk_idx]
            + [self.tag_chunk(chnks[chunk_idx], self.source_text_tag)]
            + chnks[chunk_idx + 1 :]
        )

        return {"source_chunk": chnks[chunk_idx], "source_tagged": tagged}

    @property
    def buffer(self) -> dict:
        """Returns the buffer used to store intermediate results."""
        return self._buffer or {}

    def query_model(
        self,
        chunk_id: str,
        prompt_template: PromptTemplate,
        prompt_kwargs: Optional[dict] = None,
        sysmsg_kwargs: Optional[dict] = None,
    ) -> str:
        """
        Queries the language model with a formatted prompt and system message.

        Args:
            prompt_template (str): The template for the prompt.
            prompt_kwargs (Optional[str]): Keyword arguments to be used for
                formatting the prompt template.
            sysmsg_kwargs (Optional[str]): Keyword arguments to be used for
                formatting the system message template.

        Returns:
            The completion generated by the language model.

        """
        prompt_kwargs = prompt_kwargs or {}
        sysmsg_kwargs = sysmsg_kwargs or {}

        _prompt = prompt_template.get_formatted_prompt(**prompt_kwargs)

        _sysmsg = prompt_template.get_formatted_sysmsg(**sysmsg_kwargs)

        if self.verbose:
            ic("querying model ...")

        completion = self.llm.get_completion(prompt=_prompt, sysmsg=_sysmsg)

        return chunk_id, completion

    def agentic_translate(
        self, source_text: str, clear_buffer: Optional[bool] = True
    ) -> str:
        """
        Translates the given source text using the agentic translation approach.

        Args:
            source_text (str): The text to be translated.
            clear_buffer (bool, optional): Whether to clear the buffer after translation.
                Defaults to True.

        Returns:
            str: The translated text.
        """
        # clear buffer
        if self._buffer:
            warn("Buffer is not empty. Clearing buffer.", stacklevel=1)
        self._buffer = {}

        # get chunks
        self._buffer["source_chunks"] = self.text_splitter.split_text(
            source_text
        )
        self._buffer["num_chunks"] = len(self._buffer["source_chunks"])

        # warnging if num chunks is zero

        if self._buffer["num_chunks"] == 1:
            warn(
                "Source text is too short. Switching to direct translation.",
                stacklevel=1,
            )

        # translate chunks
        if self.verbose:
            ic(self._buffer["num_chunks"])
            ic("translating chunks ...")

        self._buffer["translated_chunks"] = self.translate()

        # reflect chunks
        if self.verbose:
            ic("reflecting chunks ...")
        self._buffer["reflection_chunks"] = self.reflect()

        # improve chunks
        if self.verbose:
            ic("improving chunks ...")
        self._buffer["improved_chunks"] = improved_chunks = self.improve()

        final = "".join(improved_chunks)

        if clear_buffer:
            self._buffer = None

        return final

    def translate(self) -> List[str]:
        """Executes the translation prompt for each chunk in the buffer."""
        output = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.query_model,
                    chunk_id=idx,
                    prompt_template=self.prompts.translation_prompt,
                    prompt_kwargs={
                        **self.default_kwargs,
                        **self.get_tagged_source_text(idx),
                    },
                    sysmsg_kwargs=self.default_kwargs,
                ): idx
                for idx in range(self._buffer["num_chunks"])
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    output.append(result)
                except Exception as e:
                    # Handle the exception, e.g., logging it
                    print(f"Error reflecting on chunk {idx}: {e}")

        # re-order output based on chunk index
        output = [x[1] for x in sorted(output, key=lambda x: x[0])]

        return output

    def reflect(self) -> List[str]:
        """Executes the reflection prompt for each chunk in the buffer."""
        output = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.query_model,
                    chunk_id=idx,
                    prompt_template=self.prompts.reflection_prompt,
                    prompt_kwargs={
                        "translated_chunk": self._buffer["translated_chunks"][
                            idx
                        ],
                        **self.default_kwargs,
                        **self.get_tagged_source_text(idx),
                    },
                    sysmsg_kwargs=self.default_kwargs,
                ): idx
                for idx in range(self._buffer["num_chunks"])
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    output.append(result)
                except Exception as e:
                    # Handle the exception, e.g., logging it
                    print(f"Error reflecting on chunk {idx}: {e}")

        # re-order output based on chunk index
        output = [x[1] for x in sorted(output, key=lambda x: x[0])]

        return output

    def improve(self) -> List[str]:
        """Executes the improvement prompt for each chunk in the buffer."""
        output = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.query_model,
                    chunk_id=idx,
                    prompt_template=self.prompts.improvement_prompt,
                    prompt_kwargs={
                        "translated_chunk": self._buffer["translated_chunks"][
                            idx
                        ],
                        "reflection_chunk": self._buffer["reflection_chunks"][
                            idx
                        ],
                        **self.default_kwargs,
                        **self.get_tagged_source_text(idx),
                    },
                    sysmsg_kwargs=self.default_kwargs,
                ): idx
                for idx in range(self._buffer["num_chunks"])
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    output.append(result)
                except Exception as e:
                    # Handle the exception, e.g., logging it
                    print(f"Error reflecting on chunk {idx}: {e}")

        # re-order output based on chunk index
        output = [x[1] for x in sorted(output, key=lambda x: x[0])]

        return output
