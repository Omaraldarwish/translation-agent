import os

import openai
import tiktoken
import translation_agent as ta
from dotenv import load_dotenv


class OpenAITranslationModel(ta.TranslationModelInterface):
    def __init__(
        self, client, system_message=None, model_name=None, temperature=None
    ) -> None:
        """
        Initializes the OpenAI translation model.

        Args:
            client (openai.Client): The OpenAI API client.
            system_message (str): The system message to be displayed to the user.
            model_name (str): The name of the OpenAI model to use.
            temperature (float): The temperature parameter for the model.
        """
        self.client = client

        self.system_message = system_message or "You are a helpful assistant."
        self.model_name = model_name or "gpt-4-turbo"
        self.temperature = temperature or 0.3

    def get_completion(self, prompt: str, system_message: str = "") -> str:
        """
        Implements the get_completion method required by the TranslationModelInterface.

        Args:
            prompt (str): The user's prompt or query.
            system_message (str): The system message to be displayed to the user.

        Returns:
            str: The completion of the prompt.
        """
        _system_message = system_message or self.system_message

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            top_p=1,
            messages=[
                {"role": "system", "content": _system_message},
                {"role": "user", "content": prompt},
            ],
        )

        return response.choices[0].message.content

    def num_tokens_in_string(self, input_str: str) -> int:
        """
        Implements the num_tokens_in_string method required by the TranslationModelInterface.

        Args:
            input_str (str): The input string.

        Returns:
            int: The number of tokens in the input string.
        """
        encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(input_str))


if __name__ == "__main__":
    source_lang, target_lang, country = "English", "Spanish", "Mexico"

    load_dotenv()  # read local .env file

    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    agent_model = OpenAITranslationModel(
        client=openai_client,
        system_message="You are a helpful assistant.",
        model_name="gpt-4-turbo",
        temperature=0.3,
    )

    relative_path = "sample-texts/sample-short1.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    full_path = os.path.join(script_dir, relative_path)

    with open(full_path, encoding="utf-8") as file:
        source_text = file.read()

    print(f"Source text:\n\n{source_text}\n------------\n")

    translation = ta.translate(
        model=agent_model,
        source_lang=source_lang,
        target_lang=target_lang,
        source_text=source_text,
        country=country,
    )

    print(f"Translation:\n\n{translation}")
