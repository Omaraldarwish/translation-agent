"""
An example of an agent that translates text from English to Spanish.
"""

import os

import openai
from translation_agent.agents import SimpleTranslationAgent
from translation_agent.llms import OpenAIModel
from translation_agent.prompts import PROMPT_STORE_DIR
from translation_agent.prompts import PromptConfig
from translation_agent.prompts import PromptTemplate
from translation_agent.splitters import TikTokenMaxTokenSplitter


if __name__ == "__main__":
    # define agent parameters
    # ----------------------------------------------------------------------------------------------
    max_workers = 10
    openai_api_key = os.getenv("OPENAI_API_KEY")

    source_lang, target_lang = "English", "Spanish"

    model_name = "gpt-4-turbo"
    temperature = 0.4
    max_tokens_per_chunk = 100

    sysmsg_path = os.path.join(PROMPT_STORE_DIR, "base_sys_msg.txt")
    translation_prompt_path = os.path.join(
        PROMPT_STORE_DIR, "base_translation_prompt.txt"
    )
    reflection_prompt_path = os.path.join(
        PROMPT_STORE_DIR, "base_reflection_prompt.txt"
    )
    improvement_prompt_path = os.path.join(
        PROMPT_STORE_DIR, "base_improvement_prompt.txt"
    )
    # ----------------------------------------------------------------------------------------------

    # load prompts and init prompt config
    # ----------------------------------------------------------------------------------------------
    with open(sysmsg_path, encoding="utf-8") as f:
        _sysmsg = f.read()

    with open(translation_prompt_path, encoding="utf-8") as f:
        _translation_prompt = f.read()

    with open(reflection_prompt_path, encoding="utf-8") as f:
        _reflection_prompt = f.read()

    with open(improvement_prompt_path, encoding="utf-8") as f:
        _improvement_prompt = f.read()

    translation_prompt = PromptTemplate(
        prompt=_translation_prompt,
        sysmsg=_sysmsg,
    )

    reflection_prompt = PromptTemplate(
        prompt=_reflection_prompt,
        sysmsg=_sysmsg,
    )

    improvement_prompt = PromptTemplate(
        prompt=_improvement_prompt,
        sysmsg=_sysmsg,
    )

    prompt_config = PromptConfig(
        tranlsation_prompt=translation_prompt,
        reflection_prompt=reflection_prompt,
        improvement_prompt=improvement_prompt,
    )
    # ----------------------------------------------------------------------------------------------

    # configure llm
    # ----------------------------------------------------------------------------------------------
    openai_client = openai.OpenAI(api_key=openai_api_key)
    llm = OpenAIModel(
        client=openai_client,
        model_name=model_name,
        temperature=temperature,
    )
    # ----------------------------------------------------------------------------------------------

    # configure splitter
    # ----------------------------------------------------------------------------------------------
    text_splitter = TikTokenMaxTokenSplitter(
        model_name=model_name, max_tokens_per_chunk=max_tokens_per_chunk
    )
    # ----------------------------------------------------------------------------------------------

    # configure agent
    # ----------------------------------------------------------------------------------------------
    agent = SimpleTranslationAgent(
        llm=llm,
        text_splitter=text_splitter,
        prompts=prompt_config,
        source_lang=source_lang,
        target_lang=target_lang,
        verbose=True,
    )
    # ----------------------------------------------------------------------------------------------

    # load source text
    # ----------------------------------------------------------------------------------------------
    relative_path = "sample-texts/sample-long1.txt"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    full_path = os.path.join(script_dir, relative_path)

    with open(full_path, encoding="utf-8") as file:
        source_text = file.read()
    # ----------------------------------------------------------------------------------------------

    # translate
    # ----------------------------------------------------------------------------------------------
    translation = agent.agentic_translate(source_text)

    print(f"Source text:\n\n{source_text}\n------------\n")
    print(f"Translation:\n\n{translation}\n------------\n")
    # ----------------------------------------------------------------------------------------------
