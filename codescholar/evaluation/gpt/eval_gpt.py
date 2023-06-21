import os
import json
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# Ref: https://platform.openai.com/examples/default-stripe-api
PROMPT_NO_CHAT = """\"\"\"
The {lib} library in Python exposes the following API: {api}
\"\"\"
import {lib} as {alias}
\"\"\"
Write an idiomatic (frequent) usage example for the {api} API.
\"\"\"
"""


PROMPT_NO_CHAT_NO_ALIAS = """\"\"\"
The {lib} library in Python exposes the following API: {api}
\"\"\"
import {lib}
\"\"\"
Write an idiomatic (frequent) usage example for the {api} API.
\"\"\"
"""


PROMPT_CHAT = """
The {lib} library in Python exposes the following API: {api}
Complete the following code snippet to write an idiomatic (frequent) usage example for the {api} API.
```
import {lib} as {alias}
"""


PROMPT_CHAT_NO_ALIAS = """
The {lib} library in Python exposes the following API: {api}
Complete the following code snippet to write an idiomatic (frequent) usage example for the {api} API.
```
import {lib}
"""


def fill_template(template, lib, api, alias):
    """Fill the template with the given library, API, and alias."""
    if alias is None:
        return template.format(lib=lib, api=api)
    else:
        return template.format(lib=lib, api=api, alias=alias)


def encode_question_nochat(template, lib, api, alias):
    """Encode the prompt instructions into a conversation for no chat models."""
    return fill_template(template, lib, api, alias)


def encode_question_chat(template, lib, api, alias):
    """Encode the prompt instructions into a conversation for chat models.
    Reference: https://github.com/ShishirPatil/gorilla/blob/main/eval/get_llm_responses.py
    """
    prompt = fill_template(template, lib, api, alias)

    prompts = [
        {"role": "system", "content": "You are a helpful API assistant who can write idiomatic API usage examples given the API name."},
        {"role": "user", "content": prompt},
    ]
    return prompts


def get_llm_response(model, lib, api, alias):
    if model == "text-davinci-003":
        PROMPT_TEMPLATE = PROMPT_NO_CHAT_NO_ALIAS if alias is None else PROMPT_NO_CHAT

        response = openai.Completion.create(
            model=model,
            prompt=encode_question_nochat(PROMPT_TEMPLATE, lib, api, alias),
            temperature=0,
            max_tokens=250,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=['"""'],
        )
        return response.choices[0].text

    elif model == "gpt-3.5-turbo":
        PROMPT_TEMPLATE = PROMPT_CHAT_NO_ALIAS if alias is None else PROMPT_CHAT

        responses = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=encode_question_chat(PROMPT_TEMPLATE, lib, api, alias),
            temperature=0,
            max_tokens=250,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=['"""', "```"],
        )
        return responses["choices"][0]["message"]["content"]

    else:
        raise ValueError(f"Model {model} currently not supported.")


if __name__ == "__main__":
    # MODEL = "gpt-3.5-turbo"
    MODEL = "text-davinci-003"

    alias_map = {
        "pandas": "pd",
        "numpy": "np",
    }

    with open("../smol-benchmark.json") as f:
        benchmarks = json.load(f)

    for lib in benchmarks:
        for api in benchmarks[lib]:
            print(f"EVALUATING [{lib}] [{api}]")
            print("=====================================")
            response = get_llm_response(model=MODEL, lib=lib, api=api, alias=alias_map[lib])
            print(response)
            print("=====================================")
