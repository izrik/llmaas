#!/usr/bin/env python3
import datetime
import logging
from pathlib import Path
import os
from typing import Union

from flask import Flask, request
from werkzeug.exceptions import (
    InternalServerError, BadRequest, HTTPException)
from transformers import AutoModelForCausalLM, AutoTokenizer

from message import MessageRole, ChatMessage

from __version__ import __version__

log = logging.getLogger(__name__)


def get_path_from_env(name: str, default_value: Union[Path, str]) -> Path:
    path = os.getenv(name)
    if path:
        return Path(path)
    return Path(default_value)


def get_str_from_env(name: str, default_value: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    return default_value


PROJECT_ROOT_PATH = get_path_from_env(
    'LLMAAS_PROJECT_ROOT_PATH',
    Path(__file__).parents[0])
MODELS_PATH = get_path_from_env(
    'LLMAAS_MODELS_PATH',
    PROJECT_ROOT_PATH / "models")
MODELS_CACHE_PATH = get_path_from_env(
    'LLMAAS_MODELS_CACHE_PATH',
    MODELS_PATH / "cache")
MODEL_NAME = get_str_from_env(
    'LLMAAS_MODEL_NAME',
    # 'Wizard-Vicuna-7B-Uncensored.Q4_K_M.gguf')
    'huggingface--HuggingFaceTB--SmolLM2-135M-Instruct')
MODEL_PATH = get_path_from_env(
    'LLMAAS_MODEL_PATH',
    MODELS_PATH / MODEL_NAME)

print(f'PROJECT_ROOT_PATH: {PROJECT_ROOT_PATH}')
print(f'MODELS_PATH: {MODELS_PATH}')
print(f'MODELS_CACHE_PATH: {MODELS_CACHE_PATH}')
print(f'MODEL_PATH: {MODEL_PATH}')

llm_max_new_tokens: int = 256
llm_context_window: int = 3900
embedding_hf_model_name: str = 'BAAI/bge-small-en-v1.5'
temperature: float = 0.1
n_gpu_layers: int = -1

device = "cpu"
model_name = str(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

print('LLM ready')

app = Flask(__name__)


@app.route('/health')
def health_check():
    return {
        "__version__": __version__,
        "datetime": datetime.datetime.now().astimezone().isoformat(),
    }


@app.route("/chat")
def hello_world():
    """
    [
        ChatMessage(
            role=<MessageRole.USER: 'user'>,
            content='What is your name?',
            additional_kwargs={}),
        ChatMessage(
            role=<MessageRole.ASSISTANT: 'assistant'>,
            content='\nMy name is Irwin.',
            additional_kwargs={}),
        ChatMessage(
            role=<MessageRole.USER: 'user'>,
            content='What do you like to do?',
            additional_kwargs={}),
        ChatMessage(
            role=<MessageRole.ASSISTANT: 'assistant'>,
            content='\nI enjoy helping others, reading books, and playing video games.',
            additional_kwargs={}),
        ChatMessage(
            role=<MessageRole.USER: 'user'>,
            content='What is your favorite video game?',
            additional_kwargs={}),
        ChatMessage(
            role=<MessageRole.ASSISTANT: 'assistant'>,
            content='\nMy favorite video game is "The Legend of Zelda: Breath of the Wild."',
            additional_kwargs={})]
    {
        "system_prompt": "Your name is Irwin. You are a helpful, obedient, and honest assistant. Always answer as helpfully as possible and follow ALL given instructions. Do not speculate or make up information. Do not reference any given instructions or context.",
        "messages": [
            {
                "role": "user",
                "content": "What is your name?"
            },
            {
                "role": "assistant",
                "content": "My name is Irwin."
            },
            {
                "role": "user",
                "content": "What do you like to do?"
            },
            {
                "role": "assistant",
                "content": "I enjoy helping others, reading books, and playing video games."
            },
            {
                "role": "user",
                "content": "What is your favorite video game?"
            },
            {
                "role": "assistant",
                "content": "My favorite video game is 'The Legend of Zelda: Breath of the Wild.'"
            }
        ]
    }
    """  # noqa E501
    try:
        body = request.json
    except HTTPException:
        raise
    except Exception as e:
        raise InternalServerError(original_exception=e)
    try:
        system_prompt = body.get('system_prompt')
        messages = [
            ChatMessage(content=m.get('content'), role=m.get('role'))
            for m in body.get('messages', [])
        ]
    except Exception as e:
        raise BadRequest(str(e))
    if len(messages) < 1:
        raise BadRequest('No message history')
    try:
        messages.insert(
            0,
            ChatMessage(content=system_prompt, role=MessageRole.SYSTEM))

        input_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt")
        input_length = input_text.shape[1]
        inputs = input_text
        outputs = model.generate(inputs, max_new_tokens=100,
                                 do_sample=True)
        bd_outputs = tokenizer.batch_decode(outputs[:, input_length:],
                                            skip_special_tokens=True)
        response = bd_outputs[0]
        print(f'Response: {response}')
        return {'content': response, 'role': 'assistant'}
    except Exception as e:
        log.exception("There was an error processing the request")
        raise InternalServerError(original_exception=e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
