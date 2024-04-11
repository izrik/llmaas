#!/usr/bin/env python3

from pathlib import Path

from flask import Flask, request
from werkzeug.exceptions import InternalServerError, BadRequest
from llama_index import ServiceContext
from llama_index.chat_engine import SimpleChatEngine
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import LlamaCPP
from llama_index.llms.base import ChatMessage

PROJECT_ROOT_PATH: Path = Path(__file__).parents[0]
MODELS_PATH: Path = PROJECT_ROOT_PATH / "models"
MODELS_CACHE_PATH: Path = MODELS_PATH / "cache"

model: str = 'Wizard-Vicuna-7B-Uncensored.Q4_K_M.gguf'
llm_max_new_tokens: int = 256
llm_context_window: int = 3900
embedding_hf_model_name: str = 'BAAI/bge-small-en-v1.5'
temperature: float = 0.1
n_gpu_layers: int = -1

llm = LlamaCPP(
    model_path=str(MODELS_PATH / model),
    temperature=temperature,
    max_new_tokens=llm_max_new_tokens,
    context_window=llm_context_window,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": n_gpu_layers},
    messages_to_prompt=None,
    completion_to_prompt=None,
    verbose=False,
)
embedding_model = HuggingFaceEmbedding(
    model_name=embedding_hf_model_name,
    cache_folder=str(MODELS_CACHE_PATH),
)
service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embedding_model
)

print('LLM ready')

app = Flask(__name__)


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
    """
    try:
        body = request.json
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
        chat_history = messages[:-1]
        message = messages[-1]
        chat_engine = SimpleChatEngine.from_defaults(
            system_prompt=system_prompt,
            service_context=service_context,
        )
        wrapped_response = chat_engine.chat(
            message=message.content if message is not None else "",
            chat_history=chat_history)
        response = wrapped_response.response
        print(f'Response: {response}')
        return {'content': response, 'role': 'assistant'}
    except Exception as e:
        raise InternalServerError(original_exception=e)


if __name__ == '__main__':
    app.run(debug=False)
