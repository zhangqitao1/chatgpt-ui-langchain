import queue
import threading
import openai
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from langchain import LLMChain, GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory, ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
import json
from django.conf import settings
from langchain.tools import Tool


def index(request):
    return HttpResponse("Hello, world.")


class MyCustomHandler(BaseCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put(token)


def chat_agent(request):
    messages_queue = queue.Queue()
    handler = MyCustomHandler(messages_queue)

    data = json.loads(request.body.decode("utf-8"))
    message = data.get("prompt")
    system = data.get("systemMessage")
    uuid = data.get("uuid")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system),  # The persistent system prompt
        MessagesPlaceholder(variable_name="chat_history"),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template("{human_input}"),  # Where the human input will injectd
    ])

    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        streaming=True,
        callbacks=[handler]
    )

    message_history = RedisChatMessageHistory(
        url=settings.REDIS_URL, ttl=86400 * 30, session_id=uuid
    )

    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=message_history)
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        max_token_limit=200,
        return_messages=True,
        chat_memory=message_history
    )

    gsearch = GoogleSearchAPIWrapper(
        google_cse_id="370b93cffceb44188",
        google_api_key="AIzaSyBCGI0PWIKcadLF76Qcu90t2Sb_Vsj6rqg"
    )

    tools = load_tools(["wikipedia"], llm=llm)
    tools.append(
        Tool(
            name="Google Search",
            func=gsearch.run,
            description="useful for when you need to answer questions about current events",
        )
    )
    agent_chain = initialize_agent(
        tools,
        llm,
        prompt=prompt,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory
    )

    done = threading.Event()

    def run_llm_chain():
        try:
            agent_chain.run(message)
        finally:
            done.set()

    threading.Thread(target=run_llm_chain).start()

    def message_generator():
        result = ""
        while True:
            try:
                msg = messages_queue.get(timeout=1)
                result += msg
                yield json.dumps(
                    {
                        "role": "assistant", "id": "chatcmpl-7hG3ohZ4VljYAzxSZfjOcJWtBvGSi",
                        "text": result
                    }
                ) + "\n"
            except queue.Empty:
                if done.is_set():
                    break

    response = StreamingHttpResponse(message_generator())
    return response


def chat_process(request):
    messages_queue = queue.Queue()
    handler = MyCustomHandler(messages_queue)

    data = json.loads(request.body.decode("utf-8"))
    message = data.get("prompt")
    system = data.get("systemMessage")
    uuid = data.get("uuid")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system),  # The persistent system prompt
        MessagesPlaceholder(variable_name="chat_history"),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template("{human_input}"),  # Where the human input will injectd
    ])

    llm = ChatOpenAI(
        openai_api_key=settings.OPENAI_API_KEY,
        model_name="gpt-4",
        streaming=True,
        callbacks=[handler]
    )

    message_history = RedisChatMessageHistory(
        url=settings.REDIS_URL, ttl=86400 * 30, session_id=uuid
    )

    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=message_history)
    memory = ConversationSummaryBufferMemory(
        llm=ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY
        ),
        memory_key="chat_history",
        max_token_limit=200,
        return_messages=True,
        chat_memory=message_history
    )

    chat_llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True,
    )

    done = threading.Event()

    def run_llm_chain():
        try:
            chat_llm_chain.predict(human_input=message)
        finally:
            done.set()

    threading.Thread(target=run_llm_chain).start()

    def message_generator():
        result = ""
        while True:
            try:
                msg = messages_queue.get(timeout=1)
                result += msg
                yield json.dumps(
                    {
                        "role": "assistant", "id": "chatcmpl-7hG3ohZ4VljYAzxSZfjOcJWtBvGSi",
                        "text": result
                    }
                ) + "\n"
            except queue.Empty:
                if done.is_set():
                    break

    response = StreamingHttpResponse(message_generator())
    return response


def chat_openai(request):
    messages = [{
        "role": 'user',
        "content": 'hello'
    }]
    openai.api_key = settings.OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=1,
        stream=True
    )

    def stream():
        result = ""
        for event in response:
            result += event['choices'][0]['delta'].get('content', '')
            yield json.dumps(
                {
                    "role": "assistant", "id": "chatcmpl-7hG3ohZ4VljYAzxSZfjOcJWtBvGSi",
                    "text": result
                }
            ) + "\n"

    r = StreamingHttpResponse(streaming_content=stream(), content_type='application/octet-stream')
    r['Cache-Control'] = 'no-cache'
    return r


def session(request):
    return JsonResponse(
        {
            "status": "Success",
            "message": "",
            "data": {"auth": False, "model": "ChatGPTAPI"}
        }
    )


def config(request):
    return JsonResponse(
        {
            "status": "Success",
            "message": "",
            "data": {
                "apiModel": "ChatGPTAPI",
                "reverseProxy": "-",
                "timeoutMs": 300000,
                "socksProxy": "-",
                "httpsProxy": "-",
                "usage": "-"
            },

        }
    )
