from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.graph.message import AnyMessage, add_messages
from langchain_gigachat import GigaChat
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.types import Command
from typing import List, Dict
from tavily import TavilyClient
import time

class GraphsState(MessagesState):
    last_reason: str = ""
    user_question: str = ""
    last_answer: str = ""
    critique: List[str] = []
    final_decision: str = ""
    final_answer: str = ""
    search_query: str = ""
    search_mode: str = ""
    search_results: Dict = {}

graph = StateGraph(GraphsState)

llm = GigaChat(model="GigaChat-2-Max",
                verify_ssl_certs=False,
                profanity_check=False,
                # base_url="https://gigachat.sberdevices.ru/v1",
                streaming=True,
                max_tokens=8000,
                top_p=0,
                # temperature=1,
                timeout=600)

MAIN_TEMPLATE = f"""Ты - ИИ Ассистент на базе GigaChat.
Твоя задача качественно ответить на вопрос пользователя.
Сегодняшняя дата - {time.strftime('%Y-%m-%d')}
Рассуждай шаг за шагом. Подумай о том, как ответить на вопрос пользователя наилучшим образом.

"""

REASONER_TEMPLATE = MAIN_TEMPLATE + """Думай как аналитик, который обдумывает вопрос пользователя перед тем, как начать отвечать на него.
Напиши свои мысли о том, как ответить на вопрос пользователя.
Подумай, почему пользователь задает такой вопрос.
Как ты можешь ответить на него наилучшим образом?
Что именно нужно пользователю? Этот ли вопрос он на самом деле хотел задать? Напиши свои мысли по этим вопросам.
Учти, что у тебя будет возможность воспользоваться поиском в интернете, так что учитывай это при планировании.
Но не ищи в интернете в ситуации, если ты сам знаешь ответ.
Если тебя спрашивают про какую-то конкретную персону - прими решение искать о ней информацию в интернете.
Сейчас ты должен подумать о том, как ты будешь решать эту задачу, но не принимать конкретных решений о следующийх шагах. Ты вернешься к этому решению позднее.

Вопрос пользователя - {user_question}
"""

def reason(state: GraphsState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", REASONER_TEMPLATE)
    ])

    chain = prompt | llm | StrOutputParser()

    res = chain.invoke(
        {
            "user_question": state['user_question'],
        }
    )

    return {"last_reason": res}


class FirstStep(BaseModel):
    """Описание первого шага для ответа на вопрос пользователя"""
    search_query: str = Field(description="Текст поискового запроса на поиск данных в интернете, если нужен")
    final_decision: str = Field(description="Итоговое решение, должно быть одно из следующих: ")



FIRST_STEP_TEMPLATE = MAIN_TEMPLATE + """Думай как агент-координатор, который должен опредлить какой первый шаг нужно сделать для ответа на вопрос пользователя.
Ты должен выбрать, какой агент должен продолжить работу над вопросом пользователя.

Вопрос пользователя - {user_question}

Твои мысли по поводу вопроса пользователя:
<THINK>
{last_reason}
</THINK>

Выбрать нужно выбрать следующий шаг - какую функцию вызвать, чтобы ответить на вопрос пользователя наилучшим образом.
1. finalize - все понятно, можно переходить к написанию финального ответа
2. search - требуется поиск данных в интернете
3. writer - требуется написать первую версию подрбоного ответа, который в дальнейшем будет проанализировать критиком.

Выведи только следующую информацию в формате JSON:
{format_instructions}"""


def first_step(state: GraphsState):
    parser = PydanticOutputParser(pydantic_object=FirstStep)
    prompt = ChatPromptTemplate.from_messages([
        ("system", FIRST_STEP_TEMPLATE)
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    res = chain.invoke(
        {
            "user_question": state["user_question"],
            "last_reason": state["last_reason"],
        }
    )

    final_decision = res.final_decision
    search_query = res.search_query
    update = {"final_decision": final_decision, "search_query": search_query}
    goto = "🏁 finalizing"

    if final_decision == "search" and search_query is not None and len(search_query) > 0:
        goto = "🔍 Searcher"
    if final_decision == "writer":
        goto = "👨 answering"

    return Command(
        update=update,
        goto=goto
    )

ANSWER_TEMPLATE = MAIN_TEMPLATE + """Думай как агент-помощник пользователя.
Твоя задача - ответить на вопрос пользователя. Изучи свои предыдущие мысли и ответь пользователю на его вопрос с учетом своих мыслей.
Или, если требуется поиск, напиши, что нужно сначала вопспользоваться поиском, у тебя будет возмоность ответить после.

Вопрос пользователя - {user_question}

Твои мысли по поводу вопроса пользователя:
<THINK>
{last_reason}
</THINK>

Результаты поиска (если есть):
<SEARCH_RESULTS>
{search_results}
</SEARCH_RESULTS>

Теперь ответь пользователю:
"""

def answer(state: GraphsState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_TEMPLATE)
    ])

    chain = prompt | llm | StrOutputParser()

    res = chain.invoke(
        {
            "user_question": state["user_question"],
            "last_reason": state["last_reason"],
            "search_results": state.get("search_results", {})
        }
    )

    return {"last_answer": res}


class Critique(BaseModel):
    """Критика выступления"""

    thoughts: str = Field(description="Мысли по поводу ответа")
    critique: str = Field(description="Конструктивная критика ответа - что нужно поправить или доработать")
    is_new_critique: bool = Field(description="Содержит ли твоя критика что-то принципиально новое или подобная критика уже была дана раньше")
    search_query: str = Field(description="Текст поискового запроса на поиск данных в интернете, если нужен")
    search_mode: str = Field(description="Режим поиска - basic (простой поиск) или deep (глубокий поиск). Используй простой поиск, когда тебе нужно найти ответ на вопрос и глубокий поиск, когда нужно загрузить много подробной информации. Используй глубокий поиск только в случае, когда ты уже попробовал простой поиск.")
    final_decision: str = Field(description="Итоговое решение, должно быть одно из следующих: good (если нет новой критики, есть отрывки из книг и речь можно считать написаной), search (требуется поиск данных в интернете), fix (если требуется переписать или доработать ответ)")



CRITIQUE_TEMPLATE = MAIN_TEMPLATE + """Думай как агент-критик. Твоя задача - оценить ответ на вопрос пользователя и принять решение - что с ним делать.
Отвечать, переписать, запросить дополнительные данные у пользователя, запросить дополнительные данные в интернете.
Если для улучшения ответа нужен поиск в интернете - прими решение search и поиск в интернете будет выполнен.

Вопрос пользователя - {user_question}

Твои мысли по поводу вопроса пользователя:
<THINK>
{last_reason}
</THINK>

Предполагаемый ответ:
<LAST_ANSWER>
{last_answer}
</LAST_ANSWER>

Предыдущий набор критики, которую ты уже писал раньше (не повторяйся):
<OLD_CRITIQUE>
{critique}
</OLD_CRITIQUE>

Результаты поиска (если есть):
<SEARCH_RESULTS>
{search_results}
</SEARCH_RESULTS>

Выведи только следующую информацию в формате JSON:
{format_instructions}"""


def critique(state: GraphsState):
    parser = PydanticOutputParser(pydantic_object=Critique)
    prompt = ChatPromptTemplate.from_messages([
        ("system", CRITIQUE_TEMPLATE)
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    res = chain.invoke(
        {
            "user_question": state["user_question"],
            "last_reason": state["last_reason"],
            "last_answer": state["last_answer"],
            "critique": state.get("critique", []),
            "search_results": state.get("search_results", {})
        }
    )
    new_critique_str = res.critique
    final_decision = res.final_decision
    critique = res.critique
    search_query = res.search_query
    is_new_critique = res.is_new_critique
    search_mode = res.search_mode
    
    critique = state.get("critique", [])
    if critique is None:
        critique = []
    if is_new_critique:
        critique.append(new_critique_str)


    update = {"final_decision": final_decision, 
              "critique": critique, 
              "search_query": search_query, 
              "search_mode": search_mode, }
    goto = "🏁 finalizing"

    if final_decision == "search" and search_query is not None and len(search_query) > 0:
        goto = "🔍 Searcher"
        
    if final_decision == "fix":
        if is_new_critique:
            goto = "👨 answering"
        else:
            print("No new critique, go to finalizer")
            goto = "🏁 finalizing"

    return Command(
        update=update,
        goto=goto
    )

FINALIZER_TEMPLATE = MAIN_TEMPLATE + """Думай как агент-выпускающий редактор.
Твоя задача - написать окончательный ответ с учетом критики и предыдущих ответов и размышлений.

Вопрос пользователя - {user_question}

Твои мысли по поводу вопроса пользователя:
<THINK>
{last_reason}
</THINK>

Первая версия предполагаемого ответа:
<LAST_ANSWER>
{last_answer}
</LAST_ANSWER>

Результаты поиска (если есть):
<SEARCH_RESULTS>
{search_results}
</SEARCH_RESULTS>

Критика этой версии ответа:
<CRITICUE>
{critique}
</CRITICUE>

Напиши финальную версию ответа пользователю (четко и ясно):
"""

def finalize(state: GraphsState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", FINALIZER_TEMPLATE)
    ])

    chain = prompt | llm | StrOutputParser()

    res = chain.invoke(
        {
            "user_question": state.get("user_question", None),
            "last_reason": state.get("last_reason", None),
            "critique": state.get("critique", None),
            "last_answer": state.get("last_answer", None),
            "search_results": state.get("search_results", {})
        }
    )

    return {"final_answer": res}

def search(state: GraphsState):
    tavily_client = TavilyClient()
    search_mode = state.get("search_mode", "basic")
    if search_mode == "deep":
        response = tavily_client.search(state["search_query"], 
                                        search_depth="advanced", 
                                        include_raw_content=True)
    else:
        response = tavily_client.search(state["search_query"])

    search_results = state.get("search_results", {})
    search_results[state["search_query"]] = response
    return {"search_results": search_results}


graph.add_node("🤔 thinking", reason)
graph.add_node("1️⃣ first step think", first_step)
graph.add_node("👨 answering", answer)
graph.add_node("👨‍⚖️ self-criticque", critique)
graph.add_node("🔍 Searcher", search)
graph.add_node("🏁 finalizing", finalize)


graph.add_edge(START, "🤔 thinking")
graph.add_edge("🤔 thinking", "1️⃣ first step think")
graph.add_edge("👨 answering", "👨‍⚖️ self-criticque")
graph.add_edge("🔍 Searcher", "👨 answering")
graph.add_edge("🏁 finalizing", END)



graph_runnable = graph.compile()
