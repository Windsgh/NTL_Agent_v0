import datetime
import os
from langchain_anthropic import ChatAnthropic
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from Langchain_tool import gdelt_query_tool, Browser_Toolkit, Exa_search_tools, GoogleSerper_search
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
# Specify the persistent storage path
persistent_directory = r"C:\NTL_Agent\RAG\RAG2"
env_vars = {
    "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_API_KEY": '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',
    "OPENAI_API_KEY": ('sk-proj-rOxcV6nM5MG4s4IdamDN9Rm5ReNfjXYBTL5yKHqmfJ2ElIHddx3KbcfH'
                       '-HHkbacHUiwayVb45eT3BlbkFJypaIL5ph4qM7FbirP3z5JmQz-2Mc-HxEPMp6fhN8Fw0VKXmql4FfIGL3Euin'
                       '-qm1_5dzb2giwA'),
    "DASHSCOPE_API_KEY" : 'sk-2fb142771db4405c80212ea92cd5992d',
    "ANTHROPIC_API_KEY" : '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'
}
# Initialize the embeddings
embeddings = OpenAIEmbeddings()

# Load the existing knowledge vector store
knowledge_vector_store = Chroma(
    collection_name="knowledge-chroma",
    persist_directory=persistent_directory,
    embedding_function= embeddings
)

# Load the existing solution vector store
# solution_vector_store = Chroma(
#     collection_name="solution-chroma",
#     persist_directory=persistent_directory,
#     embedding_function= embeddings
# )

# Create retrievers for both vector stores
knowledge_retriever = knowledge_vector_store.as_retriever()
# solution_retriever = solution_vector_store.as_retriever()

print("已成功加载嵌入数据。")


knowledge_retriever = create_retriever_tool(
    knowledge_retriever,
    "knowledge_retriever",
    """Nighttime Light Remote Sensing Knowledge Base: A repository of literature, books, and reports on nighttime light remote sensing.
                Solution Handbook: Geospatial Code case and usage(like GEE python, geemap, GDAL, GeoPandas, Shapely, Rasterio, cartopy), practical solutions, case studies, and tool usage guides for nighttime light applications."""

)
# solution_retriever = create_retriever_tool(
#     solution_retriever,
#     "solution_retriever",
#     "Solution Handbook: Geospatial Code case and usage(like GEE python, geemap, GDAL, GeoPandas, Shapely, Rasterio, cartopy), practical solutions, case studies, and tool usage guides for nighttime light applications.",
# )

arxiv_toolkit = load_tools(["arxiv"])
Tavily_search = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_images=False)
tools = [knowledge_retriever,gdelt_query_tool,Tavily_search] + arxiv_toolkit
# Retrieve_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
# # llm = ChatOpenAI(model="o1-mini")
# Retrieve_llm_with_tools = Retrieve_llm.bind_tools(Retrieve_tools)



class State(TypedDict):
    messages: Annotated[list, add_messages]

### Edges


def grade_documents(state) -> Literal["agent", "rewrite"]:
    print("---CHECK RELEVANCE---")
    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4.1-mini", streaming=True)

    # llm_claude = ChatAnthropic(model="claude-3-5-sonnet-latest")

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS USEFUL---")
        return "agent"

    else:
        print("---DECISION: DOCS NOT USEFUL---")
        print(score)
        return "rewrite"


### Nodes


# 定义聊天机器人函数
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.prompts import ChatPromptTemplate


def agent(state):

    print("---CALL AGENT---")
    system_prompt_text = SystemMessage("""
            **Role**: As a Information Retriever, your task is to retrieve information from three key sources:
            1. Nighttime Light Remote Sensing Knowledge Base: A repository of literature, books, and reports on nighttime light remote sensing.
            2. Solution Handbook: Geospatial Code case and usage (like GEE python, geemap, GDAL, GeoPandas, Shapely, Rasterio, cartopy), GEE image retrieve, practical solutions, case studies, and tool usage guides for nighttime light applications.
            3. Internet-Based Information: Integrates Google search, BigQuery's GDELT, Arxiv, and the Requests toolkit for web-based data retrieval. Especially nighttime light remote sensing, socio-political and economic information.
            You should help Engineer and Code Assistant to access, analyze, and leverage domain-specific knowledge and real-world resources to solve complex tasks effectively.
            """)

    messages = state["messages"]

    # Create a prompt template that includes the system message and user messages
    prompt_template = ChatPromptTemplate.from_messages(
        [system_prompt_text] + messages  # Combine system prompt with existing messages
    )

    # Format the prompt to get a PromptValue
    formatted_prompt = prompt_template.format_prompt()

    # model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4.1-mini")
    llm_GPT = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    llm_qwen = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-max",
        # other params...
    )

    llm_claude = ChatAnthropic(model="claude-3-5-sonnet-latest")
    model = llm_GPT.bind_tools(tools)
    response = model.invoke(formatted_prompt)  # Use the formatted prompt here

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# def agent(state):
#     print("---CALL AGENT---")
#     # Initialize language model and bind tools
#     llm = ChatOpenAI(model="gpt-4o", temperature=0)
#     # Initialize language model and bind tools
#     system_prompt_text = SystemMessage("""
#         **Role**: As a Information Retriever, your task is to retrieve information from three key sources:
#         1.Nighttime Light Remote Sensing Knowledge Base: A repository of literature, books, and reports on nighttime light remote sensing.
#         2.Solution Handbook: Geospatial Code case and usage(like GEE python, geemap, GDAL, GeoPandas, Shapely, Rasterio, cartopy),GEE image retrieve, practical solutions, case studies, and tool usage guides for nighttime light applications.
#         3.Internet-Based Information: Integrates Google search, BigQuery's GDELT, Arxiv, and the Requests toolkit for web-based data retrieval. Especially nighttime light remote sensing，Socio-political and economic information.
#         You should help Engineer and Code Assistant to access, analyze, and leverage domain-specific knowledge and real-world resources to solve complex tasks effectively.
#         """)
#     memory1 = MemorySaver()
#     messages = state["messages"]
#     # 创建 ReAct 代理
#     react_agent = create_react_agent(llm, tools=tools, state_modifier=system_prompt_text, checkpointer=memory1)
#
#     # 使用代理进行调用
#     response = react_agent.invoke(messages)
#     # print(response["messages"][-1].content)
#     # print({"messages": [response]})
#     # return response["messages"][-1].content
#     # response = create_react_agent(llm, tools=tools, state_modifier=system_prompt_text).invoke(messages)
#     return {"messages": [response]}


def rewrite(state):

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Here is the origin answer:
    \n ------- \n
    {last_message} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = ChatOpenAI(temperature=0, model="gpt-4.1-mini", streaming=True)

    # llm_claude = ChatAnthropic(model="claude-3-5-sonnet-latest")
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    # prompt = hub.pull("rlm/rag-prompt")
    # 自定义 Prompt，覆盖 hub.pull 加载的内容
    custom_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to accurately answer the question. If you don't know the answer, just say that you don't know. Provide an accurate and clear answer."""),
        ("human", "Question: {question} \nContext: {context} \nAnswer:")
    ])

    # 替换原有的 prompt
    prompt = custom_prompt

    # LLM
    llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0, streaming=True)

    llm_claude = ChatAnthropic(model="claude-3-5-sonnet-latest")

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm_claude | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# print("*" * 20 + "Prompt[rlm/rag-prompt]" + "*" * 20)
# prompt = hub.pull("rlm/rag-prompt").pretty_print()  # Show what the prompt looks like


# Define a new graph
workflow = StateGraph(State)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
workflow.add_node("tools", ToolNode(tools=tools))
workflow.add_node("rewrite", rewrite)  # Re-writing the question
# workflow.add_node(
#     "generate", generate
# )  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
workflow.add_edge(START, "agent")
def select_next_node(state: State):
    return tools_condition(state)
# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    select_next_node,
    # Assess agent decision
    {"tools": "tools", "__end__": "__end__"},
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "tools",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("agent", END)
workflow.add_edge("rewrite", "agent")
memory = MemorySaver()
# Compile

graph = workflow.compile(checkpointer=memory )

# from PIL import Image
# from io import BytesIO
# # Display the graph
# image_data = graph.get_graph().draw_mermaid_png()
# # Read the image data
# image = Image.open(BytesIO(image_data))
# # Show the image
# image.show()


class Agentic_RAG_Input(BaseModel):
    query_with_background: str = Field(..., description="Your question along with brief contextual background.")

def Agentic_RAG(
        query_with_background: str,
) -> str:
    # result = graph.invoke({"messages": query_with_background})
    # The config is the **second positional argument** to stream() or invoke()!
    # print(query_with_background)
    events = graph.stream(
        input={"messages": [("user", query_with_background)]},
        config={"configurable": {"thread_id": "2"}, "recursion_limit": 11},
        stream_mode="values"
    )
    # print("OK")
    for event in events:
        event["messages"][-1].pretty_print()
        final_answer = event["messages"][-1].content
    # Get the response from the model
    return final_answer

# today = datetime.datetime.today().strftime("%D")
Agentic_RAG = StructuredTool.from_function(
    Agentic_RAG,
    name="Information_Retriever",
    description=(
    """
    Information_Retriever is a Agentic Retriever designed to retrieve information from three key sources:
    1.Nighttime Light Remote Sensing Knowledge Base: A repository of literature, books, and reports on nighttime light remote sensing.
    2.Solution Handbook: Geospatial Code case and usage(like GEE python, geemap, GDAL, GeoPandas, Shapely, Rasterio, cartopy),GEE image retrieve, practical solutions, case studies, and tool usage guides for nighttime light applications.
    3.Internet Information: Integrates Google search, BigQuery's GDELT, Arxiv, and the Requests toolkit for web-based data retrieval. Especially nighttime light remote sensing，Socio-political and economic information.
    Information_Retriever helps you to access, analyze, and leverage domain-specific knowledge and real-world resources to solve complex tasks effectively.
    To use Information_Retriever, input your Detailed question along with brief contextual background, and it will retrieve and deliver relevant results to you.
    """
    ),
    args_schema = Agentic_RAG_Input
)

# Agentic_RAG.func("Find 5 news related to the 2023 earthquake in Turkey from the GDELT")
# import pprint
#
# inputs = {
#     "messages": [
#         ("user", "Retrieve event information related to the 2023 Turkey earthquake from the GDELT Event Database."),
#     ]
# }
# for output in graph.stream(inputs,config={"configurable": {"thread_id": "2"}, "recursion_limit": 21}):
#     for key, value in output.items():
#         pprint.pprint(f"Output from node '{key}':")
#         pprint.pprint("---")
#         pprint.pprint(value, indent=2, width=80, depth=None)
#     pprint.pprint("\n---\n")