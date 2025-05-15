from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from Agentic_RAG import Agentic_RAG
from geo_data_processing_tool import Geocode_tool_local, visualization_tool, Geocode_tool_GEE
from langchain_core.messages import SystemMessage
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
system_prompt_text = SystemMessage("""
**Role**: As a Code Assistant, your first task is to ensure the robustness, reliability and scalability of the code, and to create comprehensive test cases for the incomplete function .
Second, if the test fails, return the error content and the reason for the error and advice to tell the Engineer, so that it can further optimise or provide you with additional information.
Third, generates automated test scripts to systematically evaluate spatial resolution, attribute accuracy, coordinate systems, data extents, data integrity, and logical consistency within geospatial datasets. 
Please note that when calculating the number or proportion of pixels, pixels with NoData values should be excluded first, e.g., total_pixels = np.sum(ntl_data > 0).
Upon successfully passing all validation tests, please execute the final code. 
In case of geospatial knowledge gaps, such as when selecting appropriate datasets from Google Earth Engine (GEE), please consult guidelines from the Information Retriever."""
)
tools = [Agentic_RAG, Geocode_tool_GEE,visualization_tool, Geocode_tool_local ]

# Initialize language model and bind tools
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
from langgraph.prebuilt import create_react_agent
# Initialize language model and bind tools
# llm_GPT = ChatOpenAI(model="gpt-4.1", temperature=0)
llm_GPT = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_qwen = ChatOpenAI(
    api_key="sk-89faaf7259be4eda8aca793aab170e1c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-max",
    # other params...
)

from langchain_anthropic import ChatAnthropic

llm_claude = ChatAnthropic(model="claude-3-5-sonnet-latest")

from langgraph.prebuilt import create_react_agent
# Initialize language model and bind tools
memory = MemorySaver()
graph = create_react_agent(llm_GPT, tools=tools, state_modifier=system_prompt_text, checkpointer=memory)



