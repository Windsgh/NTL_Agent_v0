import getpass
from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
import ee
from GEE_Chain import NTL_download_tool

python_repl = PythonREPL()
# You can create the tool to pass to an agent
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

# Initialize Google Earth Engine
# ee.Authenticate()
project_id = 'empyrean-caster-430308-m2'
ee.Initialize(project=project_id)

# Set environment variables
def set_env_variable(var_name, default_value=None):
    if not os.environ.get(var_name):
        if default_value:
            os.environ[var_name] = default_value
        else:
            os.environ[var_name] = getpass.getpass(f"请输入您的 {var_name}: ")

# Set required environment variables
env_vars = {
    "TAVILY_API_KEY": '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',
    "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_API_KEY": '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',
    "OPENAI_API_KEY": ('sk-proj-rOxcV6nM5MG4s4IdamDN9Rm5ReNfjXYBTL5yKHqmfJ2ElIHddx3KbcfH'
                       '-HHkbacHUiwayVb45eT3BlbkFJypaIL5ph4qM7FbirP3z5JmQz-2Mc-HxEPMp6fhN8Fw0VKXmql4FfIGL3Euin'
                       '-qm1_5dzb2giwA'),
    "DASHSCOPE_API_KEY" : 'sk-2fb142771db4405c80212ea92cd5992d',
    "ANTHROPIC_API_KEY" : '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'
}

for var, value in env_vars.items():
    set_env_variable(var, value)



# warnings.filterwarnings("ignore")

tools = [repl_tool]

system_prompt_text = SystemMessage("""
You are an expert in nighttime light image processing and analysing(named Chat-NTL),
If you need additional information, e.g. night lighting or geospatial analysis or programming related knowledge, please seek help from Information_Retriever;
If you want to solve the NTL problem with Python code (Geocode_tool_local, visualisation_tool, Geocode_tool_GEE) or implement code bugs, 
you must first go through the Code Assistant's testing and optimisation.
(The code should be executed in modular steps, avoiding the use of functions.)
Your task is to answer questions or solve problems step-by-step using the tools you already have.
Before all actions begin, you need to first plan the overall execution steps to complete the task.
The final answer should be streamlined while maintaining completeness.
""")

# Initialize language model and bind tools
llm_GPT = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_retries=3)
memory = MemorySaver()
graph = create_react_agent(llm_GPT, tools=tools, state_modifier=system_prompt_text, checkpointer=memory)

