from langchain_core.tools import StructuredTool
from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel, Field
from langchain.tools import Tool
from Code_Agent import graph
import rasterio

# tools = [repl_tool]
class Code_assistant_Input(BaseModel):
    Understand_Clarify: str = Field(..., description="Analyze the geospatial problem.")
    Algorithm_Method_Selection: str= Field(..., description="Choose the most efficient libraries (e.g., GeoPandas, Shapely, Rasterio).")
    Pseudocode_Creation: str= Field(..., description="Outline clear logical steps for data processing, analysis, or visualization.")
    Ori_Code_Generation: str= Field(..., description="Implement robust, well-documented Python code ready for testing.")
    # background: str = Field(..., description="Brief contextual background about your code.")
#         1.请你首先对Origin Code进行测试和优化
#         2.如果测试通过，则执行最终的代码
#         3.若最终代码通过则返回执行结果给Programmer；若最终代码未通过则返回错误内容及你认为的错误原因告诉Programmer，让它进一步优化或提供给你额外补充的信息。

def code_assistant(
        Understand_Clarify: str,
        Algorithm_Method_Selection: str,
        Pseudocode_Creation: str,
        Ori_Code_Generation: str ,
        # background: str,

) -> str:
    # code_llm = ChatOpenAI(model="gpt-4o", temperature=0)
    # code_llm_with_tools = code_llm.bind_tools(tools)
    user_input = f"""
        You are an expert in geographic data processing and Python programming.
        I have the following task description and Python code:

        Understand and Clarify:
        {Understand_Clarify}

        Algorithm/Method Selection:
        {Algorithm_Method_Selection}

        Pseudocode Creation:
        {Pseudocode_Creation}

        Origin Code:
        {Ori_Code_Generation}

        1. Please test and optimise the Origin Code first
        2. If the test passes, then execute the final code you have corrected
        3. If the final code passes, then return the execution result to me; if the final code does not pass, then return the final code， the error content and the reason for the error you think to tell me, so that I can further optimise or provide you with additional information.
        """

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [("user", user_input)]}, config={"configurable": {"thread_id": "3"}, "recursion_limit": 7}, stream_mode="values"
    )
    for event in events:
        event["messages"][-1].pretty_print()
        final_answer = event["messages"][-1].content
    # Get the response from the model
    return final_answer
#
Code_assistant = StructuredTool.from_function(
    code_assistant,
    name="Code_assistant",
    description=(
    """
    A Code assistant can executes python code to process, analyse and visualise geographic data，
    it will test, optimize, execute geospatial processing Python code for processing, analysis or visualization and return results or report errors for refinement.

    Task:
    Use a Chain-of-Thought approach to break down the problem, design pseudocode, and implement the solution in Python with a focus on efficiency, readability, and comments.

    Instructions:
        Understand and Clarify: Analyze the geospatial problem and provide a brief contextual background
        Algorithm/Method Selection: Choose the most efficient libraries (e.g., GeoPandas, Shapely, Rasterio, Cartopy, GDAL).
        Pseudocode Creation: Outline clear logical steps for data processing, analysis, or visualization.
        Code Generation: Implement robust, well-documented Python code ready for testing.
    """
    ),
    args_schema = Code_assistant_Input
)
#
#
# Understand_Clarify="用户需要计算位于指定路径的夜间灯光影像数据的均值和极值。",
# Algorithm_Method_Selection="使用Rasterio来读取影像数据，使用NumPy进行均值、最小值和最大值的计算。",
# Pseudocode_Creation="1. 使用Rasterio打开影像文件。\n2. 提取影像的第一个波段数据。\n3. 使用NumPy计算数据的均值、最小值和最大值。\n4. 输出结果。",
# Ori_Code="""
# import rasterio
# import numpy as np
#
# # 读取影像
# image_path = r'C:\\NTL_Agent\\Night_data\\Shanghai\\上海市_NightLights_2022-07_Masked.tif'
# with rasterio.open(image_path) as src:
# # 读取影像数据
# image_data = src.read(1)  # 读取第一个波段
#
# # 计算均值和极值
# mean_value = np.mean(image_data)
# min_value = np.min(image_data)
# max_value = np.max(image_data)
#
# # 输出结果
# print(f'Mean: {mean_value}, Min: {min_value}, Max: {max_value}')
# """


# Call the tool and capture the result
# result = Code_assistant.func(Understand_Clarify,Algorithm_Method_Selection,Pseudocode_Creation,Ori_Code)
# print(123)
# Print the result
# print(result)
