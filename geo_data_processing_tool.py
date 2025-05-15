from langchain.tools import StructuredTool
from langchain_experimental.utilities import PythonREPL
# 更新函数，使用解包参数
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
import ee
import geemap
from IPython import InteractiveShell
import io
from contextlib import redirect_stdout
import json
# Define the function to execute geographic data processing commands
import re
from IPython import get_ipython
import rasterio


# def get_fixed_code_from_llm(task_description: str, geo_python_code: str) -> str:
#     # Instantiate the GPT model for code correction
#     code_llm = ChatOpenAI(model="gpt-4o", temperature=0)
#
#     # Prepare the prompt for the model to fix and optimize the code
#     prompt = f"""
#     You are an expert in geographic data processing and Python programming.
#     I have the following task description and Python code:
#
#     Task description:
#     {task_description}
#
#     Python code:
#     {geo_python_code}
#
#     Please detect any errors or inefficiencies in the code, fix the issues, and return ONLY the corrected, optimized, executable version of the code.
#     Do not include any explanations or other text—just return the modified code.
#     """
#
#     # Get the response from the model
#     response = code_llm.invoke([{"role": "system", "content": "You are a helpful code assistant."},
#                            {"role": "user", "content": prompt}])
#
#     # Extract the fixed code from the response
#     fixed_code = response.content
#     return fixed_code
# Define the input schema for geographic data processing tasks
class GeoDataProcessingInput(BaseModel):
    task_description: str = Field(
        ...,
        description=(
            "A detailed description of the geographic data processing task to be performed. "
            "Specify the operation, input file paths, parameters, and expected outputs. "
            "Examples include calculating zonal statistics, resampling raster data, performing band calculations, "
            "mask extraction, and general raster processing operations.\n\n")
    )
    geo_python_code: str = Field(
        ...,
        description=(
            "Python code generated based on the task description. "
            "The code should be carefully reviewed before execution."
        )
    )



python_repl = PythonREPL()
### Build Python repl ###
# You can create the tool to pass to an agent
# repl_tool = Tool(
#     name="python_repl",
#     description=(
#         """
#         A Python shell. Use this to execute Python commands. Input should be a valid Python command.
#         To ensure correct path handling, all file paths should be provided as raw strings (e.g., r'C:\\path\\to\\file').
#         Paths should also be constructed using os.path.join() for cross-platform compatibility.
#         Additionally, verify the existence of files before attempting to access them.
#         After executing the command, use the `print` function to output the results.
#         """
#
#     ),
#     func=python_repl.run,
# )

# Define the function to execute geographic data processing commands
def Geocode_tool_local(task_description: str, geo_python_code: str) -> str:
    try:
        result = python_repl.run(geo_python_code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"

    # Use escaped backticks or raw string for the code block
    result_str = f"Successfully executed\nStdout: {result}"
    return result_str

    # # Initialize the interactive Python shell
    # shell = get_ipython()
    # if shell is None:
    #     return json.dumps({
    #         "status": "error",
    #         "logs": "",
    #         "error": "Interactive Python shell is not available."
    #     }, indent=2)
    #
    # # Step 1: Print the input code (for logging purposes)
    # # print(f"Task Description: {task_description}")
    # # print(f"Input Python Code:\n{geo_python_code}")
    #
    # # Step 2: Execute the Python code
    # try:
    #     # Capture stdout
    #     captured_output = io.StringIO()
    #     with redirect_stdout(captured_output):
    #         result = shell.run_cell(geo_python_code)
    #
    #     # Get captured stdout content
    #     execution_logs = captured_output.getvalue()
    #
    #     # Remove ANSI escape codes from logs (if present)
    #     execution_logs = re.sub(r'\x1b\[[0-9;]*m', '', execution_logs)
    #
    #     # Check for errors in execution
    #     if hasattr(result, 'error_in_exec') and result.error_in_exec:
    #         return json.dumps({
    #             "status": "error",
    #             "logs": execution_logs,
    #             "error": str(result.error_in_exec)
    #         }, indent=2)
    #
    #     # Return success logs
    #     return json.dumps({
    #         "status": "Run",
    #         "logs": execution_logs
    #     }, indent=2)
    #
    # except Exception as e:
    #     # Handle unexpected exceptions
    #     return json.dumps({
    #         "status": "error",
    #         "logs": captured_output.getvalue() if 'captured_output' in locals() else "",
    #         "error": str(e)
    #     }, indent=2)


class GeoVisualizationInput(BaseModel):
    task_description: str = Field(
        ...,
        description=(
            "A detailed description of the visualization task, including goals, data, and expected outputs. "
        )
    )
    visualization_code: str = Field(
        ...,
        description=(
            "Python code that generates the geographic visualization. "
            "Include imports, data processing, plotting, and saving the figure, style like Nature and Science. "
            "Ensure the code is safe and follows best practices."
        )
    )


# Define the visualization function
def geo_visualization_func(
        task_description: str,
        visualization_code: str
) -> str:
    # print(f"Task Description: {task_description}")
    # print(f"Visualization Code: {visualization_code}")
    try:
        result = python_repl.run(visualization_code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed\nStdout: {result}"
    return result_str




project_id = 'empyrean-caster-430308-m2'
ee.Initialize(project=project_id)
# Initialize IPython shell
shell = InteractiveShell().instance()


# Define the input schema for geographic data processing tasks
class GEE_Task(BaseModel):
    task_description: str = Field(
        ...,
        description=(
            "A detailed description of GEE processing task to be performed. "
            "Specify the operation, input file paths, parameters, and expected outputs. ")
    )
    GEE_python_code: str = Field(
        ...,
        description=(
            "GEE Python code generated based on the task description. "
            "The code should be carefully reviewed before execution.\n\n"
            "The code should initialize Earth Engine, process the geographic data, and output the results.\n"
            "It should include the necessary imports, initialization of the Earth Engine project, "
            "and any required error handling."
        )
    )


def Geocode_tool_GEE(task_description: str, GEE_python_code: str) -> str:
    # print(f"Task Description: {task_description}")

    # Initialize the interactive Python shell
    shell = get_ipython()
    if shell is None:
        return json.dumps({
            "status": "error",
            "logs": "",
            "error": "Interactive Python shell is not available."
        }, indent=2)

    # Step 1: Print the input code (for logging purposes)
    # print(f"Input Python Code:\n{GEE_python_code}")

    # Step 2: Execute the Python code
    try:
        # Capture stdout
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            result = shell.run_cell(GEE_python_code)

        # Get captured stdout content
        execution_logs = captured_output.getvalue()

        # Remove ANSI escape codes from logs (if present)
        execution_logs = re.sub(r'\x1b\[[0-9;]*m', '', execution_logs)

        # Check for errors in execution
        if hasattr(result, 'error_in_exec') and result.error_in_exec:
            return json.dumps({
                "status": "error",
                "logs": execution_logs,
                "error": str(result.error_in_exec)
            }, indent=2)

        # Return success logs
        return json.dumps({
            "status": "Run",
            "logs": execution_logs
        }, indent=2)

    except Exception as e:
        # Handle unexpected exceptions
        return json.dumps({
            "status": "error",
            "logs": captured_output.getvalue() if 'captured_output' in locals() else "",
            "error": str(e)
        }, indent=2)

# Create the StructuredTool for geographic data processing
Geocode_tool_local = StructuredTool.from_function(
    Geocode_tool_local,
    name="Geocode_tool_local",
    description=(
        """
        This tool (based on a Python shell) is used to processing and analysing geographic data such as nighttime light imagery.
        Key use cases include:

        1. Raster Statistics.
        2. Resampling: Resample raster data.
        3. Band Calculation.
        4. Mask Extraction. *set outside mask to -999*
        5. General Raster Processing.

        **Usage**: In `task_description`, provide a clear description with file paths and parameters.
        In `geo_python_code`, Use raw strings (e.g., r'C:\\path\\to\\file') and `os.path.join()`
        Write Python code that handles file existence with `os.path.exists()` and includes error handling.
        Use: `print(f'Result: {result}')` to return values.
        The code should be executed in modular steps, avoiding the use of functions.
        """
    )
    ,
    args_schema=GeoDataProcessingInput,
)

# Create the StructuredTool for geographic data processing
Geocode_tool_GEE = StructuredTool.from_function(
    Geocode_tool_GEE,
    name="Geocode_tool_GEE",
    description=(
        """
        This tool (based on an IPython shell) performs normal and simple GEE processing tasks.

        **Usage**: In `task_description`, provide a clear description with file paths and parameters.

        In `GEE_python_code`,Please initialise first: "ee.Initialize(project='empyrean-caster-430308-m2')".
        Using the image collection from NTL_retrieve_tool, and including error handling.
        Use raw strings (e.g., r'C:\\path\\to\\file') and `os.path.join()` for cross-platform path handling.
        you should use: `print(f'Result: {result}')` to return values.
        The tool captures all standard output (`stdout`) from the task and returns it as `logs` in the result.
        Any errors in the execution will be returned as `error` in the result.
        """

    )
    ,
    input_type=GEE_Task,
)

# Create the StructuredTool with updated description and usage instructions
visualization_tool = StructuredTool.from_function(
    geo_visualization_func,
    name="Visualization_tool",
    description=(
        """
        This tool (based on a Python shell) is specially for geographic data visualization.
        **Usage**:
        The `task_description` outlines the related visualization task.
        The `visualization_code` should include imports, data processing, plotting, and saving the figure, styled like Science. 
        Default Setting:
        vmin, vmax = np.percentile(ntl_data, (1, 99))
        cax = ax.imshow(ntl_data, cmap='cividis', vmin=vmin, vmax=vmax)
        figsize=(10, 10); title: fontsize=16, fontweight='bold'; x/ylabel: fontsize=15
        Do not visualize the nodata (-999).
        Figure must be saved in the folder path `C:/NTL_Agent/report/image` with the English name (e.g., 'NTL Image of Nanjing - June 2020.png'), 300 DPI. 
        Use raw strings (e.g., r'C:\\path\\to\\file') and `os.path.join()`.
        Use: `print(f'Full storage address: {save_path}')`.
        """

    ),
    input_type=GeoVisualizationInput,
)
# # 任务描述和 Python 代码
# task_description = """
# Mask the NTL image for Shanghai using the shapefile and set the outer values to nodata.
# """
#
# geo_python_code = """
# import numpy as np
#
# # Function to calculate the sum of squares of positive numbers
# def sum_of_squares_positive(arr):
#     positive_numbers = arr[arr > 0]  # Filter positive numbers
#     return np.sum(positive_numbers ** 2)  # Return the sum of their squares
#
# # Example array
# data = np.array([-1, 2, 3, -4, 5, -6])
#
# # Call the function and print the result
# result = sum_of_squares_positive(data)
# print(f"The sum of squares of positive numbers is: {result}")
#
# """
# #
# result1 = Geocode_tool_local.func(task_description,geo_python_code)
# print(f'Result: {result1}')
