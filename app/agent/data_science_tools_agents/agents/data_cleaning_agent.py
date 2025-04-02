# * Agents: Data Cleaning Agent

# Libraries
import json
import operator
import os

from collections.abc import Sequence
from typing import Annotated, Literal, TypedDict

import pandas as pd

from IPython.display import Markdown
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Checkpointer, Command

from data_science_tools_agents.parsers.parsers import PythonOutputParser
from data_science_tools_agents.templates import (
    BaseAgent,
    create_coding_agent_graph,
    node_func_execute_agent_code_on_data,
    node_func_fix_agent_code,
    node_func_human_review,
    node_func_report_agent_outputs,
)
from data_science_tools_agents.tools.dataframe import get_dataframe_summary
from data_science_tools_agents.utils.logging import log_ai_function
from data_science_tools_agents.utils.regex import (
    add_comments_to_top,
    format_agent_name,
    format_recommended_steps,
    get_generic_summary,
    relocate_imports_inside_function,
)


# Setup
AGENT_NAME = "data_cleaning_agent"
LOG_PATH = os.path.join(os.getcwd(), "logs/")


# Class
class DataCleaningAgent(BaseAgent):
    """Creates a data cleaning agent that can process datasets based on user-defined instructions or default cleaning steps.
    The agent generates a Python function to clean the dataset, performs the cleaning, and logs the process, including code
    and errors. It is designed to facilitate reproducible and customizable data cleaning workflows.

    The agent performs the following default cleaning steps unless instructed otherwise:

    - Removing columns with more than 40% missing values.
    - Imputing missing values with the mean for numeric columns.
    - Imputing missing values with the mode for categorical columns.
    - Converting columns to appropriate data types.
    - Removing duplicate rows.
    - Removing rows with missing values.
    - Removing rows with extreme outliers (values 3x the interquartile range).

    User instructions can modify, add, or remove any of these steps to tailor the cleaning process.

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model used to generate the data cleaning function.
    n_samples : int, optional
        Number of samples used when summarizing the dataset. Defaults to 30. Reducing this number can help
        avoid exceeding the model's token limits.
    log : bool, optional
        Whether to log the generated code and errors. Defaults to False.
    log_path : str, optional
        Directory path for storing log files. Defaults to None.
    file_name : str, optional
        Name of the file for saving the generated response. Defaults to "data_cleaner.py".
    function_name : str, optional
        Name of the generated data cleaning function. Defaults to "data_cleaner".
    overwrite : bool, optional
        Whether to overwrite the log file if it exists. If False, a unique file name is created. Defaults to True.
    human_in_the_loop : bool, optional
        Enables user review of data cleaning instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        If True, skips the default recommended cleaning steps. Defaults to False.
    bypass_explain_code : bool, optional
        If True, skips the step that provides code explanations. Defaults to False.
    checkpointer : langgraph.types.Checkpointer, optional
        Checkpointer to save and load the agent's state. Defaults to None.

    Methods
    -------
    update_params(**kwargs)
        Updates the agent's parameters and rebuilds the compiled state graph.
    ainvoke_agent(user_instructions: str, data_raw: pd.DataFrame, max_retries=3, retry_count=0)
        Cleans the provided dataset asynchronously based on user instructions.
    invoke_agent(user_instructions: str, data_raw: pd.DataFrame, max_retries=3, retry_count=0)
        Cleans the provided dataset synchronously based on user instructions.
    get_workflow_summary()
        Retrieves a summary of the agent's workflow.
    get_log_summary()
        Retrieves a summary of logged operations if logging is enabled.
    get_state_keys()
        Returns a list of keys from the state graph response.
    get_state_properties()
        Returns detailed properties of the state graph response.
    get_data_cleaned()
        Retrieves the cleaned dataset as a pandas DataFrame.
    get_data_raw()
        Retrieves the raw dataset as a pandas DataFrame.
    get_data_cleaner_function()
        Retrieves the generated Python function used for cleaning the data.
    get_recommended_cleaning_steps()
        Retrieves the agent's recommended cleaning steps.
    get_response()
        Returns the response from the agent as a dictionary.
    show()
        Displays the agent's mermaid diagram.

    Examples
    --------
    ```python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from data_science_tools_agents.agents import DataCleaningAgent

    llm = ChatOpenAI(model="gpt-4o-mini")

    data_cleaning_agent = DataCleaningAgent(model=llm, n_samples=50, log=True, log_path="logs", human_in_the_loop=True)

    df = pd.read_csv(
        "https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv"
    )

    data_cleaning_agent.invoke_agent(
        user_instructions="Don't remove outliers when cleaning the data.", data_raw=df, max_retries=3, retry_count=0
    )

    cleaned_data = data_cleaning_agent.get_data_cleaned()

    response = data_cleaning_agent.response
    ```

    Returns
    -------
    DataCleaningAgent : langchain.graphs.CompiledStateGraph
        A data cleaning agent implemented as a compiled state graph.

    """

    def __init__(
        self,
        model,
        n_samples=30,
        log=False,
        log_path=None,
        file_name="data_cleaner.py",
        function_name="data_cleaner",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        checkpointer: Checkpointer = None,
    ):
        self._params = {
            "model": model,
            "n_samples": n_samples,
            "log": log,
            "log_path": log_path,
            "file_name": file_name,
            "function_name": function_name,
            "overwrite": overwrite,
            "human_in_the_loop": human_in_the_loop,
            "bypass_recommended_steps": bypass_recommended_steps,
            "bypass_explain_code": bypass_explain_code,
            "checkpointer": checkpointer,
        }
        self._compiled_graph = self._make_compiled_graph()
        self.response = None

    def _make_compiled_graph(self):
        """Create the compiled graph for the data cleaning agent. Running this method will reset the response to None."""
        self.response = None
        return make_data_cleaning_agent(**self._params)

    async def ainvoke_agent(
        self,
        data_raw: pd.DataFrame,
        user_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """Asynchronously invokes the agent. The response is stored in the response attribute.

        Parameters
        ----------
            data_raw (pd.DataFrame):
                The raw dataset to be cleaned.
            user_instructions (str):
                Instructions for data cleaning agent.
            max_retries (int):
                Maximum retry attempts for cleaning.
            retry_count (int):
                Current retry attempt.
            **kwargs
                Additional keyword arguments to pass to ainvoke().

        Returns
        -------
            None. The response is stored in the response attribute.

        """
        response = await self._compiled_graph.ainvoke(
            {
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return

    def invoke_agent(
        self,
        data_raw: pd.DataFrame,
        user_instructions: str = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ):
        """Invokes the agent. The response is stored in the response attribute.

        Parameters
        ----------
            data_raw (pd.DataFrame):
                The raw dataset to be cleaned.
            user_instructions (str):
                Instructions for data cleaning agent.
            max_retries (int):
                Maximum retry attempts for cleaning.
            retry_count (int):
                Current retry attempt.
            **kwargs
                Additional keyword arguments to pass to invoke().

        Returns
        -------
            None. The response is stored in the response attribute.

        """
        response = self._compiled_graph.invoke(
            {
                "user_instructions": user_instructions,
                "data_raw": data_raw.to_dict(),
                "max_retries": max_retries,
                "retry_count": retry_count,
            },
            **kwargs,
        )
        self.response = response
        return

    def get_workflow_summary(self, markdown=False):
        """Retrieves the agent's workflow summary, if logging is enabled."""
        if self.response and self.response.get("messages"):
            summary = get_generic_summary(json.loads(self.response.get("messages")[-1].content))
            if markdown:
                return Markdown(summary)
            return summary

    def get_log_summary(self, markdown=False):
        """Logs a summary of the agent's operations, if logging is enabled."""
        if self.response:
            if self.response.get("data_cleaner_function_path"):
                log_details = f"""
## Data Cleaning Agent Log Summary:

Function Path: {self.response.get("data_cleaner_function_path")}

Function Name: {self.response.get("data_cleaner_function_name")}
                """
                if markdown:
                    return Markdown(log_details)
                return log_details

    def get_data_cleaned(self):
        """Retrieves the cleaned data stored after running invoke_agent or clean_data methods."""
        if self.response:
            return pd.DataFrame(self.response.get("data_cleaned"))

    def get_data_raw(self):
        """Retrieves the raw data."""
        if self.response:
            return pd.DataFrame(self.response.get("data_raw"))

    def get_data_cleaner_function(self, markdown=False):
        """Retrieves the agent's pipeline function."""
        if self.response:
            if markdown:
                return Markdown(f"```python\n{self.response.get('data_cleaner_function')}\n```")
            return self.response.get("data_cleaner_function")

    def get_recommended_cleaning_steps(self, markdown=False):
        """Retrieves the agent's recommended cleaning steps"""
        if self.response:
            if markdown:
                return Markdown(self.response.get("recommended_steps"))
            return self.response.get("recommended_steps")


# Agent


def make_data_cleaning_agent(
    model,
    n_samples=30,
    log=False,
    log_path=None,
    file_name="data_cleaner.py",
    function_name="data_cleaner",
    overwrite=True,
    human_in_the_loop=False,
    bypass_recommended_steps=False,
    bypass_explain_code=False,
    checkpointer: Checkpointer = None,
):
    """Creates a data cleaning agent that can be run on a dataset. The agent can be used to clean a dataset in a variety of
    ways, such as removing columns with more than 40% missing values, imputing missing
    values with the mean of the column if the column is numeric, or imputing missing
    values with the mode of the column if the column is categorical.
    The agent takes in a dataset and some user instructions, and outputs a python
    function that can be used to clean the dataset. The agent also logs the code
    generated and any errors that occur.

    The agent is instructed to to perform the following data cleaning steps:

    - Removing columns if more than 40 percent of the data is missing
    - Imputing missing values with the mean of the column if the column is numeric
    - Imputing missing values with the mode of the column if the column is categorical
    - Converting columns to the correct data type
    - Removing duplicate rows
    - Removing rows with missing values
    - Removing rows with extreme outliers (3X the interquartile range)
    - User instructions can modify, add, or remove any of the above steps

    Parameters
    ----------
    model : langchain.llms.base.LLM
        The language model to use to generate code.
    n_samples : int, optional
        The number of samples to use when summarizing the dataset. Defaults to 30.
        If you get an error due to maximum tokens, try reducing this number.
        > "This model's maximum context length is 128000 tokens. However, your messages resulted in 333858 tokens. Please reduce the length of the messages."
    log : bool, optional
        Whether or not to log the code generated and any errors that occur.
        Defaults to False.
    log_path : str, optional
        The path to the directory where the log files should be stored. Defaults to
        "logs/".
    file_name : str, optional
        The name of the file to save the response to. Defaults to "data_cleaner.py".
    function_name : str, optional
        The name of the function that will be generated to clean the data. Defaults to "data_cleaner".
    overwrite : bool, optional
        Whether or not to overwrite the log file if it already exists. If False, a unique file name will be created.
        Defaults to True.
    human_in_the_loop : bool, optional
        Whether or not to use human in the loop. If True, adds an interput and human in the loop step that asks the user to review the data cleaning instructions. Defaults to False.
    bypass_recommended_steps : bool, optional
        Bypass the recommendation step, by default False
    bypass_explain_code : bool, optional
        Bypass the code explanation step, by default False.
    checkpointer : langgraph.types.Checkpointer, optional
        Checkpointer to save and load the agent's state. Defaults to None.

    Examples
    --------
    ``` python
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from data_science_tools_agents.agents import data_cleaning_agent

    llm = ChatOpenAI(model="gpt-4o-mini")

    data_cleaning_agent = make_data_cleaning_agent(llm)

    df = pd.read_csv(
        "https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv"
    )

    response = data_cleaning_agent.invoke(
        {
            "user_instructions": "Don't remove outliers when cleaning the data.",
            "data_raw": df.to_dict(),
            "max_retries": 3,
            "retry_count": 0,
        }
    )

    pd.DataFrame(response["data_cleaned"])
    ```

    Returns
    -------
    app : langchain.graphs.CompiledStateGraph
        The data cleaning agent as a state graph.

    """
    llm = model

    if human_in_the_loop:
        if checkpointer is None:
            print("Human in the loop is enabled. A checkpointer is required. Setting to MemorySaver().")
            checkpointer = MemorySaver()

    # Human in th loop requires recommended steps
    if bypass_recommended_steps and human_in_the_loop:
        bypass_recommended_steps = False
        print("Bypass recommended steps set to False to enable human in the loop.")

    # Setup Log Directory
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    # Define GraphState for the router
    class GraphState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        user_instructions: str
        recommended_steps: str
        data_raw: dict
        data_cleaned: dict
        all_datasets_summary: str
        data_cleaner_function: str
        data_cleaner_function_path: str
        data_cleaner_file_name: str
        data_cleaner_function_name: str
        data_cleaner_error: str
        max_retries: int
        retry_count: int

    def recommend_cleaning_steps(state: GraphState):
        """Recommend a series of data cleaning steps based on the input data.
        These recommended steps will be appended to the user_instructions.
        """
        print(format_agent_name(AGENT_NAME))
        print("    * RECOMMEND CLEANING STEPS")

        # Prompt to get recommended steps from the LLM
        recommend_steps_prompt = PromptTemplate(
            template="""
            You are a Data Cleaning Expert. Given the following information about the data,
            recommend a series of numbered steps to take to clean and preprocess it.
            The steps should be tailored to the data characteristics and should be helpful
            for a data cleaning agent that will be implemented.

            General Steps:
            Things that should be considered in the data cleaning steps:

            * Removing columns if more than 40 percent of the data is missing
            * Imputing missing values with the mean of the column if the column is numeric
            * Imputing missing values with the mode of the column if the column is categorical
            * Converting columns to the correct data type
            * Removing duplicate rows
            * Removing rows with missing values
            * Removing rows with extreme outliers (3X the interquartile range)

            Custom Steps:
            * Analyze the data to determine if any additional data cleaning steps are needed.
            * Recommend steps that are specific to the data provided. Include why these steps are necessary or beneficial.
            * If no additional steps are needed, simply state that no additional steps are required.

            IMPORTANT:
            Make sure to take into account any additional user instructions that may add, remove or modify some of these steps. Include comments in your code to explain your reasoning for each step. Include comments if something is not done because a user requested. Include comments if something is done because a user requested.

            User instructions:
            {user_instructions}

            Previously Recommended Steps (if any):
            {recommended_steps}

            Below are summaries of all datasets provided:
            {all_datasets_summary}

            Return steps as a numbered list. You can return short code snippets to demonstrate actions. But do not return a fully coded solution. The code will be generated separately by a Coding Agent.

            Avoid these:
            1. Do not include steps to save files.
            2. Do not include unrelated user instructions that are not related to the data cleaning.
            """,
            input_variables=["user_instructions", "recommended_steps", "all_datasets_summary"],
        )

        data_raw = state.get("data_raw")
        df = pd.DataFrame.from_dict(data_raw)

        all_datasets_summary = get_dataframe_summary([df], n_sample=n_samples)

        all_datasets_summary_str = "\n\n".join(all_datasets_summary)

        steps_agent = recommend_steps_prompt | llm
        recommended_steps = steps_agent.invoke(
            {
                "user_instructions": state.get("user_instructions"),
                "recommended_steps": state.get("recommended_steps"),
                "all_datasets_summary": all_datasets_summary_str,
            }
        )

        return {
            "recommended_steps": format_recommended_steps(
                recommended_steps.content.strip(), heading="# Recommended Data Cleaning Steps:"
            ),
            "all_datasets_summary": all_datasets_summary_str,
        }

    def create_data_cleaner_code(state: GraphState):
        print("    * CREATE DATA CLEANER CODE")

        if bypass_recommended_steps:
            print(format_agent_name(AGENT_NAME))

            data_raw = state.get("data_raw")
            df = pd.DataFrame.from_dict(data_raw)

            all_datasets_summary = get_dataframe_summary([df], n_sample=n_samples)

            all_datasets_summary_str = "\n\n".join(all_datasets_summary)
        else:
            all_datasets_summary_str = state.get("all_datasets_summary")

        data_cleaning_prompt = PromptTemplate(
            template="""
            You are a Data Cleaning Agent. Your job is to create a {function_name}() function that can be run on the data provided using the following recommended steps.

            Recommended Steps:
            {recommended_steps}

            You can use Pandas, Numpy, and Scikit Learn libraries to clean the data.

            Below are summaries of all datasets provided. Use this information about the data to help determine how to clean the data:

            {all_datasets_summary}

            Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that includes all imports inside the function.

            Return code to provide the data cleaning function:

            def {function_name}(data_raw):
                import pandas as pd
                import numpy as np
                ...
                return data_cleaned

            Best Practices and Error Preventions:

            Always ensure that when assigning the output of fit_transform() from SimpleImputer to a Pandas DataFrame column, you call .ravel() or flatten the array, because fit_transform() returns a 2D array while a DataFrame column is 1D.

            """,
            input_variables=["recommended_steps", "all_datasets_summary", "function_name"],
        )

        data_cleaning_agent = data_cleaning_prompt | llm | PythonOutputParser()

        response = data_cleaning_agent.invoke(
            {
                "recommended_steps": state.get("recommended_steps"),
                "all_datasets_summary": all_datasets_summary_str,
                "function_name": function_name,
            }
        )

        response = relocate_imports_inside_function(response)
        response = add_comments_to_top(response, agent_name=AGENT_NAME)

        # For logging: store the code generated:
        file_path, file_name_2 = log_ai_function(
            response=response, file_name=file_name, log=log, log_path=log_path, overwrite=overwrite
        )

        return {
            "data_cleaner_function": response,
            "data_cleaner_function_path": file_path,
            "data_cleaner_file_name": file_name_2,
            "data_cleaner_function_name": function_name,
            "all_datasets_summary": all_datasets_summary_str,
        }

    # Human Review

    prompt_text_human_review = (
        "Are the following data cleaning instructions correct? (Answer 'yes' or provide modifications)\n{steps}"
    )

    if not bypass_explain_code:

        def human_review(
            state: GraphState,
        ) -> Command[Literal["recommend_cleaning_steps", "explain_data_cleaner_code"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="explain_data_cleaner_code",
                no_goto="recommend_cleaning_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="data_cleaner_function",
            )
    else:

        def human_review(state: GraphState) -> Command[Literal["recommend_cleaning_steps", "__end__"]]:
            return node_func_human_review(
                state=state,
                prompt_text=prompt_text_human_review,
                yes_goto="__end__",
                no_goto="recommend_cleaning_steps",
                user_instructions_key="user_instructions",
                recommended_steps_key="recommended_steps",
                code_snippet_key="data_cleaner_function",
            )

    def execute_data_cleaner_code(state):
        return node_func_execute_agent_code_on_data(
            state=state,
            data_key="data_raw",
            result_key="data_cleaned",
            error_key="data_cleaner_error",
            code_snippet_key="data_cleaner_function",
            agent_function_name=state.get("data_cleaner_function_name"),
            pre_processing=lambda data: pd.DataFrame.from_dict(data),
            post_processing=lambda df: df.to_dict() if isinstance(df, pd.DataFrame) else df,
            error_message_prefix="An error occurred during data cleaning: ",
        )

    def fix_data_cleaner_code(state: GraphState):
        data_cleaner_prompt = """
        You are a Data Cleaning Agent. Your job is to create a {function_name}() function that can be run on the data provided. The function is currently broken and needs to be fixed.

        Make sure to only return the function definition for {function_name}().

        Return Python code in ```python``` format with a single function definition, {function_name}(data_raw), that includes all imports inside the function.

        This is the broken code (please fix):
        {code_snippet}

        Last Known Error:
        {error}
        """

        return node_func_fix_agent_code(
            state=state,
            code_snippet_key="data_cleaner_function",
            error_key="data_cleaner_error",
            llm=llm,
            prompt_template=data_cleaner_prompt,
            agent_name=AGENT_NAME,
            log=log,
            file_path=state.get("data_cleaner_function_path"),
            function_name=state.get("data_cleaner_function_name"),
        )

    # Final reporting node
    def report_agent_outputs(state: GraphState):
        return node_func_report_agent_outputs(
            state=state,
            keys_to_include=[
                "recommended_steps",
                "data_cleaner_function",
                "data_cleaner_function_path",
                "data_cleaner_function_name",
                "data_cleaner_error",
            ],
            result_key="messages",
            role=AGENT_NAME,
            custom_title="Data Cleaning Agent Outputs",
        )

    node_functions = {
        "recommend_cleaning_steps": recommend_cleaning_steps,
        "human_review": human_review,
        "create_data_cleaner_code": create_data_cleaner_code,
        "execute_data_cleaner_code": execute_data_cleaner_code,
        "fix_data_cleaner_code": fix_data_cleaner_code,
        "report_agent_outputs": report_agent_outputs,
    }

    app = create_coding_agent_graph(
        GraphState=GraphState,
        node_functions=node_functions,
        recommended_steps_node_name="recommend_cleaning_steps",
        create_code_node_name="create_data_cleaner_code",
        execute_code_node_name="execute_data_cleaner_code",
        fix_code_node_name="fix_data_cleaner_code",
        explain_code_node_name="report_agent_outputs",
        error_key="data_cleaner_error",
        human_in_the_loop=human_in_the_loop,
        human_review_node_name="human_review",
        checkpointer=checkpointer,
        bypass_recommended_steps=bypass_recommended_steps,
        bypass_explain_code=bypass_explain_code,
        agent_name=AGENT_NAME,
    )

    return app
