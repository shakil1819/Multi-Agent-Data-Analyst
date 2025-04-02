# Parsers

import re

from langchain_core.output_parsers import BaseOutputParser


# Python Parser for output standardization
class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        def extract_python_code(text):
            python_code_match = re.search(r"```python(.*?)```", text, re.DOTALL)
            if python_code_match:
                python_code = python_code_match.group(1).strip()
                return python_code
            python_code_match = re.search(r"python(.*?)'", text, re.DOTALL)
            if python_code_match:
                python_code = python_code_match.group(1).strip()
                return python_code
            return None

        python_code = extract_python_code(text)
        if python_code is not None:
            return python_code
        # Assume ```sql wasn't used
        return text


# SQL Parser for output standardization
class SQLOutputParser(BaseOutputParser):
    def parse(self, text: str):
        def extract_sql_code(text):
            sql_code_match = re.search(r"```sql(.*?)```", text, re.DOTALL)
            sql_code_match_2 = re.search(r"SQLQuery:\s*(.*)", text)
            if sql_code_match:
                sql_code = sql_code_match.group(1).strip()
                return sql_code
            if sql_code_match_2:
                sql_code = sql_code_match_2.group(1).strip()
                return sql_code
            sql_code_match = re.search(r"sql(.*?)'", text, re.DOTALL)
            if sql_code_match:
                sql_code = sql_code_match.group(1).strip()
                return sql_code
            return None

        sql_code = extract_sql_code(text)
        if sql_code is not None:
            return sql_code
        # Assume ```sql wasn't used
        return text
