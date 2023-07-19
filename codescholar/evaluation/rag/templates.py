GPT_FIND_API = """
Return an appropriate python API for the following query. Only return APIs from the following libraries: {libs}.
Only return the generic API name, with no explanations, no parentheses, and no arguments. For example:
Query: How do I find the mean of a numpy array?
"\"\"
np.mean
"\"\"

Query: filter rows in pandas starting with alphabet 'm' using regular expression
"\"\"
df.filter
"\"\"

Query: get index of rows in column 'lorem'
"\"\"
df.loc
"\"\"

Query: {query}
\"\"\"
"""

GPT_NL2CODE = """
Complete the following code snippet to solve the task described in the comment.
Only complete the code in the opened code block and don't print outputs or explanations.

{prompt}
"""

GPT_NL2CODE_FEWSHOT = """
Complete the following code snippet to solve the task described in the comment.
Only complete the code in the opened code block and don't print outputs or explanations.
Here are a few examples:
{examples}

{prompt}
"""

GPT_NL2CODE_TASK = """
Query: {query}
Code:
\"\"\"
{solution}"""
