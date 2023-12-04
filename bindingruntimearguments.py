import os

from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

os.environ["OPENAI_API_KEY"] = "<open_api_key>"

prompt = ChatPromptTemplate.from_messages(
    [("system",
      "Write out the following equation using algebraic symbols then solve it. Use the "
      "format\n\nEQUATION:...\nSOLUTION:...\n\n",
      ),
     ("human", "{equation_statement}"),
     ]
)

model = ChatOpenAI(temperature=0)

runnable = (
        {"equation_statement": RunnablePassthrough()} | prompt | model | StrOutputParser()
)

print(runnable.invoke("x raised to the third plus 7 equals 12"))

"""
Output: 
EQUATION: x^3 + 7 = 12

SOLUTION:
To solve the equation, we need to isolate the variable x.

Subtracting 7 from both sides of the equation:
x^3 = 12 - 7
x^3 = 5

Taking the cube root of both sides:
∛(x^3) = ∛5
x = ∛5

Therefore, the solution to the equation x^3 + 7 = 12 is x = ∛5.
"""

"""
Making use of stopwords
"""

runnable = (
        {"equation_statement": RunnablePassthrough()}
        | prompt
        | model.bind(stop="SOLUTION")
        | StrOutputParser()
)

print(runnable.invoke("x raised to the fourth plus seven equals 12"))

"""
Output: 
EQUATION: x^4 + 7 = 12
"""

function = {
    "name": "solver",
    "description": "Formulates and solves an equation",
    "parameters": {
        "type": "object",
        "properties": {
            "equation": {
                "type": "string",
                "description": "The algebraic expression of the equation",
            },
            "solution": {
                "type": "string",
                "description": "The solution to the equation",
            },
        },
        "required": ["equation", "solution"],
    },
}

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Write out the following equation using algebraic symbols then solve it."),
        ("human", "{equation_statement}"),
    ]
)

model = ChatOpenAI(model="gpt-4", temperature=0).bind(
    function_call={"name": "solver"}, functions=[function]
)

chain = {"equation_statement": RunnablePassthrough()} | prompt | model
print(chain.invoke("x raised to the third plus 7 equals 12"))

"""
Output:
content='' additional_kwargs={'function_call': {'arguments': '{\n"equation": "x^3 + 7 = 12",\n"solution": "x = ∛5"\n}', 'name': 'solver'}}
"""
