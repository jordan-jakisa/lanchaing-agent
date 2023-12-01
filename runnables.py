
"""
Creating a simple PromptTemplate + ChatModel chain
"""
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

chain = prompt | model
chain_input_schema = chain.input_schema.schema()
prompt_input_schema = prompt.input_schema.schema()
model_input_schema = model.input_schema.schema()

chain_output_schema = chain.output_schema.schema()
prompt_output_schema = prompt.output_schema.schema()
model_output_schema = model.output_schema.schema()

print(chain_input_schema)

for s in chain.stream({"topic": "bears"}):
    print(s.content, end="", flush=True)

chain.invoke({"topic": "bears"})
# Output: AIMessage(content="Why don't bears wear shoes?\n\nBecause they already have bear feet!")

chain.batch([{"topic": "bears"}, {"topic": "cats"}])
"""
Output:     [AIMessage(content="Why don't bears wear shoes?\n\nBecause they have bear feet!"),
     AIMessage(content="Why don't cats play poker in the wild?\n\nToo many cheetahs!")]
"""

chain1 = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
chain2 = (
    ChatPromptTemplate.from_template("write a short (2 line) poem about {topic}")
    | model
)

combined = RunnableParallel(joke=chain1,bears=chain2)
combined.invoke({"topic":"bears"})
"""
Output:
        CPU times: user 167 ms, sys: 921 µs, xtotal: 168 ms
    Wall time: 1.56 s
    {'joke': AIMessage(content="Why don't bears wear shoes?\n\nBecause they already have bear feet!"),
     'poem': AIMessage(content="Fierce and wild, nature's might,\nBears roam the woods, shadows of the night.")}
"""


"""
Output:
    CPU times: user 507 ms, sys: 125 ms, total: 632 ms
    Wall time: 1.49 s

    [{'joke': AIMessage(content="Why don't bears wear shoes?\n\nBecause they already have bear feet!"),
      'poem': AIMessage(content="Majestic bears roam,\nNature's wild guardians of home.")},
     {'joke': AIMessage(content="Sure, here's a cat joke for you:\n\nWhy did the cat sit on the computer?\n\nBecause it wanted to keep an eye on the mouse!"),
      'poem': AIMessage(content='Whiskers twitch, eyes gleam,\nGraceful creatures, feline dream.')}]
"""
