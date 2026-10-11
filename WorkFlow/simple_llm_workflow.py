from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-6-luna")


# create state
class LLMState(TypedDict):
    question: str
    answer: str


# create node
def llm_qa(state: LLMState) -> LLMState:
    question = state['question']

    prompt = f'Answer the following question {question}'
    answer = model.invoke(prompt).content
    state['answer'] = answer
    return state


# create graph
graph = StateGraph(LLMState)

# add nodes
graph.add_node('llm_qa', llm_qa)

# add edges
graph.add_edge(START, 'llm_qa')
graph.add_edge('llm_qa', END)

# compile
workflow = graph.compile()

# execute
initial_state = {'question': 'How far is the moon from the earth?'}

final_state = workflow.invoke(initial_state)

print(final_state)