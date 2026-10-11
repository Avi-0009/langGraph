from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

# sr => Strike rate, bpb => balls per boundary, bp => boundary percentage
class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int

    sr: float
    bpb: float
    bp: float
    summary: str

def calc_SR(state: BatsmanState):
    sr = (state['runs'] / state['balls']) * 100
    return {'sr': sr}


def calc_BPB(state: BatsmanState):
    bpb = state['balls'] / (state['fours'] + state['sixes'])
    return {'bpb': bpb}

def calc_BP(state: BatsmanState):
    bp = 100 * ((state['fours'] * 4) + (state['sixes'] * 6)) / state['runs']
    return {'bp': bp}

def summary(state: BatsmanState):
    summary = f"""
Strike rate - {state['sr']} \n
Balls per boundary - {state['bpb']} \n
Boundary percentage - {state['bp']} \n
"""
    return {'summary': summary}

graph = StateGraph(BatsmanState)

# Nodes
graph.add_node('calc_SR', calc_SR)
graph.add_node('calc_BPB', calc_BPB)
graph.add_node('calc_BP', calc_BP)
graph.add_node('summary', summary)

# Edges
graph.add_edge(START, 'calc_SR')
graph.add_edge('calc_SR', 'summary')

graph.add_edge(START, 'calc_BPB')
graph.add_edge('calc_BPB', 'summary')

graph.add_edge(START, 'calc_BP')
graph.add_edge('calc_BP', 'summary')

graph.add_edge('summary', END)

# Compile
workflow = graph.compile()

# Execute
initial_state = {
    'runs': 100,
    'balls': 50,
    'fours': 6,
    'sixes': 4
}

final_state = workflow.invoke(initial_state)

print(final_state)
