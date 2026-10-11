from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-6-luna')

# Create state
class BlogState(TypedDict):
    title: str
    outline: str
    content: str

# Create node
def create_outline(state: BlogState) -> BlogState :
    title = state['title']

    prompt = f'Generate a detailed outline for a blog on the topic - {title}'
    outline = model.invoke(prompt).content

    state['outline'] = outline

    return state


def create_blog(state: BlogState) -> BlogState :
    title = state['title']
    outline = state['outline']

    prompt = f'Write a detailed blog on the title - {title} using the following outline \n {outline}'
    content = model.invoke(prompt).content

    state['content'] = content

    return state


# Create graph
graph = StateGraph(BlogState)

# Add Node
graph.add_node('create_outline', create_outline)
graph.add_node('create_blog', create_blog)

# Add edge
graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog', END)

# Compile
workflow = graph.compile()

# Execute

initial_state = {'title': 'Rise of AI in India and also what is the current thinking of people about AI as well?'}
final_state = workflow.invoke(initial_state)
print(final_state['outline'])