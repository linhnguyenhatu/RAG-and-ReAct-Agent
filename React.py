
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from googlesearch import search
from bs4 import BeautifulSoup
import requests
import wikipedia
from langchain.llms.ollama import Ollama
from RAG import search_in_docs

from langchain_core.prompts import PromptTemplate

def get_react_prompt_template():
    # Get the react prompt template
    return PromptTemplate.from_template(""""

You have access to the following tools:
<tool>{tools}</tool>

<question>: {input} </question>
<thought>:{agent_scratchpad} </thought>
<action>: the action to take, should be one of [{tool_names}] </action>
<action-input>: the input to the action</action-input>
<observation>: the result of the action</observation>
(this Thought/Action/Action Input/Observation can repeat N times)
Final Answer: the final answer to the original input question (extend longer than 200 words to clarify it)
Then output the final answer and stop the process
Begin!
Question: {input}
Thought:{agent_scratchpad}
""")


def search_links(string):
  query = string
  result = search(query, num_results=10, unique=True)
  links = [j for j in result]
  return links

@tool
def search_for_material(string : str) -> list:
  """Searches for material on the google"""
  query = string
  links = search_links(query)
  count = 0
  index = 0
  material = ''
  while count < 3:
    response = requests.get(links[index])
    index += 1
    if response.status_code == 200:
      soup = BeautifulSoup(response.content, 'html.parser')
      text = soup.get_text()
      material += text
      count += 1
  material = material.replace('\n', '')
  material = material.replace('  ', ' ')
  material = material.replace('�', '')
  material = material.replace('\t', '')
  return material

def search_wikipedia_link(string):
  path = "wikipedia.com"
  result = search(string + path, num_results=2, unique=True)
  links = [j for j in result]
  return links[0]

def find_title(string):
  index = len(string) - 1
  while string[index] != '/':
    index -= 1
  return string[index + 1:]

@tool
def search_wikipedia(string):
  """Searches for material on wikipedia"""
  link = search_wikipedia_link(string)
  title = find_title(link) 
  print(title)
  try:
    result = wikipedia.page(title).content
  except wikipedia.exceptions.DisambiguationError as e:
    result = wikipedia.page(e.options[0]).content
  return result

@tool
def search_in_documents(string):
  """Searches for information in existing local documents"""
  return search_in_docs(string)

tool = [search_for_material, search_wikipedia, search_in_documents]
llm = Ollama(model="deepseek-r1:7b")
prompt = get_react_prompt_template()
agent = create_react_agent(llm, tool, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True, handle_parsing_errors=True, max_iterations=3)
agent_executor.invoke({"input" : "How AI will change transportation?"})


