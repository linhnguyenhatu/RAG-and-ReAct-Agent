from langchain.llms.ollama import Ollama
# Initialize the OllamaEmbeddings object

prompt = "How AI will change transportation?"
model = Ollama(model="deepseek-r1:7b")
response_text = model.invoke(prompt)
print(response_text)