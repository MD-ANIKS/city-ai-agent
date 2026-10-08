from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 


search_tool = TavilySearchResults(max_result = 2)

llm = ChatMistralAI(model="labs-leanstral-1-5")

prompt = ChatPromptTemplate.from_template(
  """
  You are a helpful assistant
  Summarize the following news into clear bullet points
  {news}
"""
)


chain = prompt | llm | StrOutputParser()

query = search_tool.run("Latest AI news of 2026")

result = chain.invoke({
  "news": query
})

print(result)

print(search_tool.description)
print(search_tool.name + "\n")
print(search_tool.args)