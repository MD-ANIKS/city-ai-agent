# Langchain Create Agent Method - Automate

# loading all  the  libraries
from dotenv import load_dotenv
load_dotenv()

import os 
import requests

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool 
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser 
from tavily import TavilyClient
from langchain.agents import create_agent

from langchain.agents.middleware import wrap_tool_call

from rich import print

parser = StrOutputParser()

# create tools

#weather tool
@tool
def get_weather(city: str) -> str:
  """Get Current Weather of a City"""
  API_KEY = os.getenv("OPENWEATHER_API_KEY")
  url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

  response = requests.get(url)
  data = response.json()

  # print("DEBUG:", data)

  #error msg 
  if str(data.get("cod")) != "200":
    return f"Error: {data.get('message', 'Could not fetch weather')}"

  temp = data["main"]["temp"]
  desc = data["weather"][0]["description"]

  return f"Weather in {city}: {desc}, {temp}deg Celcius"



#Tavily news tool
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool 
def get_news(city : str) -> str:
  """Get Latest news about the city"""
  response = tavily_client.search(
        query=f"latest news in {city}",
        topic="news",
        max_results=3
    )

  results = response.get("results", [])

  if not results:
        return f"No recent news found for {city}."

  news = []

  for result in results:
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")

        news.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Summary: {content}"
        )

  return "\n\n".join(news)



result = get_news.invoke("Dhaka")


llm = ChatMistralAI(model="labs-leanstral-1-5")

@wrap_tool_call
def human_approval(request, handler):
    """Ask for human approval before every tool call."""
    tool_name = request.tool_call["name"]
    confirm = input(f"Agentn wants to call '{tool_name}'. Approve? (yest/no): ")

    if confirm.lower() != "yes":
        return ToolMessage(
            content="Tool call denied by user.",
            tool_call_id = request.tool_call["id"]
        )
    return handler(request)




agent = create_agent(
    llm,
    tools = [get_weather, get_news],
    system_prompt="You are a helpful city assistant",

    middleware=[human_approval]
)


print("City Agent | type exit to quit")

while True:
    user_input = input("You : ")

    if user_input.lower() == "exit":
        break 

    result = agent.invoke({
        "messages" : [{f"role": "user", "content": user_input}]
    })

    print(result['messages'][-1].content)