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

  print("DEBUG:", data)

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

tools = {
    "get_weather" : get_weather,
    "get_news" : get_news
}

# bind
llm_with_tool = llm.bind_tools([get_weather, get_news])


# Agent Loop 
messages = []

print("City Intelligence System")
print("Type Exit to quit")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    messages.append(HumanMessage(content=user_input))

    while True:
        result = llm_with_tool.invoke(messages)
        messages.append(result)

        #if tool is required
        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']

                #HUMAN_IN THE LOOP
                confirm = input(f"Agent wants to call {tool_name} Approve (yes/no): ")

                if confirm.lower() != "yes":
                    print("tool call denied and i cannot get the latest information")
                    break

                # execute tool
                tool_result = tools[tool_name].invoke(tool_call)

                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id = tool_call['id']
                ))

            continue

        else:
            print(result.content)
            break