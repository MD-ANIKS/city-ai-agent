from langchain.tools import tool

@tool #decorator for creating tool
def get_greeting(name : str) -> str: #type hints
  """Generate a greeting message for a user""" #docstring
  return f"Hello {name}, Welcome to The GenAI" 

result = get_greeting.invoke({
  "name": "Anik"
})

print(result)

print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)