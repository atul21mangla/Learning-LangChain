import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

model = init_chat_model("groq:llama-3.3-70b-versatile")

@tool
def get_weather(location:str)->str:
    """ Get the weather at a particular location"""
    return f"It's sunny in {location}."

model_with_tools = model.bind_tools([get_weather])

message = [{"role":"user","content":"What's weather in Banglore"}]
ai_msg = model_with_tools.invoke(message)
message.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    print(f"Executing tool call: {tool_call}")
    tool_result = get_weather.invoke(tool_call)
    print(f"Tool result: {tool_result}")
    message.append(tool_result)

print("Messages before final invoke:")
print(message)

final_response = model_with_tools.invoke(message)
print("Final response:")
print(final_response)
print("Final content:")
print(repr(final_response.content))
