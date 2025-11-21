from langchain_tavily.tavily_search import TavilySearch
import asyncio
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, AnyMessage
from dotenv import load_dotenv
import os

@tool
def search_web(query: str) -> str:
    """用于搜索网页信息

    Args:
        query: str 需要搜索的信息
    """
    tavily_search=TavilySearch(max_results=3)
    try:
        result = tavily_search.invoke(query)
        return str(result["results"])
    except Exception as exc:
        return f"Search error: {str(exc)}"

@tool
def get_weather(city: str) -> str:
    """查询某个城市的天气状况

    Args:
        city: str 城市名（使用英文）
    """
    return f"{city} ☀️ 25 度"

class TavilyAgent:
    def __init__(self):
        self.api_key = os.getenv("API_KEY", "")
        self.base_url = os.getenv("BASE_URL", "")
        self.model_name = os.getenv("MODEL_NAME", "")
        if not self.api_key:
            raise ValueError("❌ 没有找到API_KEY需要配置 .env 文件")
        self.tool_by_name={"search_web": search_web, "get_weather": get_weather}
        self.llm = init_chat_model(
            model=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key
        ).bind_tools([search_web, get_weather])
    

    
    def chat(self, messages):
        response = self.llm.invoke(messages)
        messages.append(response)
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool = self.tool_by_name.get(tool_call["name"], None)
                if tool:
                    observation = tool.invoke(tool_call['args'])
                messages.append(ToolMessage(content=observation, tool_call_id=tool_call['id']))
            response = self.llm.invoke(messages)
            messages.append(response)
        return messages
    
    def chat_loop(self):
        print("Agent 🤖 客户端已启动！输入 quit 退出程序")
        messages = []
        while True:
            try:
                query = input("你：")
                if query.lower().endswith("quit"):
                    break
                messages.append(HumanMessage(content=query))
                messages = self.chat(messages)
                print(messages)
                response = messages[-1]
                print(f"🤖：{response.content}")
                messages.append(response)
            except Exception as exc:
                print(f"\n⚠️ 发生错误: {str(exc)}")



def main():
    load_dotenv()
    agent = TavilyAgent()
    agent.chat_loop()

if __name__=="__main__":
    main()
                

        

