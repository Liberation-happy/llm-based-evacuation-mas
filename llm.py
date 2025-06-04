from langchain_community.chat_models import ChatTongyi

# 通义千问模型
tongyi_chat = ChatTongyi(
    model="qwen-max",
    temperature=0.5,
    api_key="sk-e849fd072e7648f59f69ed59a56f5ebc"
)


if __name__ == '__main__':
    answer = tongyi_chat.invoke("你好")
    print(answer.content)
