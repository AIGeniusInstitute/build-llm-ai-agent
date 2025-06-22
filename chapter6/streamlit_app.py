import streamlit as st
import os
from data_agent import agent_executor

st.title("📊 数据分析智能助手")

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"], caption="分析图表", use_column_width=True)

# 接收用户输入
if prompt := st.chat_input("您好，请问有什么可以帮您分析的？"):
    # 将用户消息添加到聊天记录
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用agent获取回复
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            response = agent_executor.invoke({"input": prompt})
            response_content = response.get("output", "")
            st.markdown(response_content)

            # 检查是否生成了图片
            image_path = "sales_trend.png"
            if os.path.exists(image_path):
                st.image(image_path, caption="分析图表", use_container_width=True)
                # 将图片和回复一起存入session
                st.session_state.messages.append({"role": "assistant", "content": response_content, "image": image_path})
            else:
                st.session_state.messages.append({"role": "assistant", "content": response_content})