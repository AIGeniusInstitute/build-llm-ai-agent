import streamlit as st

st.title("AI Agent Demo")
user_input = st.text_input("请输入问题：")
if user_input:
    st.write(f"Agent回复：{user_input}")