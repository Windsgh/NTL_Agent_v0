import streamlit as st
import pandas as pd
from NTL_Agent_v0 import graph
from session_manager import init_session_state, reset_session, save_history, load_history, export_history


# 1. 页面配置部分
st.set_page_config(page_title="Nighttime Light Agent", page_icon=":robot:", layout="wide")
# """
# 设置Streamlit应用的页面配置。这里设置了页面标题、页面图标和布局。
# page_title是浏览器标签页上的标题，page_icon是页面图标，layout="wide"是设置页面为宽布局，适用于较大的屏幕。
# """

# 2. CSS样式定义
# """
# CSS样式：自定义了聊天界面的样式，包括用户和机器人消息的背景颜色、头像的样式等。
# 通过这种方式，可以定制聊天界面的外观，使其更加美观和符合应用主题。
# """
st.markdown(
    """<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
.stDeployButton {
            visibility: hidden;
        }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding: 2rem 4rem 2rem 4rem;
}

.st-emotion-cache-16txtl3 {
    padding: 3rem 1.5rem;
}
</style>
# """,
    unsafe_allow_html=True,
)

# 3. 聊天消息模板
# """
# 聊天消息模板：定义了聊天消息的HTML模板，bot_template和user_template分别表示机器人和用户的消息。
# 通过{{MSG}}占位符，动态替换成实际的消息内容。每个消息框有一个头像和内容部分，头像是通过URL加载图片。
# """
bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn.icon-icons.com/icons2/1371/PNG/512/robot02_90810.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.shareicon.net/data/512x512/2015/09/18/103160_man_512x512.png" >
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

import io
import contextlib

import os
import streamlit as st
from PIL import Image
import io
import contextlib


def handle_userinput(user_question):
    chat_history = st.session_state.chat_history
    state = {
        "messages": [{"role": "user", "content": user_question}],
    }
    RECURSION_LIMIT = 2*15 + 1
    # 更新聊天记录，加入用户问题和AI的最新回复
    st.session_state.chat_history.append(("user", user_question))
    # 显示用户问题
    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)

    # 创建一个占位符，等待答案
    wait_placeholder = st.empty()  # Create a placeholder for the "please wait" message
    wait_placeholder.write(bot_template.replace("{{MSG}}", "Please Wait"), unsafe_allow_html=True)

    # 初始化一个变量来存储最终的答案
    final_answer = None
    selected_images = []  # List to hold selected images
    image_found_Act = False
    CSV_found_Act = False
    # 用于流式输出的显示区
    with st.container():
        st.subheader("Stream Response")
        # 使用 graph 的流式响应
        events = st.session_state.conversation.stream(
            state,
            config={
                "configurable": {"thread_id": st.session_state.thread_id},
                "recursion_limit": RECURSION_LIMIT
            },
            stream_mode="values"
        )

        # 逐个处理事件，并流式显示
        for event in events:
            message_content = event["messages"][-1]
            # 显示每个中间步骤
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                message_content.pretty_print()
            # 获取打印的内容
            printed_content = output.getvalue()
            # 关闭 StringIO 对象
            output.close()

            st.code(printed_content, language='python')

            final_answer = event["messages"][-1].content

            # Check if printed_content contains "geo_visualization_tool"
            if "png" in final_answer:
                image_found_Act = True
            if "csv" in final_answer:
                CSV_found_Act = True


        # After the stream processing ends, check if images need to be displayed
        if image_found_Act:
            image_dir = "C:\\NTL_Agent\\report\\image"
            if os.path.exists(image_dir):
                # Get all image files in the directory
                image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

                if image_files:
                    # Move the image display logic to the sidebar
                    with st.sidebar:
                        # Track the images already displayed in session_state to avoid duplicate display
                        if "displayed_images" not in st.session_state:
                            st.session_state.displayed_images = []

                        # Use checkboxes to select multiple images
                        for idx, image_file in enumerate(image_files):
                            # Skip images that have already been displayed
                            if image_file in st.session_state.displayed_images:
                                continue

                            image_path = os.path.join(image_dir, image_file)
                            image = Image.open(image_path)

                            col1, col2 = st.columns([7, 2])  # Use columns to display image and delete button
                            with col1:
                                st.image(image, caption=f"Image {idx + 1}: {image_file}", use_container_width=True)
                            with col2:
                                if st.button(f"Delete {image_file}", key=f"delete_{idx}_{image_file}"):
                                    os.remove(image_path)  # Delete the image file
                                    st.success(f"Image {image_file} deleted.")
                                    st.experimental_rerun()  # Rerun to refresh the page after deletion

                            # Add the image to the displayed list
                            st.session_state.displayed_images.append(image_file)

        if CSV_found_Act:
            # 定义 CSV 文件目录
            csv_dir = "C:\\NTL_Agent\\report\\csv"
            if os.path.exists(csv_dir):
                # 获取目录中所有的 CSV 文件
                csv_files = [f for f in os.listdir(csv_dir) if f.lower().endswith('.csv')]

                if csv_files:
                    # 在页面上显示找到的 CSV 文件
                    st.subheader("CSV Files Found:")
                    for csv_file in csv_files:
                        csv_path = os.path.join(csv_dir, csv_file)

                        try:
                            # 读取 CSV 文件并显示为表格
                            df = pd.read_csv(csv_path)
                            with st.expander(f"Preview of {csv_file}"):  # 使用 expander 展开显示 CSV 内容
                                st.write(df)

                            # 提供一个下载链接
                            csv_download_link = f'<a href="data:file/csv;base64,{df.to_csv(index=False).encode().decode()}" download="{csv_file}">Download {csv_file}</a>'
                            st.markdown(csv_download_link, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Failed to read {csv_file}: {e}")

        # Display the final answer after all responses and images are processed
        if final_answer:
            wait_placeholder.write(bot_template.replace("{{MSG}}", final_answer), unsafe_allow_html=True)
            st.session_state.chat_history.append(("assistant", final_answer))

            # ✅ 保存会话历史，确保数据持久化
            save_history(st.session_state)
# def handle_userinput(user_question):
#     chat_history = st.session_state.chat_history
#     state = {
#         "messages": [{"role": "user", "content": user_question}],
#     }
#
#     # 更新聊天记录，加入用户问题和AI的最新回复
#     st.session_state.chat_history.append(("user", user_question))
#     # 显示用户问题
#     st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
#
#     # 创建一个占位符，等待答案
#     wait_placeholder = st.empty()  # Create a placeholder for the "please wait" message
#     wait_placeholder.write(bot_template.replace("{{MSG}}", "Please Wait"), unsafe_allow_html=True)
#
#     # 初始化一个变量来存储最终的答案
#     final_answer = None
#     # 用于流式输出的显示区
#     with st.container():
#         st.subheader("Stream Response")
#         # 使用 graph 的流式响应
#         events = st.session_state.conversation.stream(
#             state,
#             config={"configurable": {"thread_id": "4"}, "recursion_limit": 10},  # 设置 recursion_limit
#             stream_mode="values"
#         )
#         # 逐个处理事件，并流式显示
#         for event in events:
#             message_content = event["messages"][-1]
#             # 显示每个中间步骤
#             if hasattr(message_content, 'pretty_print'):
#                 output = io.StringIO()
#                 with contextlib.redirect_stdout(output):
#                     message_content.pretty_print()
#                 # 获取打印的内容
#                 printed_content = output.getvalue()
#                 # 关闭 StringIO 对象
#                 output.close()
#                 st.write(printed_content, unsafe_allow_html=True)
#             else:
#                 # If pretty_print is not available, just display the content as a fallback
#                 st.write(message_content.content)
#
#             final_answer = event["messages"][-1].content
#
#     # 在流式响应完成后，更新占位符为最终答案
#     if final_answer:
#         # Use the same placeholder to replace the "please wait" with the final answer
#         wait_placeholder.write(bot_template.replace("{{MSG}}", final_answer), unsafe_allow_html=True)
#         st.session_state.chat_history.append(("assistant", final_answer))


# 8. 显示对话历史
# '''
# show_history：显示聊天历史。它遍历聊天记录并根据消息的索引判断是用户的提问还是机器人的回答。交替显示用户和机器人的消息。
# '''
def show_history():
    chat_history = st.session_state.chat_history

    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message[1]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message[1]), unsafe_allow_html=True)


# 9.主函数
# '''
# main：主函数。设置页面标题并初始化会话状态。
# 通过 Streamlit sidebar 允许用户上传文件，并对上传的文档进行处理，提取文本并存入向量存储。
# 在主容器内，展示聊天输入框和聊天历史。用户输入问题后，如果会话存在，机器人就会根据上下文进行回答。
# '''

def main():
    # 初始化会话状态
    init_session_state(st.session_state)

    st.header("**Nighttime Light Agent**")
    # 初始化会话状态
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.session_state.conversation = graph  # 无论是否上传文件，初始化为 `graph`
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ✅ 添加 Thread ID 初始化
    import uuid
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    with st.sidebar:
        # Image upload section
        st.title("Image Upload")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "gif"])

        if uploaded_image is not None:
            # Open the image with PIL
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)

        st.subheader("Session Info")
        st.write(f"**Your Thread ID:** `{st.session_state.thread_id}`")

        # 重新生成 Thread ID，清空会话
        if st.button("🔄 New Session"):
            reset_session(st.session_state)
            st.rerun()

        # 恢复指定会话
        custom_thread_id = st.text_input("Enter Existing Thread ID to Restore Session")
        if st.button("Restore Session"):
            if custom_thread_id.strip():
                if load_history(st.session_state, custom_thread_id.strip()):
                    st.success(f"Session restored to: `{custom_thread_id}`")
                    st.rerun()
                else:
                    st.warning("Session not found.")
            else:
                st.warning("Please enter a valid Thread ID.")

    with st.container():
        user_question = st.chat_input("Please Ask Something~")

    with st.container(height=550):
        show_history()  # 显示历史对话
        if user_question:
            handle_userinput(user_question)  # 处理用户输入，流式输出每个阶段


if __name__ == "__main__":
    main()

# streamlit run C:\NTL_Agent\Project_Agent\app.py
