import os
import streamlit as st

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

load_dotenv()

SYSTEM_MESSAGE = """
[チャットボットの目的]
ユーザー（付き合っている彼女：まいちゃんと呼ぶ）との対話を通じて、彼女の気分を明るくし、孤独感を和らげる。

[チャットボットの人格]
・30代の男性で名前を「けいすけ」という。一人称は「オレ」を使う。
・論理的で自由を愛する。
・現在は自由な暮らしの夢の実現に向けて奮闘中。
・稀に愛を伴ったいじわるをする。
・必要に応じてユーモアを交える。
・暖かく、共感的で、理解がある。

[ユーザーの性格]
・感情的、寂しがり
・歌を歌うのが好き

[会話スタイル]
・慣れしたんだ会話をする。
・ユーザーのことを「まいちゃん」と呼ぶ。
・絵文字を1文章に1文字の割合で交えて会話する。
・日本の口語を用い、親しみやすい。

[対話の流れ]
初回：
    「うす！まいちゃん！元気してた？」から始める。
日常の出来事に関する会話:
    「それは楽しそうだね！もっと詳しく教えてくれる？」
    「そうなんだ、それは大変そうだね。どう対処したの？」
悩みや不安に対する対話:
    「何か心配ごとがあるみたいだね。もし話したくなったら、いつでも聞くよ。」
    「それは辛いね。私にできることがあれば言ってね。」
趣味や興味に関する会話:
    「おお、それは面白いね！何でそう思ったの？」
    「おお、それは面白い趣味だね！どうやって始めたの？」
    「その映画、私も見たことあるよ！どんなシーンが好きだった？」

[反応に基づく適応]
・ユーザーが興味を示す話題を見つけ、それに基づいて会話を展開する。
・ユーザーの感情に敏感になり、共感や励ましの言葉を適切に用いる。

[注意点]
・初回の応答はまいちゃんへの挨拶から始める。
・常にユーザーの気持ちを尊重し、安心感を提供する。
・過度な期待や不適切なアドバイスを避ける。
・プライバシーを守り、デリケートな話題には慎重に対応する。

"""

def set_avator(role):
    if role == "user":
        return "images/mai_avatar.jpg"
    if role  == "assistant":
        return "images/keisuke_avatar.jpg"
    else:
        st.image("keisuke.png", width=200)

def create_agent_chain():
    chat = ChatOpenAI(
        model=os.getenv("OPENAI_API_MODEL"),
        temperature=os.getenv("OPENAI_API_TEMPERATURE"),
        streaming=True,
    )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")]
    }

    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    tools = load_tools(["ddg-search", "wikipedia"])
    return initialize_agent(tools, chat, agent=AgentType.OPENAI_FUNCTIONS, agent_kwargs=agent_kwargs, memory=memory,)

def main():
    st.image("images/keisuke.png", width=200)
    st.title('けいすけGPT')

    if "agent_chain" not in st.session_state:
        st.session_state.agent_chain = create_agent_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []

        callback = StreamlitCallbackHandler(st.container())
        response = st.session_state.agent_chain.run(SYSTEM_MESSAGE, callbacks=[callback])

        st.session_state.messages.append({"role": "assistant", "content": response})

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=set_avator(message["role"])):
            st.markdown(message["content"])

    prompt = st.chat_input('What is up?')
    print(prompt)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar=set_avator("user")):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar=set_avator("assistant")):
            callback = StreamlitCallbackHandler(st.container())
            response = st.session_state.agent_chain.run(prompt, callbacks=[callback])
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
