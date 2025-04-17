# app.py
import streamlit as st
import ui  # UIモジュール
import llm  # LLMモジュール
import database  # データベースモジュール
import metrics  # 評価指標モジュール
import data  # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME

# --- アプリケーション設定 ---
st.set_page_config(page_title="Gemma Chatbot", layout="wide")

# --- 初期化処理 ---
metrics.initialize_nltk()
database.init_db()
data.ensure_initial_data()


# --- LLMモデルのロード ---
@st.cache_resource
def load_model():
    """LLMモデルをロードする（UI表示は外で制御）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "text-generation",
        model=MODEL_NAME,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )
    return pipe


# --- アプリタイトル・説明 ---
st.title("🤖 Gemma 2 Chatbot with Feedback")
st.markdown(
    "### Chat with the powerful **Gemma LLM**, and provide feedback on responses!"
)
st.markdown("---")

# --- モデルの読み込み（スピナー付き） ---
pipe = None
with st.spinner("Gemmaモデルを読み込み中..."):
    try:
        pipe = load_model()
        st.success(f"✅ モデル '{MODEL_NAME}' の読み込みに成功しました。")
        st.info(f"使用デバイス: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    except Exception as e:
        st.error(f"❌ モデル '{MODEL_NAME}' の読み込みに失敗しました。")
        st.exception(e)
        if st.button("再試行"):
            st.cache_resource.clear()
            st.rerun()

# --- サイドバー ---
st.sidebar.title("ナビゲーション")
page = st.sidebar.radio(
    "ページ選択", ["チャット", "履歴閲覧", "サンプルデータ管理"], key="page"
)

st.sidebar.markdown("---")
st.sidebar.info("開発者: [Your Name]")

# --- メインページの切り替え ---
if page == "チャット":
    if pipe:
        ui.display_chat_page(pipe)
    else:
        st.warning("チャット機能を利用できません。モデルの読み込みが必要です。")
elif page == "履歴閲覧":
    ui.display_history_page()
elif page == "サンプルデータ管理":
    ui.display_data_page()
