import pandas as pd
import streamlit as st
import ollama
import chromadb

# ===== 基本設定 =====
COLLECTION_NAME = "demodocs"
CHROMA_PATH = ".chroma"
DATA_PATH = "烏龜問題集測試rag2.xlsx"
EMBEDDING_MODEL = "mxbai-embed-large"
CHAT_MODEL = "llama3.1:latest"


def create_client() -> chromadb.PersistentClient:
    """建立可持久化的 ChromaDB client。"""
    return chromadb.PersistentClient(path=CHROMA_PATH)


def initialize() -> None:
    """初始化資料庫與 Streamlit session 狀態。"""
    client = create_client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 只在首次建立 collection（沒有資料）時建立向量索引
    if collection.count() == 0:
        setup_database(collection)

    st.session_state.collection = collection

    # 初始化歷史對話
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def setup_database(collection) -> None:
    """讀取 Excel 問題集，產生 embedding 後寫入 ChromaDB。"""
    try:
        documents = pd.read_excel(DATA_PATH, header=None)
    except Exception as exc:
        raise RuntimeError(f"讀取知識庫檔案失敗：{DATA_PATH}") from exc

    for index, content in documents.iterrows():
        text = str(content[0])
        try:
            response = ollama.embeddings(prompt=text, model=EMBEDDING_MODEL)
            collection.add(ids=[str(index)], embeddings=[response["embedding"]], documents=[text])
        except Exception as exc:
            raise RuntimeError("建立向量索引失敗，請確認 Ollama 服務與 embedding 模型可用") from exc


def render_history() -> None:
    """渲染歷史對話。"""
    if not st.session_state.chat_history:
        return

    st.subheader("歷史對話")
    for idx, item in enumerate(reversed(st.session_state.chat_history), start=1):
        with st.expander(f"對話 {len(st.session_state.chat_history) - idx + 1}"):
            st.markdown("**你問：**")
            st.write(item["question"])
            st.markdown("**系統回答：**")
            st.write(item["answer"])


def main() -> None:
    """Streamlit 主流程：顯示介面並處理使用者輸入。"""
    st.set_page_config(page_title="Turtle 知識問答", page_icon="🐢")
    st.title("歡迎來到Turtle知識問答")

    # 可調檢索參數
    with st.sidebar:
        st.header("檢索參數")
        top_k = st.slider("Top-K 文件數", min_value=1, max_value=10, value=3, step=1)
        max_distance = st.slider("最大距離閾值（越小越嚴格）", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        if st.button("清除歷史對話"):
            st.session_state.chat_history = []
            st.success("已清除歷史對話")

    try:
        initialize()
    except Exception as exc:
        st.error(f"初始化失敗：{exc}")
        st.stop()

    user_input = st.text_area("您想問什麼？", "")

    if st.button("送出"):
        if user_input:
            handle_user_input(user_input, st.session_state.collection, top_k, max_distance)
        else:
            st.warning("請輸入問題！")

    render_history()


def handle_user_input(user_input: str, collection, top_k: int, max_distance: float) -> None:
    """執行檢索 + 生成（RAG）並將結果顯示在畫面上。"""
    try:
        response = ollama.embeddings(prompt=user_input, model=EMBEDDING_MODEL)
        results = collection.query(query_embeddings=[response["embedding"]], n_results=top_k)
    except Exception as exc:
        st.error(f"檢索失敗：{exc}")
        return

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # 依距離閾值過濾，並建立可視化分數
    filtered_pairs = []
    for doc, dist in zip(docs, distances):
        if dist <= max_distance:
            score = 1 / (1 + dist)  # 將 distance 轉為 0~1 區間的直覺分數
            filtered_pairs.append({"doc": doc, "distance": dist, "score": score})

    if not filtered_pairs:
        st.warning("找不到足夠相關的知識片段，請放寬距離閾值或換個問法。")
        return

    filtered_docs = [x["doc"] for x in filtered_pairs]

    try:
        output = ollama.generate(
            model=CHAT_MODEL,
            prompt=f"Using this data: {filtered_docs}. Respond to this prompt and use Chinese: {user_input}",
        )
    except Exception as exc:
        st.error(f"生成回答失敗：{exc}")
        return

    answer = output["response"]

    st.text("回答：")
    st.write(answer)

    # 保存歷史對話
    st.session_state.chat_history.append({"question": user_input, "answer": answer})

    # 顯示參考片段與檢索分數
    with st.expander("檢索到的參考片段（含分數）"):
        score_df = pd.DataFrame(
            [
                {
                    "片段": idx,
                    "距離": round(item["distance"], 4),
                    "相關分數": round(item["score"], 4),
                }
                for idx, item in enumerate(filtered_pairs, start=1)
            ]
        )
        st.dataframe(score_df, use_container_width=True)

        for idx, item in enumerate(filtered_pairs, start=1):
            st.markdown(f"**片段 {idx}**")
            st.write(item["doc"])


if __name__ == "__main__":
    main()
