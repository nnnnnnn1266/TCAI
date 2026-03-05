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


def initialize() -> None:
    """初始化資料庫與 Streamlit session 狀態。"""
    client = create_client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        setup_database(collection)

    st.session_state.collection = collection

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def get_rag_answer(user_input: str, collection, top_k: int, max_distance: float) -> tuple[str, list[dict]]:
    """回傳 RAG 生成結果與檢索片段資訊。"""
    response = ollama.embeddings(prompt=user_input, model=EMBEDDING_MODEL)
    results = collection.query(query_embeddings=[response["embedding"]], n_results=top_k)

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    filtered_pairs = []
    for doc, dist in zip(docs, distances):
        if dist <= max_distance:
            score = 1 / (1 + dist)
            filtered_pairs.append({"doc": doc, "distance": dist, "score": score})

    if not filtered_pairs:
        raise ValueError("找不到足夠相關的知識片段，請放寬距離閾值或換個問法。")

    filtered_docs = [x["doc"] for x in filtered_pairs]
    output = ollama.generate(
        model=CHAT_MODEL,
        prompt=f"Using this data: {filtered_docs}. Respond to this prompt and use Chinese: {user_input}",
    )
    return output["response"], filtered_pairs


def render_message_block(record: dict, idx: int) -> None:
    """渲染一筆對話卡片。"""
    st.markdown(f"### 對話 {idx}")
    with st.chat_message("user"):
        st.write(record["question"])
    with st.chat_message("assistant"):
        st.write(record["answer"])

    with st.expander("檢索來源（含分數）"):
        score_df = pd.DataFrame(
            [
                {
                    "片段": i,
                    "距離": round(item["distance"], 4),
                    "相關分數": round(item["score"], 4),
                }
                for i, item in enumerate(record["sources"], start=1)
            ]
        )
        st.dataframe(score_df, use_container_width=True)

        for i, item in enumerate(record["sources"], start=1):
            st.markdown(f"**片段 {i}**")
            st.write(item["doc"])


def main() -> None:
    """Streamlit 主流程：聊天式 RAG 介面。"""
    st.set_page_config(page_title="Turtle 知識問答", page_icon="🐢", layout="wide")
    st.title("🐢 Turtle 知識問答介面")
    st.caption("RAG + ChromaDB + Ollama（聊天式 UI）")

    with st.sidebar:
        st.header("⚙️ 檢索設定")
        top_k = st.slider("Top-K 文件數", min_value=1, max_value=10, value=3, step=1)
        max_distance = st.slider("最大距離閾值（越小越嚴格）", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

        st.divider()
        st.subheader("🧹 對話管理")
        if st.button("清除歷史對話", use_container_width=True):
            st.session_state.chat_history = []
            st.success("已清除歷史對話")

    try:
        initialize()
    except Exception as exc:
        st.error(f"初始化失敗：{exc}")
        st.stop()

    user_input = st.chat_input("請輸入你的烏龜照護問題…")

    if user_input:
        with st.spinner("正在檢索與生成回答..."):
            try:
                answer, sources = get_rag_answer(user_input, st.session_state.collection, top_k, max_distance)
            except Exception as exc:
                st.error(f"處理失敗：{exc}")
            else:
                st.session_state.chat_history.append(
                    {
                        "question": user_input,
                        "answer": answer,
                        "sources": sources,
                    }
                )

    if not st.session_state.chat_history:
        st.info("尚無對話，請先輸入一個問題。")
        return

    for idx, record in enumerate(st.session_state.chat_history, start=1):
        render_message_block(record, idx)


if __name__ == "__main__":
    main()
