import pandas as pd
import streamlit as st
import ollama
import chromadb
from typing import Iterable

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


def get_rag_answer(user_input: str, collection, top_k: int, max_distance: float, candidate_k: int) -> tuple[str, list[dict]]:
    """回傳 RAG 生成結果與檢索片段資訊。"""
    response = ollama.embeddings(prompt=user_input, model=EMBEDDING_MODEL)
    # 先擴大召回，再做重排
    results = collection.query(query_embeddings=[response["embedding"]], n_results=candidate_k)

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    filtered_pairs = []
    for doc, dist in zip(docs, distances):
        if dist <= max_distance:
            score = 1 / (1 + dist)
            filtered_pairs.append({"doc": doc, "distance": dist, "score": score})

    if not filtered_pairs:
        raise ValueError("找不到足夠相關的知識片段，請放寬距離閾值或換個問法。")

    reranked_pairs = rerank_with_keywords(user_input, filtered_pairs)
    selected_pairs = reranked_pairs[:top_k]

    filtered_docs = [x["doc"] for x in selected_pairs]
    output = ollama.generate(
        model=CHAT_MODEL,
        prompt=f"Using this data: {filtered_docs}. Respond to this prompt and use Chinese: {user_input}",
    )
    return output["response"], selected_pairs


def render_message_block(record: dict, idx: int, min_hybrid_score: float) -> None:
    """渲染一筆對話卡片。"""
    st.markdown(f"### 對話 {idx}")
    with st.chat_message("user"):
        st.write(record["question"])
    with st.chat_message("assistant"):
        st.write(record["answer"])

    with st.expander("檢索來源（含分數）"):
        filtered_sources = [
            item for item in record["sources"] if item.get("hybrid", item["score"]) >= min_hybrid_score
        ]

        if not filtered_sources:
            st.warning("目前門檻下沒有可顯示的片段，請降低『最低混合分數門檻』。")
            return

        score_df = pd.DataFrame(
            [
                {
                    "片段": i,
                    "距離": round(item["distance"], 4),
                    "語意分數": round(item["score"], 4),
                    "關鍵字分數": round(item.get("lexical", 0.0), 4),
                    "混合分數": round(item.get("hybrid", item["score"]), 4),
                }
                for i, item in enumerate(filtered_sources, start=1)
            ]
        )
        st.dataframe(score_df, use_container_width=True)

        for i, item in enumerate(filtered_sources, start=1):
            st.markdown(f"**片段 {i}**")
            highlighted = _highlight_query_chars(record["question"], item["doc"])
            st.markdown(highlighted, unsafe_allow_html=True)


def _normalize_model_name(name: str) -> str:
    """正規化模型名稱，忽略大小寫與空白。"""
    return name.strip().lower()


def _extract_installed_model_names(raw_models: Iterable[dict]) -> set[str]:
    """從 ollama.list() 的結果抽出可比對名稱。"""
    names: set[str] = set()
    for model in raw_models:
        model_name = str(model.get("name", "")).strip()
        if not model_name:
            continue
        names.add(_normalize_model_name(model_name))
        # 同時加入去掉 tag 的名稱，方便比對 like llama3.1 vs llama3.1:latest
        names.add(_normalize_model_name(model_name.split(":")[0]))
    return names


def verify_ollama_ready() -> None:
    """檢查 Ollama 服務可用且所需模型已安裝。"""
    try:
        list_resp = ollama.list()
    except Exception as exc:
        raise RuntimeError(
            "無法連線到 Ollama，請先啟動 Ollama 應用程式/服務。"
        ) from exc

    raw_models = list_resp.get("models", []) if isinstance(list_resp, dict) else []
    installed = _extract_installed_model_names(raw_models)

    # 只要 exact 或無 tag 版本存在即視為可用
    missing = []
    if _normalize_model_name(EMBEDDING_MODEL) not in installed and _normalize_model_name(EMBEDDING_MODEL.split(":")[0]) not in installed:
        missing.append(EMBEDDING_MODEL)
    if _normalize_model_name(CHAT_MODEL) not in installed and _normalize_model_name(CHAT_MODEL.split(":")[0]) not in installed:
        missing.append(CHAT_MODEL)

    if missing:
        hint_cmds = "\n".join([f"ollama pull {m}" for m in missing])
        raise RuntimeError(
            "缺少必要 Ollama 模型：" + ", ".join(missing) + f"\n請先執行：\n{hint_cmds}"
        )


def _extract_keywords(text: str) -> set[str]:
    """以簡單字元集合抽取關鍵字（過濾空白與常見標點）。"""
    punct = set(" ，。！？；：,.!?;:\n\t()[]{}<>'\"`（）【】「」『』、")
    return {ch for ch in text if ch not in punct and ch.strip()}


def _keyword_overlap_score(query: str, doc: str) -> float:
    """計算 query/doc 的關鍵字重疊分數（0~1）。"""
    q = _extract_keywords(query)
    d = _extract_keywords(doc)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q)


def rerank_with_keywords(user_input: str, pairs: list[dict]) -> list[dict]:
    """向量召回後再用關鍵字重排，提升問題對齊度。"""
    reranked = []
    for item in pairs:
        lexical = _keyword_overlap_score(user_input, item["doc"])
        # 混合分數：語意(0.7) + 關鍵字(0.3)
        hybrid = item["score"] * 0.7 + lexical * 0.3
        row = dict(item)
        row["lexical"] = lexical
        row["hybrid"] = hybrid
        reranked.append(row)

    reranked.sort(key=lambda x: x["hybrid"], reverse=True)
    return reranked


def _highlight_query_chars(query: str, doc: str) -> str:
    """將文件中與問題重疊的字元以 <mark> 標註（HTML）。"""
    qset = _extract_keywords(query)
    out = []
    for ch in doc:
        if ch in qset and ch.strip():
            out.append(f"<mark>{ch}</mark>")
        else:
            out.append(ch)
    return "".join(out)


def main() -> None:
    """Streamlit 主流程：聊天式 RAG 介面。"""
    st.set_page_config(page_title="Turtle 知識問答", page_icon="🐢", layout="wide")
    st.title("🐢 Turtle 知識問答介面")
    st.caption("RAG + ChromaDB + Ollama（聊天式 UI）")

    with st.sidebar:
        st.header("⚙️ 檢索設定")
        top_k = st.slider("Top-K 文件數", min_value=1, max_value=10, value=3, step=1)
        candidate_k = st.slider("初步召回數（重排前）", min_value=top_k, max_value=30, value=max(10, top_k), step=1)
        max_distance = st.slider("最大距離閾值（越小越嚴格）", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        min_hybrid_score = st.slider("最低混合分數門檻", min_value=0.0, max_value=1.0, value=0.45, step=0.01)

        st.divider()
        st.subheader("🧹 對話管理")
        if st.button("清除歷史對話", use_container_width=True):
            st.session_state.chat_history = []
            st.success("已清除歷史對話")

    try:
        verify_ollama_ready()
        initialize()
    except Exception as exc:
        st.error(f"初始化失敗：{exc}")
        st.info("請確認：1) Ollama 服務已啟動 2) 已執行模型下載指令。")
        st.code(f"ollama pull {EMBEDDING_MODEL}\nollama pull {CHAT_MODEL}")
        st.stop()

    user_input = st.chat_input("請輸入你的烏龜照護問題…")

    if user_input:
        with st.spinner("正在檢索與生成回答..."):
            try:
                answer, sources = get_rag_answer(user_input, st.session_state.collection, top_k, max_distance, candidate_k)
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
        render_message_block(record, idx, min_hybrid_score)


if __name__ == "__main__":
    main()
