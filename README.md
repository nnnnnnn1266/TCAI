# TCAI: A Domain-Specific AI Assistant for Turtle Care
![畫畫](https://github.com/user-attachments/assets/f37dd31d-4b8f-488c-a797-ee5aeea21a70)

### 🔗 Google Colab 筆記本
* **模型微調 (Fine-tuning):** [點此開啟 Colab 連結](https://colab.research.google.com/drive/1ObUhsEMFp2aHvy-fSGOtUY84A9JCBVH-)
* **指標評估 (Metrics):** [點此開啟 Colab 連結](https://colab.research.google.com/drive/1IuupF9GvOvMlW0m2Eq3U08EP-ZtQMotB#scrollTo=lx7TZQ6HXM4r)

TCAI (Turtle Care AI) 是一個專門為烏龜養殖與照護設計的領域特定人工智慧助手。本專案結合了大型語言模型（LLM）與檢索增強生成（RAG）概念，旨在為飼主、執業者和研究人員提供準確、可靠且易於獲取的烏龜照護知識。

📖 專案背景</br>
雖然通用型 LLM（如 GPT-4）功能強大，但在海洋生物與特定動物照護等專業領域中，常會出現資訊不足或「幻覺」現象。TCAI 透過以下技術改善此問題：</br>
1.領域語料庫建構：涵蓋物種特徵、棲息環境、營養需求及疾病管理。</br>
2.模型微調：採用 Low-Rank Adaptation (LoRA) 與 Supervised Fine-Tuning (SFT)。</br>
3.量化技術：使用 4-bit 量化以提升部署效率。</br>
 
🛠️ 技術架構 </br>
本專案主要使用了 Unsloth 框架進行 Llama 3.1 (8B) 的快速微調，並開發了完整的評估系統。</br>
核心檔案說明:</br>
`turtle_llama3_1_(8b).py`: 使用 Unsloth 進行模型微調的腳本，包含數據預處理與訓練配置。</br>
`所有指標.py`: 多維度評估工具，計算 F1-score、ROUGE-L、Semantic Similarity 以及 BERTScore。</br>
`turtle1QA.csv`: 專案核心語料庫，包含專業的烏龜照護問答。</br>
`LLM_Evaluation_Metrics.ipynb`: 評估指標的視覺化實驗手冊。</br>

📊 評估指標 </br>
為了確保回答的專業性，我們對比了多個模型（如 Qwen2, Llama3.1, DeepSeek-R1 等），主要衡量指標包括：</br>
指標說明: </br>
1. F1-Score,衡量模型回答與標準答案的詞彙重疊度。</br>
2. ROUGE-L,評估文本生成的流暢度與內容保留。</br>
3. Semantic Similarity,使用 SBERT 計算語義相近程度。</br>
4. BERTScore,利用預訓練模型嵌入向量評估語義一致性。</br>

🚀 快速上手</br>
1. 環境安裝</br>
`pip install torch torchvision torchaudio`</br>
`pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"` </br>
`pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes` </br>
`pip install rouge-score sentence-transformers bert-score`</br>
`pip install -r requirements.txt`</br>

2. 模型微調 </br>
執行` turtle_llama3_1_(8b).py `開始訓練：</br>
#載入基礎模型 </br>
`from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained( 
    model_name = "unsloth/meta-llama-3.1-8b-bnb-4bit",
    max_seq_length = 2048, </br>
    load_in_4bit = True,</br>
) 
#開始微調... 
`
3. 性能測試 </br>
執行評估腳本：</br>
`python 所有指標.py`

📊 資料集 (目前有291筆) </br>
我們構建了一個涵蓋四大領域的高質量烏龜照護資料庫 ：</br>
1.品種特性：體型、生長速度、壽命與行為模式 。</br>
2.飼養環境：水質管理、溫度控制、UVB 燈光需求與棲息地佈置 。</br>
3.飲食習性：營養比例、推薦/禁忌食物及維生素補充 。</br>
4.常見疾病：呼吸道感染、眼部發炎、軟殼症及寄生蟲防治建議 。</br>

實驗表格: </br>

| Model | F1-score (%) | Semantic Similarity (%) | BERTScore (%) | ROUGE-L (%) |
| :--- | :---: | :---: | :---: | :---: |
| **TCAI** | **77.36%** | 93.12% | **77.14%** | **68.83%** |
| Qwen2-7B | 75.92% | **94.64%** | 74.20% | 67.06% |
| LLaMA3.1-8B | 77.32% | 86.91% | 74.59% | 66.67% |
| DeepSeek-R1-Distill-Llama-8B | 76.84% | 92.85% | 74.95% | 66.17% |
| DeepSeek-R1-Distill-Qwen-7B | 76.03% | 92.61% | 74.21% | 66.04% |

**Table 1. Performance comparison of TCAI and baseline models.**

<img width="1564" height="906" alt="image" src="https://github.com/user-attachments/assets/a5d06b17-53cc-452a-b9e5-7a75a388d776" />





//

檢索增強生成 (RAG)：結合向量數據庫，即時檢索相關知識片段，確保回答具備事實依據 。

介面: Streamlit </br>
檢索技術:Embedding models mxbai-embed-large</br>
模型:llama3.1</br>
向量庫: ChromaDB PersistentClient（本地持久化）</br>

==建虛擬環境==</br>
python版本3.12</br>
cd C:\Users\YourName\YourProject</br>
python -m venv myenv(環境名稱)</br>
myenv(環境名稱)\Scripts\activate</br>

==安裝步驟== </br>
pip install requests </br>
ollama pull mxbai-embed-large (安裝嵌入模型) </br>
ollama pull meta-llama/Meta-Llama-3.1(安裝llama3模型) </br>
pip install chromadb安裝chromadb(向量庫) </br>
pip install streamlit </br>
pip install ollama </br>
pip install openpyxl </br>

==開啟== </br>
**記得先打開ollma </br>
進入環境:myenv\Scripts\activate </br>
啟動: streamlit run temp.py </br>

==評估腳本==</br>
`python 所有指標.py --input turtle1QA.xlsx --output model_scores.xlsx`</br>

==RAG 參數調整==</br>
可在側邊欄調整 `Top-K 文件數` 與 `最大距離閾值`，當檢索不到有效片段時可放寬閾值；並可查看歷史對話與每次檢索的相關分數。</br>

![image](https://github.com/user-attachments/assets/1c311b3f-7bfe-4e75-9742-4e9c2ba04f15)
