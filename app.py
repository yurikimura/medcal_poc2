import os
import streamlit as st
from langchain_community.document_loaders import PubMedLoader, ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor
import time

# Page configuration
st.set_page_config(
    page_title="Medical Diagnosis Bot",
    page_icon="🏥",
    layout="wide"
)

# App title and description
st.title("🏥 多言語対応 症状から病名を特定する医療診断ボット")
st.markdown("""
このアプリケーションは、入力された症状から考えられる病名を提案します。
PubMedとArXivから最新の医学論文を検索し、可能性のある診断を提示します。

**注意**: これは医学的アドバイスを提供するものではありません。実際の診断には必ず医師に相談してください。
""")

# Sidebar for API key
with st.sidebar:
    st.header("設定")
    api_key = st.text_input("OpenAI API キー", type="password", 
                           help="自分のOpenAI APIキーを入力してください")
    model_name = st.selectbox(
        "使用するモデル",
        ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]
    )
    st.markdown("---")
    st.markdown("## アプリについて")
    st.markdown("""
    このアプリは症状に基づいて考えられる病名を提案します。
    1. 症状を入力言語で記述
    2. 必要に応じて患者情報を追加
    3. 「診断開始」ボタンをクリック
    4. PubMedとArXivから関連する論文が検索されます
    5. AIが考えられる病名と説明を提供します
    """)


class MultilingualMedicalBot:
    def __init__(self, api_key, model_name="gpt-3.5-turbo"):
        # OpenAI APIキーの設定
        os.environ["OPENAI_API_KEY"] = api_key
        
        self.llm = ChatOpenAI(model_name=model_name)
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        # 翻訳用LLM（同じモデルを使用）
        self.translator_llm = ChatOpenAI(model_name=model_name)

    def translate_symptoms_to_english(self, symptoms, source_language="日本語"):
        """
        症状を英語に翻訳
        """
        translation_prompt = ChatPromptTemplate.from_template("""
        あなたは医学用語に精通した翻訳者です。以下の{source_language}の症状を、
        医学的に正確な英語に翻訳してください。PubMedやArXivなどの医学データベースで
        検索する際に使用するため、専門的な医学用語を使用してください。

        症状: {symptoms}

        英語翻訳（医学用語）:
        """)

        # 翻訳を実行
        translation_chain = translation_prompt | self.translator_llm
        result = translation_chain.invoke({
            "source_language": source_language,
            "symptoms": symptoms
        })

        translated = result.content.strip()
        return translated

    def _load_pubmed_docs(self, english_symptoms):
        """PubMedから文献を取得"""
        search_query = f"{english_symptoms} symptoms diagnosis"
        try:
            loader = PubMedLoader(query=search_query)
            documents = loader.load()
            return documents, len(documents)
        except Exception as e:
            st.error(f"PubMed検索エラー: {e}")
            return [], 0

    def _load_arxiv_docs(self, english_symptoms, max_docs=5):
        """ArXivから文献を取得"""
        search_query = f"{english_symptoms} symptoms diagnosis medical"

        try:
            loader = ArxivLoader(
                query=search_query,
                load_max_documents=max_docs,
                load_all_available_meta=True
            )
            documents = loader.load()
            return documents, len(documents)
        except Exception as e:
            st.error(f"ArXiv検索エラー: {e}")
            return [], 0

    def create_knowledge_base(self, symptoms, source_language="日本語", progress_bar=None):
        """
        症状に基づいてPubMedとArXivから関連文献を検索し、統合知識ベースを作成
        """
        # 症状を英語に翻訳
        if progress_bar:
            progress_bar.progress(0.1, text="症状を英語に翻訳中...")
        
        english_symptoms = self.translate_symptoms_to_english(symptoms, source_language)
        
        if progress_bar:
            progress_bar.progress(0.2, text="PubMedとArXivから文献を検索中...")

        # マルチスレッドで両方のソースから同時に取得
        with ThreadPoolExecutor(max_workers=2) as executor:
            pubmed_future = executor.submit(self._load_pubmed_docs, english_symptoms)
            arxiv_future = executor.submit(self._load_arxiv_docs, english_symptoms)

            pubmed_docs, pubmed_count = pubmed_future.result()
            arxiv_docs, arxiv_count = arxiv_future.result()

        if progress_bar:
            progress_bar.progress(0.4, text="検索結果を処理中...")

        # 取得した文献を統合
        all_docs = pubmed_docs + arxiv_docs

        if not all_docs:
            # エラー用のダミードキュメント
            all_docs = [Document(
                page_content="症状に対する医学文献が見つかりませんでした。より一般的な症状の説明を試してください。",
                metadata={"source": "error_message"}
            )]

        # 文献の情報を返す
        pubmed_info = []
        for i, doc in enumerate(pubmed_docs):
            pubmed_info.append({
                "index": i + 1,
                "title": doc.metadata.get('Title', 'タイトル情報なし'),
                "authors": doc.metadata.get('Authors', '著者情報なし'),
                "pubdate": doc.metadata.get('Publication Date', '出版日情報なし')
            })

        arxiv_info = []
        for i, doc in enumerate(arxiv_docs):
            arxiv_info.append({
                "index": i + 1,
                "title": doc.metadata.get('Title', 'タイトル情報なし'),
                "authors": doc.metadata.get('Authors', '著者情報なし'),
                "published": doc.metadata.get('Published', '出版日情報なし')
            })

        if progress_bar:
            progress_bar.progress(0.6, text="ベクトルデータベースを作成中...")

        # テキストを分割
        splits = self.text_splitter.split_documents(all_docs)

        # ベクトルデータベースの作成
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        if progress_bar:
            progress_bar.progress(0.7, text="診断準備完了...")

        return retriever, english_symptoms, pubmed_info, arxiv_info

    def diagnose(self, symptoms, patient_data=None, source_language="日本語", target_language="日本語", progress_bar=None):
        """
        症状に基づいて病名を特定し、指定された言語で回答
        """
        # 患者データの処理
        additional_context = ""
        if patient_data:
            patient_info = "\n".join([f"{k}: {v}" for k, v in patient_data.items() if v])
            if patient_info.strip():
                additional_context = f"患者情報:\n{patient_info}"

        # 知識ベースを作成（症状を英語に翻訳）
        retriever, english_symptoms, pubmed_info, arxiv_info = self.create_knowledge_base(
            symptoms, source_language, progress_bar
        )

        if progress_bar:
            progress_bar.progress(0.8, text="診断分析中...")

        # プロンプトの作成（言語に応じて）
        if target_language.lower() == "日本語":
            prompt = ChatPromptTemplate.from_template("""
            あなたは難病診断の専門家です。以下の症状に基づいて、考えうる病名を全て挙げ、その理由を日本語で簡潔に説明してください。

            患者の症状: {input}
            英語に翻訳された症状: {english_symptoms}

            追加情報: {additional_context}

            以下は参考となる医学論文の情報です:
            {context}

            回答形式:
            考えられる病名：[病名のリスト]
            説明：[簡潔な説明と理由]
            注意点：[診断における注意点や追加で必要な検査など]

            可能な限り多くの病名を、鑑別診断の観点から幅広く列挙してください。
            頻度の高いものから低いものまで、可能性のある病名を網羅的に提示してください。
            考えられる病名は、10つ以上挙げ、それぞれについて根拠を簡潔に説明してください。

            注意：これは医学的アドバイスではなく、実際の診断には医師の診察が必要です。
            """)
        else:
            prompt = ChatPromptTemplate.from_template("""
            You are an expert in diagnosing intractable diseases. Based on the symptoms below, please list all possible names of the disease and briefly explain why in English.

            Patient's symptoms: {input}
            Symptoms translated to English: {english_symptoms}

            Additional information: {additional_context}

            Below is information from relevant medical literature:
            {context}

            Response format:
            Possible diagnoses: [list of conditions]
            Explanation: [concise explanation and reasoning]
            Important notes: [diagnostic considerations or additional tests needed]

            List as many disease names as possible, broadly in terms of differential diagnosis.
            Please provide a comprehensive list of possible disease names, from most frequent to least frequent.
            List at least ten (10) possible disease names and briefly explain the rationale for each.

            Note: This is not medical advice and an actual diagnosis requires consultation with a physician.
            """)

        # 文書結合チェーンの作成
        document_chain = create_stuff_documents_chain(self.llm, prompt)

        # 検索チェーンの作成
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # 診断の実行
        result = retrieval_chain.invoke({
            "input": symptoms,
            "english_symptoms": english_symptoms,
            "additional_context": additional_context
        })

        if progress_bar:
            progress_bar.progress(1.0, text="診断完了！")

        return result["answer"], english_symptoms, pubmed_info, arxiv_info


# メイン部分
def main():
    # タブの作成
    tab1, tab2 = st.tabs(["診断", "使い方"])
    
    with tab1:
        # フォームの作成
        with st.form("diagnosis_form"):
            # 入力言語と出力言語の選択
            col1, col2 = st.columns(2)
            with col1:
                source_language = st.selectbox(
                    "入力する症状の言語",
                    ["日本語", "英語", "中国語", "韓国語", "スペイン語", "フランス語", "ドイツ語"], 
                    index=0
                )
            
            with col2:
                target_language = st.selectbox(
                    "診断結果の言語",
                    ["日本語", "英語"],
                    index=0
                )
                
            # 症状の入力
            symptoms = st.text_area("症状を詳しく入力してください", height=150)
            
            # 患者情報の入力（折りたたみ可能なセクション）
            with st.expander("患者情報（任意）"):
                col1, col2 = st.columns(2)
                with col1:
                    age = st.text_input("年齢")
                    gender = st.selectbox("性別", ["", "男性", "女性", "その他"])
                    allergies = st.text_input("アレルギー")
                
                with col2:
                    medical_history = st.text_area("既往歴", height=80)
                    current_medication = st.text_area("現在服用中の薬", height=80)
            
            # 診断ボタン
            submitted = st.form_submit_button("診断開始")
    
        # 診断が開始された場合
        if submitted:
            if not api_key:
                st.error("OpenAI APIキーを入力してください。")
            elif not symptoms:
                st.error("症状を入力してください。")
            else:
                # 患者データの処理
                patient_data = {
                    "年齢": age,
                    "性別": gender,
                    "既往歴": medical_history,
                    "アレルギー": allergies,
                    "現在の薬": current_medication
                }
                
                # プログレスバーの表示
                progress_bar = st.progress(0, text="準備中...")
                
                # 診断ボットの初期化
                try:
                    bot = MultilingualMedicalBot(api_key, model_name)
                    
                    # 診断の実行
                    with st.spinner("診断中..."):
                        diagnosis, english_symptoms, pubmed_info, arxiv_info = bot.diagnose(
                            symptoms,
                            patient_data,
                            source_language=source_language,
                            target_language=target_language,
                            progress_bar=progress_bar
                        )
                    
                    # 翻訳された症状の表示
                    st.subheader("症状（英語翻訳）")
                    st.info(english_symptoms)
                    
                    # 診断結果の表示
                    st.subheader("診断結果")
                    st.markdown(diagnosis)
                    
                    # 参照文献の表示
                    with st.expander("参照文献"):
                        st.subheader("PubMedの文献")
                        if pubmed_info:
                            for doc in pubmed_info:
                                st.markdown(f"**{doc['index']}. {doc['title']}**")
                                st.markdown(f"著者: {doc['authors']}")
                                st.markdown(f"出版日: {doc['pubdate']}")
                                st.markdown("---")
                        else:
                            st.write("PubMedからの文献はありません。")
                            
                        st.subheader("ArXivの文献")
                        if arxiv_info:
                            for doc in arxiv_info:
                                st.markdown(f"**{doc['index']}. {doc['title']}**")
                                st.markdown(f"著者: {doc['authors']}")
                                st.markdown(f"出版日: {doc['published']}")
                                st.markdown("---")
                        else:
                            st.write("ArXivからの文献はありません。")
                    
                    # 注意事項の表示
                    st.warning("注意: これは医学的アドバイスではありません。適切な医療専門家に相談してください。")
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
    
    with tab2:
        st.subheader("使い方")
        st.markdown("""
        このアプリケーションは、入力された症状から考えられる病名を提案するツールです。以下の手順で使用できます：
        
        1. サイドバーで自分のOpenAI APIキーを入力します。
        2. 使用するモデルを選択します（GPT-3.5またはGPT-4）。
        3. 症状を入力する言語と診断結果の言語を選択します。
        4. 症状をテキストエリアに詳しく入力します。
        5. 必要に応じて患者情報（年齢、性別、既往歴など）を入力します。
        6. 「診断開始」ボタンをクリックします。
        
        システムは入力された症状を英語に翻訳し、PubMedとArXivから関連する医学論文を検索します。
        AIが文献を分析し、考えられる病名とその説明を提供します。
        
        **重要な注意点**：
        - このツールは医学的アドバイスの代わりになるものではありません。
        - 実際の診断には必ず医療専門家に相談してください。
        - APIキーは安全に保管され、サーバーに保存されることはありません。
        """)
        
        st.subheader("技術的な詳細")
        st.markdown("""
        このアプリケーションは以下の技術を使用しています：
        
        - **Streamlit**: ウェブインターフェース
        - **LangChain**: AIモデルとのインテグレーション
        - **OpenAI API**: GPT-3.5/GPT-4による症状の翻訳と診断
        - **PubMed & ArXiv API**: 医学論文の検索
        - **FAISS**: ベクトル検索による関連性の高い情報の取得
        
        症状は最初に英語に翻訳され、その後PubMedとArXivから関連する医学論文が検索されます。
        これらの文献はAIモデルに提供され、可能性のある診断を導き出すために使用されます。
        """)

if __name__ == "__main__":
    main()