# OpenSearch-RAG
FastAPIとOpenSearchを使ったハイブリッドRAGシステム。密ベクトル検索とBM25検索をRRF融合し、リランク・コンテキスト展開を実装。日本語テキスト処理対応。
# OpenSearch RAG システム

FastAPI と OpenSearch を使ったハイブリッド検索 RAG システム。

## 機能
- ハイブリッド検索（密ベクトル検索 + BM25）
- リランク（BGE-M3 Reranker）
- コンテキスト展開（前後チャンク結合）
- 日本語テキスト処理対応
- PDF / TXT ファイルアップロード対応

## 技術スタック
| 役割 | 技術 |
|------|------|
| API サーバー | FastAPI |
| ベクトル DB | OpenSearch |
| 埋め込みモデル | BAAI/BGE-M3 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | DeepSeek |

## セットアップ

### 1. OpenSearch 起動
```bash
docker run -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e 'OPENSEARCH_INITIAL_ADMIN_PASSWORD=MyPassword123!' \
  opensearchproject/opensearch:latest
```

### 2. 依存関係インストール
```bash
pip install fastapi uvicorn opensearch-py sentence-transformers \
langchain-text-splitters FlagEmbedding openai pypdf cryptography
```

### 3. 環境変数設定
```bash
export OPENAI_API_KEY=your_api_key
export OPENSEARCH_PASSWORD=MyPassword123!
```

### 4. サーバー起動
```bash
uvicorn app:app --reload
```

## API
| エンドポイント | メソッド | 説明 |
|--------------|--------|------|
| /upload | POST | ファイルアップロード |
| /hybrid | GET | ハイブリッド検索 |
| /chat | POST | 会話 |
