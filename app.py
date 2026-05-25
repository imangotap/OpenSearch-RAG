from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List,Optional
import io

from opensearchpy import OpenSearch,RequestsHttpConnection
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

#import cohere                          # サーバーGPUが搭載してない場合使用する。Reranker
from FlagEmbedding import FlagReranker  # GPU搭載の場合使用

from openai import OpenAI
import os

# OpenSearch
os_client = OpenSearch(
    hosts = [{"host":"localhost","port":9200}],
    http_auth = ("admin",""),# <<<<<<<<<<< Password
    use_ssl = True,
    verify_certs = False,
    ssl_show_warn = False,
    connection_class = RequestsHttpConnection,
)
#　埋め込みモデル
embedding_model = SentenceTransformer("BAAI/BGE-M3")

# cohere  　GPUなし場合、cohere使用、サーバーにGPU搭載している場合は　手書きFlagreranker
#cohere_client = cohere.Client(os.getenv("COHERE_API_KEY",""))
#OpenAI　　　
openai_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY","sk-###"),#<<< key
                       base_url="https://openrouter.ai/api/v1",)

# ================= INDEX Setting
INDEX_NAME = "knowledge"
VECTOR_DIM =  1024

INDEX_MAPPING = {
    "settings":{ "index":{"knn":True,"knn.algo_param.ef_search":512,},
                 "analysis":{"analyzer":{"japanese_analyzer":{"type":"custom",
                                                              "tokenizer":"kuromoji_tokenizer", #"standard"
                                                              "filter":["lowercase"],
                                                              }
                                         }
                            }},
    "mappings":{"properties":{"embedding":{"type": "knn_vector",
                                           "dimension": VECTOR_DIM,
                                           "method":{
                                               "name": "hnsw",
                                               "space_type": "cosinesimil",
                                               "engine": "faiss",
                                           },
                                           },
                              "text":{"type":"text","analyzer":"japanese_analyzer"},
                              "filename":{"type":"keyword"},
                              "chunk_index":{"type":"integer"},
                              }
                }
                 }


def create_index_if_not_exist():
    if not os_client.indices.exists(index=INDEX_NAME):
        os_client.indices.create(index=INDEX_NAME,body=INDEX_MAPPING)
        print(f"index{INDEX_NAME} created")

# =============  Fast API
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/upload")
async def upload(file:UploadFile = File(...)):
    content = await file.read()
    full_text = extract_text(file.filename,content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        separators = ["\n\n","。\n","。","\n","　",""],
    )
    chunks = splitter.split_text(full_text)
    embeddings = embedding_model.encode(chunks).tolist()

    bulk_body = []
    for i,(chunk,emb) in enumerate (zip(chunks,embeddings)):
        doc_id = f"{file.filename}_chunk_{i}"
        bulk_body.append({"index": {"_index":INDEX_NAME, "_id": doc_id,}})
        bulk_body.append({"text":chunk,
                          "embedding":emb,
                          "filename":file.filename,
                          "chunk_index":i,
                          })
    resp = os_client.bulk(body=bulk_body)
    #resp = os_client.bulk(bulk_body)

    return {
        "filename":file.filename,
        "text_length":len(full_text),
        "chunk_count":len(chunks),
        "chunks_preview":chunks[:3],
        "status":"Save to OpenSearch",
        "errors":resp.get("errors",False),
        }

def extract_text(filename,content):
    if filename.endswith(".txt"):
        return content.decode("utf-8",errors="ignore")
    if filename.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
            if (page.extract_text() or "").strip()
        )
    else:
        raise ValueError(f"対象外タイプです{filename}")

@app.get("/hybrid")
def hybrid_search(query:str,top_k:int=10):

    query_embeddings = embedding_model.encode(query).tolist()

    search_body = {
        "size": top_k,
        "query": {
            "hybrid":{"queries":[
                {"knn":{"embedding":{"vector":query_embeddings,"k":top_k * 2}}},
                {"match":{"text":{"query":query,"analyzer":"japanese_analyzer"}}},
            ] }
        },
        "_source":["text","filename","chunk_index"],
    }
    response = os_client.search(
        index = INDEX_NAME,
        body  = search_body,
        params = {"search_pipeline":"hybrid-search-pipeline"},
    )

    hits = response["hits"]["hits"]

    final_result = [{
        "doc_id": hit["_id"],
        "text"  : hit["_source"]["text"],
        "metadata": {
            "filename": hit["_source"]["filename"],
            "index": hit["_source"]["chunk_index"],
        },
        "hybrid_score":round(hit["_score"],3),
    }   for hit in hits
    ]
    return {"final_result":final_result}

def expand_context(doc_id: str, window: int = 1) ->str:
    parts = doc_id.rsplit("_chunk_",1)
    filename = parts[0]
    index = int(parts[1])

    texts = []
    #               5-1, 5+1   ->  context [4-6]
    for i in range(index - window , index + window + 1):
        neighbor_id = f"{filename}_chunk_{i}"
        try:
            result = os_client.get(index=INDEX_NAME,id=neighbor_id)
            texts.append(result["_source"]["text"])
        except Exception as e:
            continue
    return "\n".join(texts)

reranker = FlagReranker("BAAI/bge-reranker-v2-m3",use_fp16=True)
def rerank(query,chunks,top_n:int=3):

    pairs = [[query,chunk["text"]] for chunk in chunks]
    scores = reranker.compute_score(pairs)
    for i,chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])

    return sorted(chunks,key=lambda x:x["rerank_score"],reverse=True)[:top_n]

class Message(BaseModel):
    role:str
    content:str
class ChatRequest(BaseModel):
    query:str
    history:List[Message] = []
    session_id:Optional[str]=None
@app.post("/chat")
def chat(request:ChatRequest):

    result = hybrid_search(query=request.query,top_k=10)
    chunks = result["final_result"]

    reranked = rerank(request.query,chunks,top_n=3)

    context = "\n\n".join([
        expand_context(chunk["metadata"]["filename"] + "_chunk_" + str(chunk["metadata"]["index"]))
        for chunk in reranked
    ])

    system_prompt=(
        "あなたは知識庫アシスタントです。以下の【知識庫】に基づいて回答してください。"
        "知識庫にない内容は「資料にないため回答できません」と述べてください。"
        "勝手に作り話は絶対にしないでください。"
    )

    user_prompt = f"""
    【知識庫】
    {context}
    【質問】
    {request.query}
"""
    messages = [{"role":"system","content":system_prompt}]
    messages.extend([{"role":msg.role,"content":msg.content} for msg in request.history])
    messages.append({"role":"user","content":user_prompt})

    response = openai_client.chat.completions.create(
        model="deepseek/deepseek-chat",
        #model="gpt-4o",
        messages=messages,
        temperature=0.0,
    )

    answer_text = response.choices[0].message.content

    updated_history = [
        *[{"role":msg.role,"content":msg.content} for msg in request.history],
        {"role":"user","content":request.query},
        {"role":"assistant","content":answer_text},
    ]
    return {
        "query":request.query,
        "answer":answer_text,
        "source":[chunk["metadata"]for chunk in reranked],
        "history":updated_history,
    }
create_index_if_not_exist()
