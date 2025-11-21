# export DOUBAO_API_KEY=""
# export DOUBAO_BASE_URL="https://api.zhizengzeng.com/v1"
# export DOUBAO_MODEL_NAME="doubao-seed-1-6-250615"

# export SEMANTIC_CHUNK_URL="http://127.0.0.1:6000/v1/semantic-chunks"
# nohup uvicorn preprocess.pymupdf.split_server:app --port=6000 --host=127.0.0.1 --workers=1 > ./log/uvicorn.log 2>&1 &

# export PYTHONPATH=$PYTHONPATH:$PWD

# mongodb-linux-x86_64-ubuntu2204-8.0.15/bin/mongod --port=27017 --dbpath=./data/mongodb --logpath=./log/mongodb.log --bind_ip=127.0.0.1 --fork

# nohup vllm serve models/BAAI/bge-m3 \
#     --host 127.0.0.1 \
#     --port 8080 \
#     --task embed \
#     --dtype float16 \
#     --max-model-len 8192 \
#     --max-num-seqs 16 \
#     --gpu-memory-utilization 0.2 \
#     > ./log/vllm_bge_m3_serve.log 2>&1 &

# nohup vllm serve models/Qwen/Qwen3-Embedding-0.6B \
#     --host 127.0.0.1 \
#     --port 8081 \
#     --task embed \
#     --dtype float16 \
#     --max-model-len 8192 \
#     --max-num-seqs 16 \
#     --gpu-memory-utilization 0.3 \
#     > ./log/vllm_qwen3_embed_serve.log 2>&1 &

# nohup vllm serve models/Qwen/Qwen3-Reranker-4B \
#     --hf_overrides '{"architectures": ["Qwen3ForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
#     --host 127.0.0.1 \
#     --port 8082 \
#     --task score \
#     --dtype float16 \
#     --max-model-len 8192 \
#     --max-num-seqs 16 \
#     --gpu-memory-utilization 0.5 \
#     > ./log/vllm_qwen3_reranker_serve.log 2>&1 &

# nohup vllm serve models/Qwen/Qwen3-4B \
#     --host 127.0.0.1 \
#     --port 8083 \
#     --reasoning-parser deepseek_r1 \
#     --dtype float16 \
#     --max-model-len 8192 \
#     --max-num-seqs 16 \
#     --gpu-memory-utilization 0.4 \
#     > ./log/vllm_qwen3_chat_serve.log 2>&1 &


## 测试命令
# curl -sS -X POST 'http://127.0.0.1:8000/v1/rag' -H 'Content-Type: application/json' -d '{"query":"怎么打开车窗","streaming":true}'