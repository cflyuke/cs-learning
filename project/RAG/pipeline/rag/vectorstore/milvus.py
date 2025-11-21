from typing import Dict, Any

from pymilvus import (
    MilvusClient,
    AnnSearchRequest, 
    RRFRanker
)

from rag.utils import run_in_executor

class Milvus:
    def __init__(self, uri: str, collection_name: str):
        self.uri = uri
        self.collection_name = collection_name
        self.client = MilvusClient(uri=uri)


    def search(self, vector_dict: Dict[str, Any], limit: int = 8):
        """
        Perform a hybrid search in Milvus using the provided vector dictionary.
        """
        reqs = []
        for name, embeddings in vector_dict.items():
            dense_req = AnnSearchRequest(
                data=embeddings,
                anns_field=name,
                param={"metric_type": "IP", "params": {}},
                limit=limit
            )
            reqs.append(dense_req)

        ranker = RRFRanker()
        res = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=limit,
            output_fields=["unique_id"]
        )
        return res[0]

    async def asearch(self, vector_dict: Dict[str, Any], limit: int = 8):
        """
        Perform an asynchronous hybrid search in Milvus using the provided vector dictionary.
        """
        return await run_in_executor(None, self.search, vector_dict, limit)
