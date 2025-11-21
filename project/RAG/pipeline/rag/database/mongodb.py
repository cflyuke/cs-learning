import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError

class MongoDB:
    _host = os.getenv("MONGO_HOST", "localhost")
    _port = os.getenv("MONGO_PORT", 27017)
    _username = os.getenv("MONGO_USERNAME")
    _password = os.getenv("MONGO_PASSWORD")
    _auth_source = os.getenv("MONGO_AUTH_SOURCE", "admin")

    _max_pool_size = 100
    _connect_timeout = 5000
    _socket_timeout = 3000

    _client = None
    

    @classmethod
    def _build_connection_url(cls):
        """构建MongoDB连接url"""
        if cls._username and cls._password:
            return f"mongodb://{cls._username}:{cls._password}@{cls._host}:{cls._port}/~authSource={cls._auth_source}"
        return f"mongodb://{cls._host}:{cls._port}"
    
    @classmethod
    def initialize(cls):
        """构建MongoDB连接"""
        if cls._client is None:
            try:
                cls._client = MongoClient(
                    cls._build_connection_url(),
                    maxPoolSize=cls._max_pool_size,
                    connectTimeoutMS=cls._connect_timeout,
                    socketTimeoutMS = cls._connect_timeout,
                    serverSelectionTimeoutMS=5000
                )
                cls._client.admin.command("ping")
                print("Successfully connected to MongoDB")
            except ConfigurationError as e:
                raise RuntimeError(f"MongoDB cnfiguration error: {str(e)}")
            except ConnectionFailure as e:
                raise RuntimeError(f"MongoDB connection error: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Unexpected MongoDB connection errors: {str(e)}")
            
    @classmethod
    def get_db(cls, db_name):
        """构建MongoDB数据库"""
        if cls._client is None:
            cls.initialize()
        if cls._client is None:
            raise RuntimeError("MongoDB client is not initialized")
        return cls._client[db_name]
    
    @classmethod
    def get_collection(cls, db_name, collection_name):
        return cls.get_db(db_name)[collection_name]
    
    @classmethod
    def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            print("MongoDB connection closed")
    
    @classmethod
    def save(cls, docs, db_name, collection_name):
        collection = cls.get_collection(db_name, collection_name)
        for doc in docs:
            collection.update_one(
                {"unique_id": doc["unique_id"]},
                {"$set": doc},
                upsert=True
            )

MongoDB.initialize()