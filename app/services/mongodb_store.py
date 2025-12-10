"""
File: app/services/mongodb_store.py
MongoDB Service Layer with Task 6: Metadata Query Operations
"""

import logging
import time
from typing import Optional, List, Dict, Any
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, BulkWriteError
from pymongo.results import UpdateResult, InsertManyResult
from pymongo import ReplaceOne

from app.core.models import NewsArticle

logger = logging.getLogger(__name__)

class MongoDBStore:
    """
    MongoDB Service Layer for article storage with connection management.
    Provides connection pooling, health checks, retry logic, and CRUD operations.
    """
    
    def __init__(
        self,
        connection_string: str,
        database_name: str,
        collection_name: str,
        max_pool_size: int = 100,
        timeout_ms: int = 5000,
        max_retries: int = 3
    ):
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.max_pool_size = max_pool_size
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self.collection: Optional[Collection] = None
        self.is_connected: bool = False
        
        self.connect()
    
    def connect(self) -> bool:
        """Establish MongoDB connection with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                self.client = MongoClient(
                    self.connection_string,
                    maxPoolSize=self.max_pool_size,
                    serverSelectionTimeoutMS=self.timeout_ms,
                    connectTimeoutMS=self.timeout_ms,
                    socketTimeoutMS=self.timeout_ms * 2
                )
                
                self.client.admin.command('ping')
                
                self.db = self.client[self.database_name]
                self.collection = self.db[self.collection_name]
                self.is_connected = True
                
                logger.info(f"✓ MongoDB connected: {self.database_name}.{self.collection_name}")
                return True
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(f"MongoDB connection attempt {attempt}/{self.max_retries} failed: {str(e)}")
                self.is_connected = False
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Unexpected error during MongoDB connection: {str(e)}")
                self.is_connected = False
                return False
        
        logger.error("Could not establish connection to MongoDB after maximum retries.")
        return False
    
    def health_check(self) -> bool:
        """Validate MongoDB connection health."""
        if not self.is_connected or not self.client:
            return False
        
        try:
            self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB health check failed: {str(e)}")
            self.is_connected = False
            return False
    
    def close(self) -> None:
        """Close MongoDB connection and cleanup resources."""
        if self.client:
            try:
                self.client.close()
                logger.info("MongoDB connection closed")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {str(e)}")
            finally:
                self.client = None
                self.db = None
                self.collection = None
                self.is_connected = False

    # ========================================================================
    # TASK 4: MONGODB INSERT OPERATIONS
    # ========================================================================
    
    def insert_article(self, article: NewsArticle) -> str:
        """Insert or update article in MongoDB using upsert."""
        if not self.is_connected or not self.collection:
            raise ConnectionError("Not connected to MongoDB")
        
        mongo_doc = article.to_mongo_document()
        
        result: UpdateResult = self.collection.replace_one(
            {"id": article.id},
            mongo_doc,
            upsert=True
        )
        
        return article.id
    
    def bulk_insert_articles(self, articles: List[NewsArticle]) -> List[str]:
        """Batch insert or update multiple articles using bulk write operations."""
        if not self.is_connected or not self.collection:
            raise ConnectionError("Not connected to MongoDB")
        
        if not articles:
            return []
        
        operations = [
            ReplaceOne(
                filter={"id": article.id},
                replacement=article.to_mongo_document(),
                upsert=True
            )
            for article in articles
        ]
        
        try:
            result = self.collection.bulk_write(operations, ordered=False)
            
            logger.info(
                f"✓ Bulk operation complete: {result.upserted_count} inserted, "
                f"{result.modified_count} updated, "
                f"{result.matched_count} matched."
            )
            
            return [article.id for article in articles]
            
        except BulkWriteError as e:
            logger.error(f"Bulk write error details: {e.details}")
            
            write_errors = e.details.get('writeErrors', [])
            failed_indices = {error['index'] for error in write_errors}
            
            successful_ids = [
                articles[i].id 
                for i in range(len(articles)) 
                if i not in failed_indices
            ]
            return successful_ids
            
        except Exception as e:
            logger.error(f"Unexpected error during bulk insert: {str(e)}")
            raise

    # ========================================================================
    # TASK 5: MONGODB RETRIEVAL OPERATIONS
    # ========================================================================
    
    def get_article_by_id(self, article_id: str) -> Optional[NewsArticle]:
        """Retrieve a single article by its business ID."""
        if not self.is_connected or not self.collection:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            doc = self.collection.find_one({"id": article_id})
            
            if doc is None:
                return None
            
            return NewsArticle.from_mongo_document(doc)
            
        except Exception as e:
            logger.error(f"Error retrieving article {article_id}: {str(e)}")
            raise
    
    def get_articles_by_ids(self, article_ids: List[str]) -> List[NewsArticle]:
        """Retrieve multiple articles by IDs, preserving order."""
        if not self.is_connected or not self.collection:
            raise ConnectionError("Not connected to MongoDB")
        
        if not article_ids:
            return []
        
        try:
            cursor = self.collection.find({"id": {"$in": article_ids}})
            
            article_dict = {
                doc["id"]: NewsArticle.from_mongo_document(doc)
                for doc in cursor
            }
            
            articles = [
                article_dict[article_id] 
                for article_id in article_ids 
                if article_id in article_dict
            ]
            
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving articles by IDs: {str(e)}")
            raise
    
    def get_all_articles(self) -> List[NewsArticle]:
        """Retrieve all articles from the collection."""
        if not self.is_connected or not self.collection:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            cursor = self.collection.find({})
            
            articles = [
                NewsArticle.from_mongo_document(doc)
                for doc in cursor
            ]
            
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving all articles: {str(e)}")
            raise
    
    def get_articles_with_sentiment(self) -> List[NewsArticle]:
        """Retrieve all articles that have sentiment analysis data."""
        if not self.is_connected or not self.collection:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            cursor = self.collection.find({"sentiment": {"$ne": None}})
            
            articles = [
                NewsArticle.from_mongo_document(doc)
                for doc in cursor
            ]
            
            return articles
            
        except Exception as e:
            logger.error(f"Error retrieving articles with sentiment: {str(e)}")
            raise

    # ========================================================================
    # TASK 6: MONGODB METADATA QUERY OPERATIONS
    # ========================================================================
    
    def filter_by_metadata(
        self, 
        filters: Dict[str, Any], 
        limit: Optional[int] = None
    ) -> List[str]:
        """Filter articles by metadata and return matching article IDs sorted by recency."""
        if not self.is_connected or not self.collection:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            query = self.collection.find(
                filters,
                {"id": 1, "_id": 0}
            ).sort("timestamp", -1)
            
            if limit is not None:
                query = query.limit(limit)
            
            article_ids = [doc["id"] for doc in query]
            
            return article_ids
            
        except Exception as e:
            logger.error(f"Error filtering by metadata: {str(e)}")
            raise
    
    def filter_by_sectors(
        self, 
        sectors: List[str], 
        limit: Optional[int] = None
    ) -> List[str]:
        """Filter articles by sectors and return matching article IDs."""
        if not sectors:
            return []
            
        mongo_filter = {"entities.Sectors": {"$in": sectors}}
        
        return self.filter_by_metadata(mongo_filter, limit)

    def filter_by_sentiment(
        self, 
        classification: str, 
        limit: Optional[int] = None
    ) -> List[str]:
        """Filter articles by sentiment classification and return matching article IDs."""
        if not classification:
            return []
            
        mongo_filter = {"sentiment.classification": classification}
        
        return self.filter_by_metadata(mongo_filter, limit)
    
    def count_articles(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count articles matching the specified filters."""
        if not self.is_connected or not self.collection:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            count = self.collection.count_documents(filters or {})
            return count
            
        except Exception as e:
            logger.error(f"Error counting articles: {str(e)}")
            raise
    
    def article_count(self) -> int:
        """
        Get total number of articles in the collection.
        Replaces NewsStorage.article_count().
        """
        return self.count_articles()

    # ========================================================================
    # CONTEXT MANAGER SUPPORT
    # ========================================================================

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()