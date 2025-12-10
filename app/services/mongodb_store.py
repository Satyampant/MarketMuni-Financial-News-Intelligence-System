"""
File: app/services/mongodb_store.py
MongoDB Service Layer with connection management, retry logic, and CRUD operations.
Implements Task 4: MongoDB Insert Operations
"""

import logging
import time
from typing import Optional, List
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, BulkWriteError
from pymongo.results import UpdateResult, InsertManyResult

from app.core.models import NewsArticle

# Configure logger
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
        
        # Attempt initial connection
        self.connect()
    
    def connect(self) -> bool:
        """
        Establish MongoDB connection with retry logic.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                self.client = MongoClient(
                    self.connection_string,
                    maxPoolSize=self.max_pool_size,
                    serverSelectionTimeoutMS=self.timeout_ms,
                    connectTimeoutMS=self.timeout_ms,
                    socketTimeoutMS=self.timeout_ms * 2
                )
                
                # 'ping' to verify connection is actually alive
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
                    time.sleep(2 ** attempt)  # Exponential backoff
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
        """
        Insert or update article in MongoDB using upsert to handle duplicates.
        
        Args:
            article: NewsArticle object to insert
            
        Returns:
            Article ID (business key) as string
            
        Raises:
            ConnectionError: If not connected to MongoDB
            Exception: For other database errors
        """
        if not self.is_connected or not self.collection:
            raise ConnectionError("Not connected to MongoDB")
        
        # Convert article to MongoDB document format
        mongo_doc = article.to_mongo_document()
        
        # Use upsert with article.id as unique key to handle duplicates
        result: UpdateResult = self.collection.replace_one(
            {"id": article.id},
            mongo_doc,
            upsert=True
        )
        
        # Return the business key (article.id) which is known and avoids extra read
        # MongoDB _id is only for internal use
        return article.id
    
    def bulk_insert_articles(self, articles: List[NewsArticle]) -> List[str]:
        """
        Batch insert or update multiple articles using bulk write operations.
        Uses ReplaceOne with upsert=True to handle duplicates gracefully.
        
        Args:
            articles: List of NewsArticle objects to insert/update
            
        Returns:
            List of article IDs (business keys) for successfully processed articles
            
        Raises:
            ConnectionError: If not connected to MongoDB
        """
        if not self.is_connected or not self.collection:
            raise ConnectionError("Not connected to MongoDB")
        
        if not articles:
            return []
        
        # Prepare bulk operations: Replace document if ID exists, Insert if not.
        operations = [
            ReplaceOne(
                filter={"id": article.id},           # Match by business ID
                replacement=article.to_mongo_document(),
                upsert=True                          # Update if exists, Insert if new
            )
            for article in articles
        ]
        
        try:
            # ordered=False allows parallel processing and doesn't stop on a single failure
            result = self.collection.bulk_write(operations, ordered=False)
            
            logger.info(
                f"✓ Bulk operation complete: {result.upserted_count} inserted, "
                f"{result.modified_count} updated, "
                f"{result.matched_count} matched."
            )
            
            # Since we used upsert, we can assume all valid articles in the list 
            # are now present in the database.
            return [article.id for article in articles]
            
        except BulkWriteError as e:
            # Handle partial failures (e.g., validation errors, network glitches)
            logger.error(f"Bulk write error details: {e.details}")
            
            # Extract IDs that failed to write
            write_errors = e.details.get('writeErrors', [])
            failed_indices = {error['index'] for error in write_errors}
            
            # Return only the IDs that didn't fail
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
    # CONTEXT MANAGER SUPPORT
    # ========================================================================

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()