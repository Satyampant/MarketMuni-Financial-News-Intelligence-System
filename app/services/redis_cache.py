"""
Redis Cache Service for Entity Extraction
Provides shared, persistent caching across multiple workers.
File: app/services/redis_cache.py
"""

import json
import hashlib
from typing import Optional, Any, Dict
from datetime import timedelta

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠ Warning: redis not installed. Install with: pip install redis")

from app.core.config_loader import get_config


class RedisCacheService:
    """
    Redis-based caching service for entity extraction results.
    Provides shared cache across multiple workers with TTL management.
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        password: str = None,
        ttl_seconds: int = None,
        key_prefix: str = "marketmuni:entities:"
    ):
        """
        Initialize Redis cache service.
        
        Args:
            host: Redis host (default: localhost)
            port: Redis port (default: 6379)
            db: Redis database number (default: 0)
            password: Redis password (default: None)
            ttl_seconds: Cache TTL in seconds (default: 86400 = 24 hours)
            key_prefix: Prefix for all cache keys
        """
        config = get_config()
        
        # Load from config or use defaults
        self.host = host or getattr(config, 'redis_host', 'localhost')
        self.port = port or getattr(config, 'redis_port', 6379)
        self.db = db or getattr(config, 'redis_db', 0)
        self.password = password or getattr(config, 'redis_password', None)
        self.ttl_seconds = ttl_seconds or getattr(config, 'redis_ttl_seconds', 86400)
        self.key_prefix = key_prefix
        
        self.client: Optional[redis.Redis] = None
        self.is_connected = False
        
        if not REDIS_AVAILABLE:
            print("⚠ Redis not available - caching disabled")
            return
        
        self._connect()
    
    def _connect(self) -> None:
        """Establish Redis connection with retry logic."""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,  # Auto-decode bytes to strings
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            self.client.ping()
            self.is_connected = True
            print(f"✓ Redis connected: {self.host}:{self.port} (db={self.db})")
            
        except (RedisConnectionError, RedisError) as e:
            print(f"⚠ Redis connection failed: {e}")
            print("  Entity extraction will work without caching")
            self.is_connected = False
            self.client = None
    
    def _make_key(self, article_id: str) -> str:
        """
        Generate cache key for article.
        
        Args:
            article_id: Unique article identifier
            
        Returns:
            Prefixed cache key
        """
        # Hash article ID to avoid key length issues
        hashed_id = hashlib.md5(article_id.encode()).hexdigest()
        return f"{self.key_prefix}{hashed_id}"
    
    def get(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached entity extraction result.
        
        Args:
            article_id: Article identifier
            
        Returns:
            Cached result dict or None if not found
        """
        if not self.is_connected or self.client is None:
            return None
        
        try:
            key = self._make_key(article_id)
            cached_data = self.client.get(key)
            
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except (RedisError, json.JSONDecodeError) as e:
            print(f"⚠ Redis GET error for {article_id}: {e}")
            return None
    
    def set(
        self,
        article_id: str,
        entity_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache entity extraction result.
        
        Args:
            article_id: Article identifier
            entity_data: Entity extraction result to cache
            ttl: Optional TTL override (seconds)
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.is_connected or self.client is None:
            return False
        
        try:
            key = self._make_key(article_id)
            cache_ttl = ttl or self.ttl_seconds
            
            # Serialize to JSON
            serialized_data = json.dumps(entity_data)
            
            # Set with TTL
            self.client.setex(
                name=key,
                time=timedelta(seconds=cache_ttl),
                value=serialized_data
            )
            
            return True
            
        except (RedisError, TypeError, ValueError) as e:
            print(f"⚠ Redis SET error for {article_id}: {e}")
            return False
    
    def delete(self, article_id: str) -> bool:
        """
        Delete cached entry.
        
        Args:
            article_id: Article identifier
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.is_connected or self.client is None:
            return False
        
        try:
            key = self._make_key(article_id)
            deleted = self.client.delete(key)
            return deleted > 0
            
        except RedisError as e:
            print(f"⚠ Redis DELETE error for {article_id}: {e}")
            return False
    
    def clear_all(self) -> int:
        """
        Clear all cached entities (use with caution).
        
        Returns:
            Number of keys deleted
        """
        if not self.is_connected or self.client is None:
            return 0
        
        try:
            # Find all keys with prefix
            pattern = f"{self.key_prefix}*"
            keys = list(self.client.scan_iter(match=pattern, count=100))
            
            if not keys:
                return 0
            
            # Delete in pipeline for efficiency
            pipeline = self.client.pipeline()
            for key in keys:
                pipeline.delete(key)
            pipeline.execute()
            
            return len(keys)
            
        except RedisError as e:
            print(f"⚠ Redis CLEAR error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        if not self.is_connected or self.client is None:
            return {
                "connected": False,
                "cached_keys": 0,
                "total_keys": 0,
                "memory_used": "0 MB"
            }
        
        try:
            # Count cached entity keys
            pattern = f"{self.key_prefix}*"
            cached_keys = sum(1 for _ in self.client.scan_iter(match=pattern, count=100))
            
            # Get Redis info
            info = self.client.info('memory')
            memory_used_mb = info.get('used_memory', 0) / (1024 * 1024)
            
            return {
                "connected": True,
                "cached_keys": cached_keys,
                "total_keys": self.client.dbsize(),
                "memory_used": f"{memory_used_mb:.2f} MB",
                "ttl_seconds": self.ttl_seconds,
                "host": self.host,
                "port": self.port,
                "db": self.db
            }
            
        except RedisError as e:
            print(f"⚠ Redis STATS error: {e}")
            return {
                "connected": False,
                "error": str(e)
            }
    
    def health_check(self) -> bool:
        """
        Check if Redis is healthy.
        
        Returns:
            True if Redis is reachable, False otherwise
        """
        if not self.is_connected or self.client is None:
            return False
        
        try:
            self.client.ping()
            return True
        except RedisError:
            self.is_connected = False
            return False
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to Redis.
        
        Returns:
            True if reconnected successfully
        """
        if not REDIS_AVAILABLE:
            return False
        
        self._connect()
        return self.is_connected
    
    def close(self) -> None:
        """Close Redis connection."""
        if self.client is not None:
            try:
                self.client.close()
                print("✓ Redis connection closed")
            except RedisError as e:
                print(f"⚠ Error closing Redis: {e}")
            finally:
                self.is_connected = False
                self.client = None


# Singleton instance for shared access
_redis_cache_instance: Optional[RedisCacheService] = None


def get_redis_cache() -> RedisCacheService:
    """
    Get singleton Redis cache instance.
    
    Returns:
        Shared RedisCacheService instance
    """
    global _redis_cache_instance
    
    if _redis_cache_instance is None:
        _redis_cache_instance = RedisCacheService()
    
    return _redis_cache_instance