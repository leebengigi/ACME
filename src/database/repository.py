import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import sqlite3
import json
import logging
import threading
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from contextlib import contextmanager

from src.models.data_models import SecurityRequest
from src.models.enums import RequestType, Outcome

logger = logging.getLogger(__name__)

class SecurityRequestRepository:
    """
    Repository class for managing security requests in a SQLite database.
    Handles CRUD operations for security requests and their associated interactions.
    Implements thread-safe operations using a lock mechanism.
    """

    def __init__(self, db_path: str):
        """
        Initialize the repository with a database path.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self.init_database()

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        Ensures proper connection handling and cleanup.
        
        Yields:
            sqlite3.Connection: Database connection object
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            yield conn
        finally:
            conn.close()

    def init_database(self):
        """
        Initialize the database schema.
        Creates necessary tables and indexes for storing requests and interactions.
        """
        with self.get_connection() as conn:
            cur = conn.cursor()

            # Create requests table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    thread_ts TEXT,
                    request_text TEXT NOT NULL,
                    request_type TEXT,
                    risk_score REAL,
                    required_fields TEXT,
                    outcome TEXT,
                    rationale TEXT,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Create interactions table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id INTEGER NOT NULL,
                    interaction_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (request_id) REFERENCES requests (id)
                )
            ''')

            # Create indexes for better performance
            cur.execute('CREATE INDEX IF NOT EXISTS idx_requests_user_channel ON requests(user_id, channel_id)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_requests_outcome ON requests(outcome)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_interactions_request_id ON interactions(request_id)')

            conn.commit()
            logger.info("Database initialized successfully")

    def save_request(self, request: SecurityRequest) -> int:
        """
        Save a new security request to the database.
        
        Args:
            request (SecurityRequest): The security request to save
            
        Returns:
            int: The ID of the newly created request
        """
        with self._lock:
            with self.get_connection() as conn:
                cur = conn.cursor()

                cur.execute('''
                    INSERT INTO requests (
                        user_id, channel_id, thread_ts, request_text, request_type,
                        risk_score, required_fields, outcome, rationale, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    request.user_id,
                    request.channel_id,
                    request.thread_ts,
                    request.request_text,
                    request.request_type.value if request.request_type else None,
                    request.risk_score,
                    json.dumps(request.required_fields),
                    request.outcome.value if request.outcome else None,
                    request.rationale,
                    request.timestamp.isoformat()
                ))

                request_id = cur.lastrowid
                conn.commit()

                logger.debug(f"Saved request {request_id} for user {request.user_id}")
                return request_id

    def get_pending_request(self, user_id: str, channel_id: str) -> Optional[Tuple[int, Dict, str, float]]:
        """
        Retrieve the latest pending request for a user in a specific channel.
        
        Args:
            user_id (str): The ID of the user
            channel_id (str): The ID of the channel
            
        Returns:
            Optional[Tuple[int, Dict, str, float]]: Tuple containing request ID, required fields,
            request type, and risk score, or None if no pending request exists
        """
        with self.get_connection() as conn:
            cur = conn.cursor()

            cur.execute("""
                        SELECT id, required_fields, request_type, risk_score
                        FROM requests
                        WHERE user_id = ?
                          AND channel_id = ?
                          AND outcome IS NULL
                        ORDER BY id DESC LIMIT 1
                        """, (user_id, channel_id))

            row = cur.fetchone()
            if not row:
                return None

            request_id, fields_json, request_type, risk_score = row

            try:
                if fields_json:
                    required_fields = json.loads(fields_json)
                else:
                    required_fields = {}
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Could not parse required_fields JSON for request {request_id}: {e}")
                required_fields = {}

            # Ensure it's a dictionary
            if not isinstance(required_fields, dict):
                logger.warning(f"required_fields was not a dict for request {request_id}, converting to dict")
                required_fields = {}

            return request_id, required_fields, request_type, risk_score

    def update_request_fields(self, request_id: int, required_fields: Dict[str, Any]):
        """
        Update the required fields for a specific request.
        
        Args:
            request_id (int): The ID of the request to update
            required_fields (Dict[str, Any]): The new required fields to set
        """
        with self._lock:
            with self.get_connection() as conn:
                cur = conn.cursor()

                cur.execute(
                    "UPDATE requests SET required_fields=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (json.dumps(required_fields), request_id)
                )

                conn.commit()
                logger.debug(f"Updated fields for request {request_id}")

    def finalize_request(self, request_id: int, outcome: str, rationale: str):
        """
        Mark a request as complete with an outcome and rationale.
        
        Args:
            request_id (int): The ID of the request to finalize
            outcome (str): The outcome of the request
            rationale (str): The rationale for the outcome
        """
        with self._lock:
            with self.get_connection() as conn:
                cur = conn.cursor()

                cur.execute(
                    "UPDATE requests SET outcome=?, rationale=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (outcome, rationale, request_id)
                )

                conn.commit()
                logger.debug(f"Finalized request {request_id} with outcome: {outcome}")

    def log_interaction(self, request_id: int, interaction_type: str, content: str):
        """
        Log an interaction related to a request.
        
        Args:
            request_id (int): The ID of the associated request
            interaction_type (str): The type of interaction
            content (str): The content of the interaction
        """
        with self._lock:
            with self.get_connection() as conn:
                cur = conn.cursor()

                cur.execute('''
                    INSERT INTO interactions (request_id, interaction_type, content, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (
                    request_id,
                    interaction_type,
                    content,
                    datetime.now().isoformat()
                ))

                conn.commit()
                logger.debug(f"Logged interaction for request {request_id}: {interaction_type}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve statistics about the requests in the database.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - total_requests: Total number of requests
                - outcomes: Count of requests by outcome
                - request_types: Count of requests by type
                - average_risk_score: Average risk score across all requests
        """
        with self.get_connection() as conn:
            cur = conn.cursor()

            # Total requests
            cur.execute("SELECT COUNT(*) FROM requests")
            total_requests = cur.fetchone()[0]

            # Requests by outcome
            cur.execute("SELECT outcome, COUNT(*) FROM requests WHERE outcome IS NOT NULL GROUP BY outcome")
            outcomes = dict(cur.fetchall())

            # Requests by type
            cur.execute("SELECT request_type, COUNT(*) FROM requests WHERE request_type IS NOT NULL GROUP BY request_type")
            types = dict(cur.fetchall())

            # Average risk score
            cur.execute("SELECT AVG(risk_score) FROM requests WHERE risk_score IS NOT NULL")
            avg_risk = cur.fetchone()[0] or 0

            return {
                'total_requests': total_requests,
                'outcomes': outcomes,
                'request_types': types,
                'average_risk_score': round(avg_risk, 2)
            }

