from colorama import Fore, Style

from ollama_chat.core import on_print

class FullDocumentStore:
    """
    SQLite-based storage for full document content.
    This enables retrieval of complete original documents when chunks are found via semantic search.
    """
    def __init__(self, db_path='full_documents.db', verbose=False):
        """
        Initialize the full document store.
        
        :param db_path: Path to the SQLite database file
        :param verbose: Enable verbose logging
        """
        self.db_path = db_path
        self.verbose = verbose
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Create the database and table if they don't exist."""
        import sqlite3
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create table with document_id as primary key and full_content as text
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS full_documents (
                document_id TEXT PRIMARY KEY,
                full_content TEXT NOT NULL,
                file_path TEXT,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index on file_path for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_file_path ON full_documents(file_path)
        ''')
        
        self.conn.commit()
        
        if self.verbose:
            on_print(f"FullDocumentStore initialized at: {self.db_path}", Fore.WHITE + Style.DIM)
    
    def store_document(self, document_id, full_content, file_path=None):
        """
        Store a full document in the database.
        
        :param document_id: Unique identifier for the document
        :param full_content: Full text content of the document
        :param file_path: Optional file path for reference
        """
        import sqlite3
        cursor = self.conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO full_documents (document_id, full_content, file_path)
                VALUES (?, ?, ?)
            ''', (document_id, full_content, file_path))
            
            self.conn.commit()
            
            if self.verbose:
                on_print(f"Stored full document: {document_id}", Fore.WHITE + Style.DIM)
            
            return True
        except sqlite3.Error as e:
            on_print(f"Error storing document {document_id}: {e}", Fore.RED)
            return False
    
    def get_document(self, document_id):
        """
        Retrieve a full document by its ID.
        
        :param document_id: Document identifier
        :return: Full document content or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT full_content FROM full_documents WHERE document_id = ?', (document_id,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        return None
    
    def document_exists(self, document_id):
        """
        Check if a document exists in the store.
        
        :param document_id: Document identifier
        :return: True if exists, False otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT 1 FROM full_documents WHERE document_id = ? LIMIT 1', (document_id,))
        return cursor.fetchone() is not None
    
    def get_all_document_ids(self):
        """
        Get all document IDs in the store.
        
        :return: List of document IDs
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT document_id FROM full_documents')
        return [row[0] for row in cursor.fetchall()]
    
    def count_documents(self):
        """
        Get the total number of documents in the store.
        
        :return: Count of documents
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM full_documents')
        result = cursor.fetchone()
        return result[0] if result else 0
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            if self.verbose:
                on_print("FullDocumentStore connection closed", Fore.WHITE + Style.DIM)


