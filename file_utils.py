import os
import magic
from pathlib import Path
from typing import Tuple, List, Optional
import logging
from datetime import datetime
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader, 
    UnstructuredPowerPointLoader, UnstructuredExcelLoader,
    UnstructuredFileLoader
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File type configurations
ALLOWED_EXTENSIONS = {
    '.txt', '.md', '.pdf', '.doc', '.docx',
    '.xls', '.xlsx', '.ppt', '.pptx'
}

def validate_file(filepath: str) -> Tuple[bool, str]:
    """Validate file existence and type."""
    if not os.path.exists(filepath):
        return False, "File does not exist."
    
    file_ext = Path(filepath).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type. Supported: {', '.join(ALLOWED_EXTENSIONS)}"
    
    try:
        if magic.Magic(mime=True).from_file(filepath) == 'text/plain':
            return True, ""
    except:
        pass
    
    return True, ""

def load_document(filepath: str) -> Tuple[Optional[List], str]:
    """Load document with appropriate loader based on file type."""
    try:
        file_ext = Path(filepath).suffix.lower()
        
        # Select appropriate loader
        if file_ext == '.pdf':
            loader = PyPDFLoader(filepath)
        elif file_ext in ['.doc', '.docx']:
            loader = UnstructuredWordDocumentLoader(filepath)
        elif file_ext in ['.ppt', '.pptx']:
            loader = UnstructuredPowerPointLoader(filepath)
        elif file_ext in ['.xls', '.xlsx']:
            loader = UnstructuredExcelLoader(filepath)
        else:
            loader = UnstructuredFileLoader(filepath)
        
        # Load document and add metadata
        documents = loader.load()
        file_mod_time = datetime.fromtimestamp(Path(filepath).stat().st_mtime)
        
        for doc in documents:
            doc.metadata['last_modified'] = file_mod_time
            doc.metadata['source'] = str(filepath)
        
        return documents, ""
        
    except Exception as e:
        error_msg = f"Error loading {filepath}: {str(e)}"
        logger.error(error_msg)
        return None, error_msg
