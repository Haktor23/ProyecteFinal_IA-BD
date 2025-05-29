# elasticsearch_api_client.py
import requests
import json
import logging
import urllib3
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ElasticsearchApiClient:
    def __init__(self, host: str, username: str = "", password: str = ""):
        self.host = host.rstrip('/')
        self.session = requests.Session()
        if username and password:
            self.session.auth = (username, password)
        self.session.headers.update({'Content-Type': 'application/json', 'Accept': 'application/json'})
        
        # Consider SSL verification in production
        self.session.verify = False 
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def health_check(self) -> bool:
        try:
            response = self.session.get(f"{self.host}/_cluster/health", timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def index_exists(self, index_name: str) -> bool:
        try:
            response = self.session.head(f"{self.host}/{index_name}", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def delete_index(self, index_name: str) -> bool:
        try:
            response = self.session.delete(f"{self.host}/{index_name}", timeout=30)
            return response.status_code in [200, 201, 404] # 404 means it didn't exist, so effectively deleted
        except requests.RequestException as e:
            logger.error(f"Error deleting index {index_name}: {e}")
            return False

    def create_index(self, index_name: str, mapping: Dict[str, Any]) -> bool:
        try:
            response = self.session.put(f"{self.host}/{index_name}", json=mapping, timeout=30)
            response.raise_for_status()
            return response.status_code in [200, 201]
        except requests.RequestException as e:
            logger.error(f"Error creating index {index_name}: {e}. Response: {response.text if 'response' in locals() else 'N/A'}")
            return False

    def bulk_index(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not operations:
            return {'errors': False, 'items': []}
        try:
            bulk_body_parts = []
            for op in operations:
                action = list(op.keys())[0] # "index", "create", "update", "delete"
                meta = op[action]
                bulk_body_parts.append(json.dumps({action: meta}))
                if 'body' in op: # body is not present for delete operations
                    bulk_body_parts.append(json.dumps(op['body']))
            
            bulk_data = '\n'.join(bulk_body_parts) + '\n'
            
            response = self.session.post(
                f"{self.host}/_bulk",
                data=bulk_data.encode('utf-8'), # Ensure UTF-8 encoding
                headers={'Content-Type': 'application/x-ndjson'},
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error during bulk indexing: {e}. Response: {response.text if 'response' in locals() else 'N/A'}")
            return {'errors': True, 'items': [], 'error_message': str(e)}