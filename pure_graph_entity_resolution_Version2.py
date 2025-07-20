import json
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values, Json
import jellyfish
import networkx as nx
import recordlinkage
from recordlinkage.preprocessing import clean
import re
from datetime import datetime

class GraphEntityResolution:
    def __init__(self, db_config):
        """Initialize the entity resolution system with database configuration."""
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        
    def connect_to_db(self):
        """Establish database connection."""
        self.conn = psycopg2.connect(**self.db_config)
        self.cursor = self.conn.cursor()
        
    def close_connection(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def import_source_data(self, system_id, data_list):
        """Import raw data from a source system into nodes."""
        query = """
            INSERT INTO nodes (
                type, source_system_id, external_id, raw_data, 
                created_at, updated_at
            )
            VALUES %s
            ON CONFLICT (source_system_id, external_id) 
            DO UPDATE SET 
                raw_data = EXCLUDED.raw_data, 
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
        """
        template = "(%(type)s, %(system_id)s, %(external_id)s, %(raw_data)s, %(created_at)s, %(updated_at)s)"
        
        now = datetime.utcnow()
        params = [
            {
                'type': 'RECORD',
                'system_id': system_id, 
                'external_id': item.get('id', f'unknown_{i}'), 
                'raw_data': Json(item),
                'created_at': now,
                'updated_at': now
            } 
            for i, item in enumerate(data_list)
        ]
        
        ids = execute_values(self.cursor, query, params, template, fetch=True)
        self.conn.commit()
        return [id[0] for id in ids]
    
    def normalize_node_data(self, node_id, mapping_config):
        """Normalize raw data in a node using mapping configuration."""
        # Get raw data
        self.cursor.execute(
            "SELECT raw_data FROM nodes WHERE id = %s AND type = 'RECORD'", 
            (node_id,)
        )
        result = self.cursor.fetchone()
        if not result:
            print(f"Node {node_id} not found or not a record")
            return None
            
        raw_data = result[0]
        
        # Initialize normalized data structure
        normalized = {
            "identity": {},
            "contact": {},
            "attributes": {},
            "metadata": {
                "normalization_date": datetime.utcnow().isoformat(),
                "normalization_version": "1.0"
            }
        }
        
        # Extract and normalize fields
        for target_path, source_path in mapping_config.items():
            value = self._get_nested_value(raw_data, source_path)
            if value is not None:
                # Determine target location based on path structure
                parts = target_path.split('.')
                if len(parts) == 1:
                    # Top level attribute
                    normalized[parts[0]] = value
                else:
                    # Nested attribute
                    section = normalized
                    for i in range(len(parts) - 1):
                        section = section.setdefault(parts[i], {})
                    section[parts[-1]] = value
        
        # Apply standardization to key fields
        if "identity" in normalized:
            identity = normalized["identity"]
            if "first_name" in identity:
                identity["first_name_normalized"] = self._normalize_name(identity["first_name"])
            if "last_name" in identity:
                identity["last_name_normalized"] = self._normalize_name(identity["last_name"])
        
        if "contact" in normalized:
            contact = normalized["contact"]
            if "email" in contact:
                contact["email_normalized"] = self._normalize_email(contact["email"])
            if "phone" in contact:
                contact["phone_normalized"] = self._normalize_phone(contact["phone"])
        
        # Add phonetic encoding for names to improve matching
        if "identity" in normalized:
            identity = normalized["identity"]
            if "first_name" in identity:
                identity["first_name_metaphone"] = jellyfish.metaphone(str(identity.get("first_name", "")))
            if "last_name" in identity:
                identity["last_name_metaphone"] = jellyfish.metaphone(str(identity.get("last_name", "")))
        
        # Update the node with normalized data
        self.cursor.execute(
            "UPDATE nodes SET normalized_data = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
            (Json(normalized), node_id)
        )
        self.conn.commit()
        return normalized

    def _normalize_name(self, name):
        """Normalize a name: lowercase, remove extra spaces and punctuation."""
        if not name:
            return ""
        # Convert to string in case it's a number or other type
        name = str(name).lower()
        # Remove titles, suffixes, etc.
        name = re.sub(r'\b(mr|mrs|ms|dr|prof|jr|sr|i|ii|iii|iv|v)\.?\b', '', name, flags=re.IGNORECASE)
        # Remove punctuation and standardize spacing
        name = re.sub(r'[^\w\s]', '', name)
        return ' '.join(name.split())
    
    def _normalize_email(self, email):
        """Normalize email address."""
        if not email:
            return ""
        return str(email).lower().strip()
    
    def _normalize_phone(self, phone):
        """Normalize phone number to digits only."""
        if not phone:
            return ""
        return re.sub(r'\D', '', str(phone))
    
    def _get_nested_value(self, data, path):
        """Get a value from a nested dictionary using a path string like 'contact.name.first'."""
        if isinstance(path, str):
            parts = path.split('.')
        else:
            parts = path
            
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
    
    def find_similar_nodes_with_blocking(self, min_similarity=0.7):
        """Find similar nodes using blocking strategies to reduce comparison space."""
        # Get all normalized record nodes
        self.cursor.execute("""
            SELECT id, normalized_data 
            FROM nodes 
            WHERE type = 'RECORD' AND normalized_data IS NOT NULL
        """)
        nodes = self.cursor.fetchall()
        
        print(f"Processing {len(nodes)} nodes using blocking strategy")
        
        # Create blocks based on multiple criteria
        blocks = {}
        
        for node_id, data in nodes:
            # Skip nodes without sufficient data
            if not data or 'identity' not in data:
                continue
                
            identity = data.get('identity', {})
            
            # Create multiple blocking keys
            blocking_keys = []
            
            # Block by first letter of last name (if available)
            if 'last_name' in identity and identity['last_name']:
                last_name_initial = str(identity['last_name'])[0:1].lower()
                blocking_keys.append(f"ln:{last_name_initial}")
            
            # Block by year of birth (if available)
            if 'dob' in identity and identity['dob']:
                try:
                    # Extract year from date string (handling different formats)
                    dob = identity['dob']
                    # Try to extract year from ISO format (YYYY-MM-DD)
                    if isinstance(dob, str) and len(dob) >= 4:
                        year = dob[0:4]
                        if year.isdigit():
                            blocking_keys.append(f"yob:{year}")
                except:
                    pass
            
            # Block by postal code prefix (if available)
            contact = data.get('contact', {})
            address = contact.get('address', {})
            if 'postal_code' in address and address['postal_code']:
                postal_prefix = str(address['postal_code'])[0:3]
                blocking_keys.append(f"zip:{postal_prefix}")
            
            # Block by email domain (if available)
            if 'email' in contact and contact['email']:
                email = contact['email']
                if '@' in email:
                    domain = email.split('@')[1]
                    blocking_keys.append(f"domain:{domain}")
            
            # Ensure every record gets assigned to at least one block
            if not blocking_keys:
                blocking_keys.append("unblocked")
            
            # Add node to each of its blocks
            for key in blocking_keys:
                if key not in blocks:
                    blocks[key] = []
                blocks[key].append((node_id, data))
        
        print(f"Created {len(blocks)} blocks for comparison")
        
        # Process comparisons within each block
        relationships = []
        total_comparisons = 0
        total_potential_comparisons = sum(len(block) * (len(block) - 1) // 2 for block in blocks.values())
        
        print(f"Will perform {total_potential_comparisons} comparisons (vs {len(nodes) * (len(nodes) - 1) // 2} without blocking)")
        
        # Network graph for clustering
        G = nx.Graph()
        processed_pairs = set()  # Track which pairs we've already processed
        
        for block_key, block_nodes in blocks.items():
            # Compare each pair in the block
            for i in range(len(block_nodes)):
                for j in range(i + 1, len(block_nodes)):
                    node1_id, node1_data = block_nodes[i]
                    node2_id, node2_data = block_nodes[j]
                    
                    # Skip if we've already compared these nodes in another block
                    node_pair = tuple(sorted([node1_id, node2_id]))
                    if node_pair in processed_pairs:
                        continue
                    
                    processed_pairs.add(node_pair)
                    
                    # Calculate similarity
                    similarity_score, match_evidence = self._calculate_similarity(node1_data, node2_data)
                    total_comparisons += 1
                    
                    # If similarity above threshold, create a relationship
                    if similarity_score >= min_similarity:
                        # Store scores and evidence in properties
                        properties = {
                            'similarity_score': similarity_score,
                            'match_evidence': match_evidence,
                            'status': 'proposed'
                        }
                        
                        relationships.append({
                            'source': node1_id,
                            'target': node2_id,
                            'type': 'SIMILAR_TO',
                            'properties': properties
                        })
                        
                        # Add edge to graph
                        G.add_edge(node1_id, node2_id, weight=similarity_score)
                    
                    # Batch insert relationships periodically
                    if len(relationships) >= 1000:
                        self._insert_relationships(relationships)
                        relationships = []
        
        # Insert any remaining relationships
        if relationships:
            self._insert_relationships(relationships)
            
        print(f"Completed {total_comparisons} actual comparisons. Found {len(relationships)} potential matches")
        return G
    
    def _insert_relationships(self, relationships):
        """Insert relationships into the database."""
        query = """
            INSERT INTO relationships (
                source, target, type, properties
            )
            VALUES %s
            ON CONFLICT (
                LEAST(source, target), 
                GREATEST(source, target),
                type
            )
            DO UPDATE SET 
                properties = EXCLUDED.properties,
                updated_at = CURRENT_TIMESTAMP
        """
        template = "(%(source)s, %(target)s, %(type)s, %(properties)s)"
        
        params = [{
            'source': rel['source'],
            'target': rel['target'],
            'type': rel['type'],
            'properties': Json(rel.get('properties', {}))
        } for rel in relationships]
        
        execute_values(self.cursor, query, params, template)
        self.conn.commit()
    
    def _calculate_similarity(self, node1_data, node2_data):
        """Calculate similarity between two nodes based on their normalized data."""
        scores = {}
        
        # Compare identity fields
        identity1 = node1_data.get('identity', {})
        identity2 = node2_data.get('identity', {})
        
        # First name comparison (using multiple techniques)
        if 'first_name_normalized' in identity1 and 'first_name_normalized' in identity2:
            fn1 = identity1['first_name_normalized']
            fn2 = identity2['first_name_normalized']
            if fn1 and fn2:
                name_similarity = jellyfish.jaro_winkler_similarity(fn1, fn2)
                scores['first_name'] = name_similarity
                
                # Bonus for metaphone match (same pronunciation)
                if 'first_name_metaphone' in identity1 and 'first_name_metaphone' in identity2:
                    if identity1['first_name_metaphone'] == identity2['first_name_metaphone']:
                        scores['first_name'] = max(scores['first_name'], 0.9)  # High confidence for phonetic match
        
        # Last name comparison
        if 'last_name_normalized' in identity1 and 'last_name_normalized' in identity2:
            ln1 = identity1['last_name_normalized']
            ln2 = identity2['last_name_normalized']
            if ln1 and ln2:
                name_similarity = jellyfish.jaro_winkler_similarity(ln1, ln2)
                scores['last_name'] = name_similarity
                
                # Bonus for metaphone match
                if 'last_name_metaphone' in identity1 and 'last_name_metaphone' in identity2:
                    if identity1['last_name_metaphone'] == identity2['last_name_metaphone']:
                        scores['last_name'] = max(scores['last_name'], 0.9)
        
        # Date of birth (exact match only)
        if 'dob' in identity1 and 'dob' in identity2:
            dob1 = identity1['dob']
            dob2 = identity2['dob']
            if dob1 and dob2 and dob1 == dob2:
                scores['dob'] = 1.0
            else:
                scores['dob'] = 0.0
        
        # Email comparison
        contact1 = node1_data.get('contact', {})
        contact2 = node2_data.get('contact', {})
        
        if 'email_normalized' in contact1 and 'email_normalized' in contact2:
            email1 = contact1['email_normalized']
            email2 = contact2['email_normalized']
            if email1 and email2:
                if email1 == email2:
                    scores['email'] = 1.0  # Exact match
                else:
                    # Compare username part
                    username1 = email1.split('@')[0] if '@' in email1 else email1
                    username2 = email2.split('@')[0] if '@' in email2 else email2
                    scores['email'] = jellyfish.jaro_winkler_similarity(username1, username2)
        
        # Phone comparison
        if 'phone_normalized' in contact1 and 'phone_normalized' in contact2:
            phone1 = contact1['phone_normalized']
            phone2 = contact2['phone_normalized']
            if phone1 and phone2:
                # For phone, consider last N digits
                last_digits1 = phone1[-7:] if len(phone1) >= 7 else phone1
                last_digits2 = phone2[-7:] if len(phone2) >= 7 else phone2
                if last_digits1 == last_digits2:
                    scores['phone'] = 1.0
                else:
                    # Partial match based on digit sequence
                    scores['phone'] = max(0.0, len(set(phone1) & set(phone2)) / max(len(phone1), len(phone2)))
        
        # Address comparison
        address1 = contact1.get('address', {})
        address2 = contact2.get('address', {})
        
        if address1 and address2:
            # Postal code exact match is strong indicator
            if 'postal_code' in address1 and 'postal_code' in address2:
                if address1['postal_code'] == address2['postal_code']:
                    scores['postal_code'] = 1.0
            
            # City comparison
            if 'city' in address1 and 'city' in address2:
                city1 = str(address1['city']).lower()
                city2 = str(address2['city']).lower()
                scores['city'] = jellyfish.jaro_winkler_similarity(city1, city2)
        
        # Calculate weighted average score
        weights = {
            'first_name': 0.2,
            'last_name': 0.25,
            'dob': 0.3,
            'email': 0.15,
            'phone': 0.1,
            'postal_code': 0.05,
            'city': 0.05
        }
        
        total_weight = 0.0
        total_score = 0.0
        
        for field, score in scores.items():
            field_weight = weights.get(field, 0.05)  # Default weight for other fields
            total_score += score * field_weight
            total_weight += field_weight
        
        # Avoid division by zero
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0.0
        
        return final_score, scores
    
    def create_clusters_from_graph(self, min_similarity=0.7):
        """Create clusters by finding connected components in the similarity graph."""
        # Load edges into a networkx graph
        self.cursor.execute("""
            SELECT source, target, properties->>'similarity_score' 
            FROM relationships 
            WHERE type = 'SIMILAR_TO' 
              AND (properties->>'similarity_score')::float >= %s
        """, (min_similarity,))
        
        edges = self.cursor.fetchall()
        
        # Create graph
        G = nx.Graph()
        for source, target, weight in edges:
            G.add_edge(source, target, weight=float(weight))
        
        # Get all record nodes to include singletons
        self.cursor.execute("SELECT id FROM nodes WHERE type = 'RECORD'")
        all_nodes = [row[0] for row in self.cursor.fetchall()]
        for node_id in all_nodes:
            if node_id not in G:
                G.add_node(node_id)
        
        # Find connected components (clusters)
        clusters = list(nx.connected_components(G))
        print(f"Found {len(clusters)} clusters")
        
        # Create cluster nodes and membership relationships
        for i, cluster_nodes in enumerate(clusters):
            # Convert from set to list
            node_list = list(cluster_nodes)
            
            # Skip empty clusters
            if not node_list:
                continue
                
            # Create new cluster node
            self.cursor.execute(
                """
                INSERT INTO nodes 
                (type, created_at, updated_at) 
                VALUES ('CLUSTER', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                RETURNING id
                """,
            )
            cluster_id = self.cursor.fetchone()[0]
            
            # Add nodes to cluster by creating BELONGS_TO relationships
            cluster_relationships = [
                {
                    'source': node_id,
                    'target': cluster_id,
                    'type': 'BELONGS_TO',
                    'properties': {
                        'date_added': datetime.utcnow().isoformat()
                    }
                }
                for node_id in node_list
            ]
            
            self._insert_relationships(cluster_relationships)
            
            # Generate golden record for this cluster
            self._generate_golden_record(cluster_id, node_list)
            
            # Progress reporting for large datasets
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} clusters")
        
        self.conn.commit()
        return clusters
    
    def _generate_golden_record(self, cluster_id, node_ids):
        """Generate golden record for a cluster node."""
        # Get all record data in this cluster
        placeholders = ','.join(['%s'] * len(node_ids))
        self.cursor.execute(
            f"""
            SELECT id, normalized_data 
            FROM nodes 
            WHERE id IN ({placeholders}) AND type = 'RECORD'
            """, 
            node_ids
        )
        
        nodes = self.cursor.fetchall()
        
        # No nodes found
        if not nodes:
            return
        
        # Generate golden record
        golden_record = {
            "identity": {},
            "contact": {},
            "attributes": {},
            "metadata": {
                "cluster_id": cluster_id,
                "node_count": len(nodes),
                "source_node_ids": [node[0] for node in nodes],
                "generation_date": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
        }
        
        # Merge identity fields
        self._merge_section(golden_record, "identity", nodes)
        
        # Merge contact fields
        self._merge_section(golden_record, "contact", nodes)
        
        # Merge attributes
        self._merge_section(golden_record, "attributes", nodes)
        
        # Calculate confidence score (simple average)
        confidence = 0.9 if len(nodes) > 1 else 0.7
        
        # Update cluster with golden record
        self.cursor.execute(
            """
            UPDATE nodes 
            SET golden_data = %s, confidence_score = %s, updated_at = CURRENT_TIMESTAMP
            WHERE id = %s AND type = 'CLUSTER'
            """,
            (Json(golden_record), confidence, cluster_id)
        )
        self.conn.commit()
        return golden_record
    
    def _merge_section(self, golden_record, section, nodes):
        """Merge a section of data from multiple nodes into the golden record."""
        # Count occurrences of each value
        field_values = {}
        
        for _, node_data in nodes:
            if section in node_data:
                section_data = node_data[section]
                for field, value in section_data.items():
                    # Skip normalized and derived fields
                    if field.endswith('_normalized') or field.endswith('_metaphone'):
                        continue
                        
                    if field not in field_values:
                        field_values[field] = {}
                    
                    # Convert value to string for counting
                    str_value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
                    if str_value not in field_values[field]:
                        field_values[field][str_value] = {
                            'value': value,
                            'count': 0
                        }
                    field_values[field][str_value]['count'] += 1
        
        # For each field, choose the most common value
        for field, values in field_values.items():
            # Sort by count (descending)
            sorted_values = sorted(values.values(), key=lambda x: x['count'], reverse=True)
            if sorted_values:
                golden_record[section][field] = sorted_values[0]['value']
                
                # Add evidence if there were multiple values
                if len(sorted_values) > 1:
                    if 'field_evidence' not in golden_record['metadata']:
                        golden_record['metadata']['field_evidence'] = {}
                    
                    golden_record['metadata']['field_evidence'][f"{section}.{field}"] = {
                        'values': [{'value': v['value'], 'count': v['count']} for v in sorted_values],
                        'total': sum(v['count'] for v in sorted_values)
                    }
                    
    def get_cluster_for_record(self, record_id):
        """Get the cluster information for a specific record."""
        self.cursor.execute("""
            SELECT c.id, c.golden_data, c.confidence_score
            FROM nodes r
            JOIN relationships rel ON r.id = rel.source
            JOIN nodes c ON rel.target = c.id
            WHERE r.id = %s
              AND r.type = 'RECORD'
              AND c.type = 'CLUSTER'
              AND rel.type = 'BELONGS_TO'
        """, (record_id,))
        
        result = self.cursor.fetchone()
        if result:
            return {
                'cluster_id': result[0],
                'golden_data': result[1],
                'confidence_score': result[2]
            }
        else:
            return None
            
    def get_records_in_cluster(self, cluster_id):
        """Get all records in a specific cluster."""
        self.cursor.execute("""
            SELECT r.id, r.source_system_id, r.external_id, r.raw_data, r.normalized_data
            FROM nodes c
            JOIN relationships rel ON c.id = rel.target
            JOIN nodes r ON rel.source = r.id
            WHERE c.id = %s
              AND c.type = 'CLUSTER'
              AND r.type = 'RECORD'
              AND rel.type = 'BELONGS_TO'
        """, (cluster_id,))
        
        columns = [desc[0] for desc in self.cursor.description]
        records = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        return records

# Example usage
if __name__ == "__main__":
    db_config = {
        'dbname': 'entity_resolution',
        'user': 'postgres',
        'password': 'password',
        'host': 'localhost'
    }
    
    # Initialize the entity resolution system
    resolver = GraphEntityResolution(db_config)
    resolver.connect_to_db()
    
    try:
        # Example field mappings for different systems
        system1_mapping = {
            'identity.first_name': 'name.first',
            'identity.last_name': 'name.last',
            'identity.dob': 'birthdate',
            'contact.email': 'contact.email',
            'contact.phone': 'contact.phone',
            'contact.address.street': 'address.line1',
            'contact.address.city': 'address.city',
            'contact.address.state': 'address.state',
            'contact.address.postal_code': 'address.zip'
        }
        
        system2_mapping = {
            'identity.first_name': 'firstName',
            'identity.last_name': 'lastName',
            'identity.dob': 'dateOfBirth',
            'contact.email': 'emailAddress',
            'contact.phone': 'phoneNumber',
            'contact.address.street': 'address.street',
            'contact.address.city': 'address.city',
            'contact.address.state': 'address.state',
            'contact.address.postal_code': 'address.postalCode'
        }
        
        system3_mapping = {
            'identity.first_name': 'personal.firstName',
            'identity.last_name': 'personal.surname',
            'identity.dob': 'personal.birthDate',
            'contact.email': 'contactDetails.email',
            'contact.phone': 'contactDetails.mobile',
            'contact.address.street': 'contactDetails.address.streetAddress',
            'contact.address.city': 'contactDetails.address.city',
            'contact.address.state': 'contactDetails.address.stateProvince',
            'contact.address.postal_code': 'contactDetails.address.postalCode'
        }
        
        # Full pipeline execution
        print("Starting entity resolution pipeline")
        
        # 1. Import data from systems (this would be replaced with your actual data)
        # sample_data1 = [...]
        # sample_data2 = [...]
        # sample_data3 = [...]
        # system1_ids = resolver.import_source_data(1, sample_data1)
        # system2_ids = resolver.import_source_data(2, sample_data2)
        # system3_ids = resolver.import_source_data(3, sample_data3)
        
        # 2. Normalize data for each system using appropriate mapping
        print("Normalizing record data")
        # Process system 1 records
        resolver.cursor.execute("SELECT id FROM nodes WHERE type = 'RECORD' AND source_system_id = 1")
        for (node_id,) in resolver.cursor.fetchall():
            resolver.normalize_node_data(node_id, system1_mapping)
            
        # Process system 2 records
        resolver.cursor.execute("SELECT id FROM nodes WHERE type = 'RECORD' AND source_system_id = 2")
        for (node_id,) in resolver.cursor.fetchall():
            resolver.normalize_node_data(node_id, system2_mapping)
            
        # Process system 3 records
        resolver.cursor.execute("SELECT id FROM nodes WHERE type = 'RECORD' AND source_system_id = 3")
        for (node_id,) in resolver.cursor.fetchall():
            resolver.normalize_node_data(node_id, system3_mapping)
        
        # 3. Find similar records using blocking
        print("Finding similar records")
        resolver.find_similar_nodes_with_blocking(min_similarity=0.7)
        
        # 4. Create clusters and generate golden records
        print("Creating clusters and golden records")
        resolver.create_clusters_from_graph(min_similarity=0.7)
        
        print("Entity resolution completed successfully!")
        
    finally:
        resolver.close_connection()