"""
Neo4j Database Manager for Community Detection project.
"""
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
from neo4j import GraphDatabase
from .config import NEO4J_CONFIG

logger = logging.getLogger(__name__)


class Neo4jManager:
    """Manages Neo4j database connections and operations."""
    
    def __init__(self):
        self.driver = None
        self.connect()
    
    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_CONFIG["uri"],
                auth=(NEO4J_CONFIG["username"], NEO4J_CONFIG["password"])
            )
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self) -> None:
        """Close database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def clear_database(self) -> None:
        """Clear all nodes and relationships from the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")
    
    def load_cora_dataset(self) -> None:
        """Load the Cora citation network dataset into Neo4j."""
        query_create_papers = """
        LOAD CSV WITH HEADERS FROM 'file:///cora/cora.content' AS row FIELDTERMINATOR '\t'
        CREATE (p:Paper {
            id: toInteger(row.paper_id),
            subject: row.subject,
            features: split(row.words, ',')
        })
        RETURN count(p) as papers_created
        """
        
        query_create_citations = """
        LOAD CSV FROM 'file:///cora/cora.cites' AS row FIELDTERMINATOR '\t'
        MATCH (citing:Paper {id: toInteger(row[0])})
        MATCH (cited:Paper {id: toInteger(row[1])})
        CREATE (citing)-[:CITES]->(cited)
        RETURN count(*) as citations_created
        """
        
        with self.driver.session() as session:
            # Create papers
            result1 = session.run(query_create_papers)
            papers_count = result1.single()["papers_created"]
            
            # Create citations
            result2 = session.run(query_create_citations)
            citations_count = result2.single()["citations_created"]
            
            logger.info(f"Loaded {papers_count} papers and {citations_count} citations")
    
    def run_louvain_algorithm(self) -> None:
        """Run Louvain community detection algorithm using GDS."""
        with self.driver.session() as session:
            # Drop existing graph projection if it exists
            try:
                session.run("CALL gds.graph.drop('cora-graph') YIELD graphName")
                logger.info("Dropped existing cora-graph projection")
            except Exception:
                logger.info("No existing cora-graph projection to drop")
            
            # Create graph projection
            session.run("""
                CALL gds.graph.project(
                    'cora-graph',
                    'Paper',
                    {
                        CITES: {
                            orientation: 'UNDIRECTED'
                        }
                    }
                )
            """)
            
            # Run Louvain algorithm
            result = session.run("""
                CALL gds.louvain.write(
                    'cora-graph',
                    {
                        writeProperty: 'louvainCommunityId',
                        includeIntermediateCommunities: true
                    }
                )
                YIELD communityCount, modularities
                RETURN communityCount, modularities
            """)
            
            record = result.single()
            logger.info(f"Louvain algorithm completed: {record}")
    
    def run_label_propagation(self) -> None:
        """Run Label Propagation algorithm using GDS."""
        query = """
        CALL gds.labelPropagation.write(
            'cora-graph',
            {
                writeProperty: 'lpaCommunityId',
                maxIterations: 10
            }
        )
        YIELD communityCount, ranIterations
        RETURN communityCount, ranIterations
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            record = result.single()
            logger.info(f"Label Propagation completed: {record}")
    
    def get_graph_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract graph data for PyTorch processing."""
        # Get nodes with their properties
        nodes_query = """
        MATCH (p:Paper)
        RETURN p.id as id, p.subject as subject, 
               p.louvainCommunityId as louvain_community,
               p.lpaCommunityId as lpa_community
        ORDER BY p.id
        """
        
        # Get edges
        edges_query = """
        MATCH (p1:Paper)-[:CITES]->(p2:Paper)
        RETURN p1.id as source, p2.id as target
        """
        
        with self.driver.session() as session:
            # Get nodes
            nodes_result = session.run(nodes_query)
            nodes_df = pd.DataFrame([dict(record) for record in nodes_result])
            
            # Get edges
            edges_result = session.run(edges_query)
            edges_df = pd.DataFrame([dict(record) for record in edges_result])
            
            logger.info(f"Retrieved {len(nodes_df)} nodes and {len(edges_df)} edges")
            return nodes_df, edges_df
    
    def write_gnn_predictions(self, predictions: Dict[int, int]) -> None:
        """Write GNN community predictions back to Neo4j."""
        query = """
        UNWIND $predictions as pred
        MATCH (p:Paper {id: pred.node_id})
        SET p.gnnCommunityId = pred.community_id
        """
        
        predictions_list = [
            {"node_id": node_id, "community_id": community_id}
            for node_id, community_id in predictions.items()
        ]
        
        with self.driver.session() as session:
            session.run(query, predictions=predictions_list)
            logger.info(f"Updated {len(predictions)} nodes with GNN predictions")
    
    def get_community_statistics(self) -> pd.DataFrame:
        """Get statistics about different community detection methods."""
        query = """
        MATCH (p:Paper)
        RETURN 
            p.subject as true_subject,
            CASE 
                WHEN p.louvainCommunityId IS NOT NULL AND size(p.louvainCommunityId) > 0 
                THEN p.louvainCommunityId[-1]
                ELSE p.louvainCommunityId
            END as louvain_community,
            p.lpaCommunityId as lpa_community,
            p.gnnCommunityId as gnn_community
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            return pd.DataFrame([dict(record) for record in result])
    
    def cleanup_graph_projection(self) -> None:
        """Remove graph projection to free memory."""
        query = "CALL gds.graph.drop('cora-graph')"
        
        try:
            with self.driver.session() as session:
                session.run(query)
                logger.info("Graph projection cleaned up")
        except Exception as e:
            logger.warning(f"Could not cleanup graph projection: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 