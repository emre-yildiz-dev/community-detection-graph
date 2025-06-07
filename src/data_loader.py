"""
Data loader for Community Detection project.
Downloads and preprocesses the Cora dataset.
"""
import os
import requests
import tarfile
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .config import DATA_DIR

logger = logging.getLogger(__name__)


class CoraDataLoader:
    """Loads and preprocesses the Cora citation network dataset."""
    
    def __init__(self):
        self.data_dir = DATA_DIR / "cora"
        self.data_dir.mkdir(exist_ok=True)
        self.cora_url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
        
    def download_cora_dataset(self) -> None:
        """Download the Cora dataset if not already present."""
        cora_file = self.data_dir / "cora.tgz"
        
        if cora_file.exists():
            logger.info("Cora dataset already exists")
            return
        
        logger.info("Downloading Cora dataset...")
        try:
            response = requests.get(self.cora_url, stream=True)
            response.raise_for_status()
            
            with open(cora_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the tar file
            with tarfile.open(cora_file, 'r:gz') as tar:
                tar.extractall(self.data_dir)
            
            logger.info("Cora dataset downloaded and extracted successfully")
            
        except Exception as e:
            logger.error(f"Failed to download Cora dataset: {e}")
            raise
    
    def load_cora_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Cora dataset into pandas DataFrames."""
        self.download_cora_dataset()
        
        # Load content file (papers with features and labels)
        content_file = self.data_dir / "cora" / "cora.content"
        if not content_file.exists():
            raise FileNotFoundError(f"Cora content file not found at {content_file}")
        
        # Load citations file (edges)
        cites_file = self.data_dir / "cora" / "cora.cites"
        if not cites_file.exists():
            raise FileNotFoundError(f"Cora cites file not found at {cites_file}")
        
        # Read content file
        content_df = pd.read_csv(content_file, sep='\t', header=None)
        
        # The last column is the label, the first is paper ID, the rest are features
        n_features = content_df.shape[1] - 2
        
        # Create column names
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        content_df.columns = ['paper_id'] + feature_cols + ['subject']
        
        # Read citations file
        cites_df = pd.read_csv(cites_file, sep='\t', header=None)
        cites_df.columns = ['cited_paper_id', 'citing_paper_id']
        
        logger.info(f"Loaded {len(content_df)} papers and {len(cites_df)} citations")
        return content_df, cites_df
    
    def preprocess_for_neo4j(self) -> None:
        """Preprocess and save data in Neo4j-friendly format."""
        content_df, cites_df = self.load_cora_data()
        
        # Prepare content file for Neo4j
        neo4j_content = content_df.copy()
        
        # Combine features into a single string
        feature_cols = [col for col in content_df.columns if col.startswith('feature_')]
        neo4j_content['words'] = neo4j_content[feature_cols].apply(
            lambda row: ','.join(row.astype(str)), axis=1
        )
        
        # Keep only necessary columns
        neo4j_content = neo4j_content[['paper_id', 'words', 'subject']]
        
        # Save files
        neo4j_content_file = self.data_dir / "cora.content"
        neo4j_cites_file = self.data_dir / "cora.cites"
        
        neo4j_content.to_csv(neo4j_content_file, sep='\t', index=False)
        cites_df.to_csv(neo4j_cites_file, sep='\t', index=False, header=False)
        
        logger.info(f"Preprocessed data saved to {neo4j_content_file} and {neo4j_cites_file}")
    
    def get_pytorch_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get data in PyTorch-friendly format."""
        content_df, cites_df = self.load_cora_data()
        
        # Encode labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(content_df['subject'])
        
        # Get features (binary features)
        feature_cols = [col for col in content_df.columns if col.startswith('feature_')]
        features = content_df[feature_cols].values.astype(np.float32)
        
        # Create paper ID to index mapping
        paper_ids = content_df['paper_id'].values
        id_to_idx = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}
        
        # Create edge index
        edges = []
        for _, row in cites_df.iterrows():
            cited_id = row['cited_paper_id']
            citing_id = row['citing_paper_id']
            
            if cited_id in id_to_idx and citing_id in id_to_idx:
                edges.append([id_to_idx[citing_id], id_to_idx[cited_id]])
        
        edge_index = np.array(edges).T
        
        logger.info(f"PyTorch data: {features.shape[0]} nodes, {features.shape[1]} features, "
                   f"{edge_index.shape[1]} edges, {len(set(labels))} classes")
        
        return features, edge_index, labels, paper_ids
    
    def get_class_info(self) -> dict:
        """Get information about classes in the dataset."""
        content_df, _ = self.load_cora_data()
        
        class_counts = content_df['subject'].value_counts()
        class_info = {
            'classes': class_counts.index.tolist(),
            'counts': class_counts.values.tolist(),
            'num_classes': len(class_counts)
        }
        
        return class_info 