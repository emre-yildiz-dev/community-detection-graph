"""
Graph Neural Network models for Community Detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from typing import Dict, Tuple, Optional
import logging

from .config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class GCNCommunityDetector(nn.Module):
    """Graph Convolutional Network for community detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 dropout: float = 0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph convolutional layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the network."""
        x, edge_index = data.x, data.edge_index
        
        # Apply graph convolutions
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GraphSAGECommunityDetector(nn.Module):
    """GraphSAGE model for community detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 dropout: float = 0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the network."""
        x, edge_index = data.x, data.edge_index
        
        # Apply GraphSAGE convolutions
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GATCommunityDetector(nn.Module):
    """Graph Attention Network for community detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 dropout: float = 0.5, heads: int = 4):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Graph attention layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GATConv(input_dim, hidden_dim // heads, heads=heads))
        
        for _ in range(num_layers - 1):
            self.conv_layers.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the network."""
        x, edge_index = data.x, data.edge_index
        
        # Apply graph attention convolutions
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class CommunityDetectionTrainer:
    """Trainer class for community detection models."""
    
    def __init__(self, model_type: str = "gcn", device: str = "cpu"):
        self.model_type = model_type
        self.device = torch.device(device)
        self.model = None
        self.optimizer = None
        
    def create_model(self, input_dim: int, num_classes: int) -> nn.Module:
        """Create the specified model type."""
        config = MODEL_CONFIG
        
        if self.model_type == "gcn":
            model = GCNCommunityDetector(
                input_dim=input_dim,
                hidden_dim=config["hidden_channels"],
                num_layers=config["num_layers"],
                dropout=config["dropout"]
            )
        elif self.model_type == "sage":
            model = GraphSAGECommunityDetector(
                input_dim=input_dim,
                hidden_dim=config["hidden_channels"],
                num_layers=config["num_layers"],
                dropout=config["dropout"]
            )
        elif self.model_type == "gat":
            model = GATCommunityDetector(
                input_dim=input_dim,
                hidden_dim=config["hidden_channels"],
                num_layers=config["num_layers"],
                dropout=config["dropout"]
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def train_unsupervised(self, data: Data, num_communities: int) -> Dict[int, int]:
        """Train model in unsupervised manner using node2vec-like approach."""
        self.model = self.create_model(data.x.size(1), num_communities)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=MODEL_CONFIG["learning_rate"]
        )
        
        data = data.to(self.device)
        
        # Train the model to learn good node representations
        self.model.train()
        for epoch in range(MODEL_CONFIG["epochs"]):
            self.optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.model(data)
            
            # Contrastive loss (simplified)
            loss = self._contrastive_loss(embeddings, data.edge_index)
            
            loss.backward()
            self.optimizer.step()
            
            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch:03d}, Loss: {loss:.4f}")
        
        # Get final embeddings
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(data)
        
        # Cluster embeddings to find communities
        embeddings_np = embeddings.cpu().numpy()
        kmeans = KMeans(n_clusters=num_communities, random_state=42)
        community_labels = kmeans.fit_predict(embeddings_np)
        
        # Create node_id to community mapping
        node_to_community = {}
        for i, community in enumerate(community_labels):
            node_to_community[i] = int(community)
        
        return node_to_community
    
    def train_semi_supervised(self, data: Data, train_mask: torch.Tensor, 
                            val_mask: torch.Tensor, num_classes: int) -> Dict[int, int]:
        """Train model in semi-supervised manner."""
        self.model = self.create_model(data.x.size(1), num_classes)
        
        # Add a classification head
        self.classifier = nn.Linear(MODEL_CONFIG["hidden_channels"], num_classes).to(self.device)
        
        # Combined parameters
        all_params = list(self.model.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=MODEL_CONFIG["learning_rate"])
        
        data = data.to(self.device)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(MODEL_CONFIG["epochs"]):
            # Training
            self.model.train()
            self.classifier.train()
            self.optimizer.zero_grad()
            
            embeddings = self.model(data)
            logits = self.classifier(embeddings)
            
            loss = criterion(logits[train_mask], data.y[train_mask])
            loss.backward()
            self.optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                self.model.eval()
                self.classifier.eval()
                with torch.no_grad():
                    embeddings = self.model(data)
                    logits = self.classifier(embeddings)
                    val_acc = self._calculate_accuracy(logits[val_mask], data.y[val_mask])
                
                logger.info(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= MODEL_CONFIG["patience"]:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Get final predictions
        self.model.eval()
        self.classifier.eval()
        with torch.no_grad():
            embeddings = self.model(data)
            logits = self.classifier(embeddings)
            predictions = torch.argmax(logits, dim=1)
        
        # Create node_id to community mapping
        node_to_community = {}
        for i, community in enumerate(predictions.cpu().numpy()):
            node_to_community[i] = int(community)
        
        return node_to_community
    
    def _contrastive_loss(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Simplified contrastive loss for unsupervised training."""
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Positive pairs (connected nodes)
        pos_pairs = embeddings[edge_index[0]] * embeddings[edge_index[1]]
        pos_loss = -torch.log(torch.sigmoid(pos_pairs.sum(dim=1))).mean()
        
        # Negative pairs (random sampling)
        num_nodes = embeddings.size(0)
        neg_idx = torch.randint(0, num_nodes, (edge_index.size(1), 2), device=embeddings.device)
        neg_pairs = embeddings[neg_idx[:, 0]] * embeddings[neg_idx[:, 1]]
        neg_loss = -torch.log(1 - torch.sigmoid(neg_pairs.sum(dim=1))).mean()
        
        return pos_loss + neg_loss
    
    def _calculate_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Calculate accuracy."""
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == labels).float()
        return correct.mean().item()
    
    def evaluate_communities(self, predicted_communities: Dict[int, int], 
                           true_communities: Dict[int, int]) -> Dict[str, float]:
        """Evaluate community detection results."""
        # Convert to lists for sklearn metrics
        true_labels = []
        pred_labels = []
        
        for node_id in predicted_communities:
            if node_id in true_communities:
                true_labels.append(true_communities[node_id])
                pred_labels.append(predicted_communities[node_id])
        
        # Calculate metrics
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        
        return {
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
            "num_predicted_communities": len(set(pred_labels)),
            "num_true_communities": len(set(true_labels))
        } 