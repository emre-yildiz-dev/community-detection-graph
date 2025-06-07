"""
Visualization utilities for Community Detection project.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

from .config import RESULTS_DIR

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CommunityVisualizer:
    """Visualization utilities for community detection results."""
    
    def __init__(self, save_plots: bool = True):
        self.save_plots = save_plots
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(exist_ok=True)
    
    def plot_dataset_statistics(self, class_info: Dict, save_name: str = "dataset_stats") -> None:
        """Plot dataset statistics."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Class distribution
        classes = class_info['classes']
        counts = class_info['counts']
        
        axes[0].bar(range(len(classes)), counts)
        axes[0].set_xlabel('Subject Classes')
        axes[0].set_ylabel('Number of Papers')
        axes[0].set_title('Distribution of Papers by Subject')
        axes[0].set_xticks(range(len(classes)))
        axes[0].set_xticklabels(classes, rotation=45, ha='right')
        
        # Pie chart
        axes[1].pie(counts, labels=classes, autopct='%1.1f%%')
        axes[1].set_title('Subject Distribution')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.results_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_community_comparison(self, community_stats: pd.DataFrame, 
                                save_name: str = "community_comparison") -> None:
        """Compare different community detection methods."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Number of communities found by each method
        methods = ['louvain_community', 'lpa_community', 'gnn_community']
        method_names = ['Louvain', 'Label Propagation', 'GNN']
        
        num_communities = []
        for method in methods:
            if method in community_stats.columns:
                unique_communities = community_stats[method].nunique()
                num_communities.append(unique_communities)
            else:
                num_communities.append(0)
        
        axes[0, 0].bar(method_names, num_communities, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 0].set_title('Number of Communities Detected')
        axes[0, 0].set_ylabel('Number of Communities')
        
        # Community size distributions
        for i, (method, name) in enumerate(zip(methods, method_names)):
            if method in community_stats.columns:
                community_sizes = community_stats.groupby(method).size()
                axes[0, 1].hist(community_sizes, alpha=0.6, label=name, bins=20)
        
        axes[0, 1].set_xlabel('Community Size')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Community Size Distributions')
        axes[0, 1].legend()
        
        # Subject vs Community heatmaps (for first two methods)
        if 'true_subject' in community_stats.columns:
            # Louvain vs True subjects
            if 'louvain_community' in community_stats.columns:
                pivot_louvain = pd.crosstab(community_stats['true_subject'], 
                                          community_stats['louvain_community'])
                sns.heatmap(pivot_louvain, ax=axes[1, 0], cmap='Blues', 
                           cbar_kws={'label': 'Count'})
                axes[1, 0].set_title('True Subject vs Louvain Communities')
                axes[1, 0].set_xlabel('Louvain Community ID')
                axes[1, 0].set_ylabel('True Subject')
            
            # GNN vs True subjects
            if 'gnn_community' in community_stats.columns:
                pivot_gnn = pd.crosstab(community_stats['true_subject'], 
                                      community_stats['gnn_community'])
                sns.heatmap(pivot_gnn, ax=axes[1, 1], cmap='Greens', 
                           cbar_kws={'label': 'Count'})
                axes[1, 1].set_title('True Subject vs GNN Communities')
                axes[1, 1].set_xlabel('GNN Community ID')
                axes[1, 1].set_ylabel('True Subject')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.results_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_evaluation_metrics(self, evaluation_results: Dict[str, Dict], 
                              save_name: str = "evaluation_metrics") -> None:
        """Plot evaluation metrics comparison."""
        methods = list(evaluation_results.keys())
        metrics = ['adjusted_rand_index', 'normalized_mutual_info']
        metric_names = ['Adjusted Rand Index', 'Normalized Mutual Info']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            values = [evaluation_results[method][metric] for method in methods]
            
            bars = axes[i].bar(methods, values, color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[i].set_title(f'{metric_name} Comparison')
            axes[i].set_ylabel(metric_name)
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.results_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_network_plot(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                                      community_column: str = 'louvain_community',
                                      save_name: str = "network_plot") -> None:
        """Create an interactive network plot using Plotly."""
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for _, node in nodes_df.iterrows():
            G.add_node(node['id'], **node.to_dict())
        
        # Add edges (sample if too many)
        if len(edges_df) > 1000:
            edges_sample = edges_df.sample(n=1000, random_state=42)
            logger.info(f"Sampling {len(edges_sample)} edges from {len(edges_df)} for visualization")
        else:
            edges_sample = edges_df
        
        for _, edge in edges_sample.iterrows():
            if edge['source'] in G.nodes and edge['target'] in G.nodes:
                G.add_edge(edge['source'], edge['target'])
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Prepare data for Plotly
        node_trace = go.Scatter(
            x=[], y=[], text=[], mode='markers+text',
            hoverinfo='text', textposition='middle center',
            marker=dict(size=[], color=[], colorscale='Viridis', showscale=True)
        )
        
        edge_trace = go.Scatter(
            x=[], y=[], mode='lines',
            line=dict(width=0.5, color='gray'),
            hoverinfo='none'
        )
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        
        # Add nodes
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            
            node_info = G.nodes[node]
            community = node_info.get(community_column, 'Unknown')
            subject = node_info.get('subject', 'Unknown')
            
            node_trace['text'] += (f"ID: {node}<br>Community: {community}<br>Subject: {subject}",)
            node_trace['marker']['size'] += (8,)
            node_trace['marker']['color'] += (community if community != 'Unknown' else 0,)
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Network Visualization - {community_column.replace("_", " ").title()}',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Hover over nodes to see details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        if self.save_plots:
            fig.write_html(self.results_dir / f"{save_name}.html")
        
        fig.show()
    
    def plot_training_curves(self, training_history: Dict[str, List[float]], 
                           save_name: str = "training_curves") -> None:
        """Plot training curves for GNN models."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = range(1, len(training_history['loss']) + 1)
        
        # Loss curve
        axes[0].plot(epochs, training_history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in training_history:
            axes[0].plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy curve (if available)
        if 'accuracy' in training_history:
            axes[1].plot(epochs, training_history['accuracy'], 'b-', label='Training Accuracy')
            if 'val_accuracy' in training_history:
                axes[1].plot(epochs, training_history['val_accuracy'], 'r-', label='Validation Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training and Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        else:
            axes[1].text(0.5, 0.5, 'No accuracy data available', 
                        transform=axes[1].transAxes, ha='center', va='center')
            axes[1].set_title('Accuracy Not Available')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(self.results_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comparison_dashboard(self, community_stats: pd.DataFrame,
                                 evaluation_results: Dict[str, Dict],
                                 save_name: str = "dashboard") -> None:
        """Create a comprehensive comparison dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Community Count Comparison', 'Evaluation Metrics',
                          'Community Size Distribution', 'Method Performance'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Community count comparison
        methods = ['louvain_community', 'lpa_community', 'gnn_community']
        method_names = ['Louvain', 'Label Propagation', 'GNN']
        
        num_communities = []
        for method in methods:
            if method in community_stats.columns:
                unique_communities = community_stats[method].nunique()
                num_communities.append(unique_communities)
            else:
                num_communities.append(0)
        
        fig.add_trace(
            go.Bar(x=method_names, y=num_communities, name="Communities"),
            row=1, col=1
        )
        
        # Evaluation metrics
        if evaluation_results:
            ari_values = [evaluation_results[method].get('adjusted_rand_index', 0) 
                         for method in evaluation_results.keys()]
            nmi_values = [evaluation_results[method].get('normalized_mutual_info', 0) 
                         for method in evaluation_results.keys()]
            
            fig.add_trace(
                go.Bar(x=list(evaluation_results.keys()), y=ari_values, 
                      name="ARI", marker_color='lightblue'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=list(evaluation_results.keys()), y=nmi_values, 
                      name="NMI", marker_color='lightcoral'),
                row=1, col=2
            )
        
        # Community size distribution
        if 'louvain_community' in community_stats.columns:
            community_sizes = community_stats.groupby('louvain_community').size()
            fig.add_trace(
                go.Histogram(x=community_sizes, name="Louvain", opacity=0.7),
                row=2, col=1
            )
        
        # Method performance overview (best scores)
        if evaluation_results:
            best_ari = max([evaluation_results[method].get('adjusted_rand_index', 0) 
                           for method in evaluation_results.keys()])
            best_nmi = max([evaluation_results[method].get('normalized_mutual_info', 0) 
                           for method in evaluation_results.keys()])
            
            fig.add_trace(
                go.Bar(x=['Best ARI', 'Best NMI'], y=[best_ari, best_nmi], 
                      name="Best Scores", marker_color='green'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Community Detection Methods Comparison Dashboard",
            showlegend=True
        )
        
        if self.save_plots:
            fig.write_html(self.results_dir / f"{save_name}.html")
        
        fig.show()
    
    def save_results_summary(self, evaluation_results: Dict[str, Dict],
                           community_stats: pd.DataFrame,
                           save_name: str = "results_summary") -> None:
        """Save a comprehensive results summary."""
        summary = {
            "evaluation_metrics": evaluation_results,
            "community_statistics": {
                "total_nodes": len(community_stats),
                "methods_compared": list(community_stats.columns),
            }
        }
        
        # Add community counts for each method
        methods = ['louvain_community', 'lpa_community', 'gnn_community']
        for method in methods:
            if method in community_stats.columns:
                summary["community_statistics"][f"{method}_count"] = community_stats[method].nunique()
        
        # Save as JSON and CSV
        import json
        
        with open(self.results_dir / f"{save_name}.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        community_stats.to_csv(self.results_dir / f"{save_name}_detailed.csv", index=False)
        
        logger.info(f"Results summary saved to {self.results_dir}")
        
        return summary 