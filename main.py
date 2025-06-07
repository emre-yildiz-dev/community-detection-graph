"""
Main application for Community Detection project.
Compares classical algorithms (Louvain, Label Propagation) with Graph Neural Networks.
"""
import logging
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.config import PROJECT_CONFIG, MODEL_CONFIG
from src.neo4j_manager import Neo4jManager
from src.data_loader import CoraDataLoader
from src.gnn_models import CommunityDetectionTrainer
from src.visualization import CommunityVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('community_detection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def setup_data():
    """Download and prepare the Cora dataset."""
    logger.info("Setting up Cora dataset...")
    
    data_loader = CoraDataLoader()
    
    # Download and preprocess data
    data_loader.preprocess_for_neo4j()
    
    # Get dataset information
    class_info = data_loader.get_class_info()
    logger.info(f"Dataset info: {class_info}")
    
    return data_loader, class_info


def run_classical_algorithms(neo4j_manager: Neo4jManager):
    """Run classical community detection algorithms in Neo4j."""
    logger.info("Running classical community detection algorithms...")
    
    # Load data into Neo4j
    logger.info("Loading data into Neo4j...")
    neo4j_manager.clear_database()
    neo4j_manager.load_cora_dataset()
    
    # Run Louvain algorithm
    logger.info("Running Louvain algorithm...")
    neo4j_manager.run_louvain_algorithm()
    
    # Run Label Propagation algorithm
    logger.info("Running Label Propagation algorithm...")
    neo4j_manager.run_label_propagation()
    
    logger.info("Classical algorithms completed")


def prepare_pytorch_data(data_loader: CoraDataLoader, nodes_df: pd.DataFrame) -> Data:
    """Prepare data for PyTorch Geometric."""
    logger.info("Preparing PyTorch data...")
    
    # Get raw data
    features, edge_index, labels, paper_ids = data_loader.get_pytorch_data()
    
    # Create mapping from paper_id to index
    id_to_idx = {paper_id: idx for idx, paper_id in enumerate(paper_ids)}
    
    # Map community labels
    louvain_communities = np.zeros(len(paper_ids), dtype=int)
    lpa_communities = np.zeros(len(paper_ids), dtype=int)
    
    for _, row in nodes_df.iterrows():
        if row['id'] in id_to_idx:
            idx = id_to_idx[row['id']]
            
            # Handle louvain_community (could be array for hierarchical communities)
            if row['louvain_community'] is not None:
                louvain_val = row['louvain_community']
                if isinstance(louvain_val, (list, tuple)) and len(louvain_val) > 0:
                    # Take the final (most refined) community level
                    louvain_communities[idx] = int(louvain_val[-1])
                elif not pd.isna(louvain_val):
                    louvain_communities[idx] = int(louvain_val)
            
            # Handle lpa_community (single value)
            if row['lpa_community'] is not None and not pd.isna(row['lpa_community']):
                lpa_communities[idx] = int(row['lpa_community'])
    
    # Create PyTorch Geometric data object
    data = Data(
        x=torch.FloatTensor(features),
        edge_index=torch.LongTensor(edge_index),
        y=torch.LongTensor(labels),
        paper_ids=torch.LongTensor(paper_ids),
        louvain_communities=torch.LongTensor(louvain_communities),
        lpa_communities=torch.LongTensor(lpa_communities)
    )
    
    logger.info(f"PyTorch data prepared: {data}")
    return data


def run_gnn_experiments(data: Data) -> dict:
    """Run GNN experiments with different models."""
    logger.info("Running GNN experiments...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    results = {}
    model_types = ['gcn', 'sage', 'gat']
    
    # Get number of unique communities from classical methods
    num_louvain_communities = len(torch.unique(data.louvain_communities))
    num_lpa_communities = len(torch.unique(data.lpa_communities))
    num_true_classes = len(torch.unique(data.y))
    
    logger.info(f"Number of communities - Louvain: {num_louvain_communities}, "
               f"LPA: {num_lpa_communities}, True classes: {num_true_classes}")
    
    for model_type in model_types:
        logger.info(f"Training {model_type.upper()} model...")
        
        trainer = CommunityDetectionTrainer(model_type=model_type, device=str(device))
        
        # Try unsupervised learning first (clustering based on embeddings)
        try:
            gnn_communities = trainer.train_unsupervised(data, num_louvain_communities)
            results[f'{model_type}_unsupervised'] = gnn_communities
            logger.info(f"{model_type.upper()} unsupervised training completed")
        except Exception as e:
            logger.error(f"Error in {model_type} unsupervised training: {e}")
        
        # Try semi-supervised learning (if we have labeled data)
        try:
            # Create train/val splits
            num_nodes = data.x.size(0)
            indices = np.arange(num_nodes)
            train_indices, val_indices = train_test_split(
                indices, test_size=0.3, random_state=PROJECT_CONFIG["random_seed"]
            )
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[train_indices] = True
            val_mask[val_indices] = True
            
            trainer_semi = CommunityDetectionTrainer(model_type=model_type, device=str(device))
            gnn_communities_semi = trainer_semi.train_semi_supervised(
                data, train_mask, val_mask, num_true_classes
            )
            results[f'{model_type}_semi_supervised'] = gnn_communities_semi
            logger.info(f"{model_type.upper()} semi-supervised training completed")
            
        except Exception as e:
            logger.error(f"Error in {model_type} semi-supervised training: {e}")
    
    return results


def evaluate_results(gnn_results: dict, data: Data, neo4j_manager: Neo4jManager) -> dict:
    """Evaluate and compare all community detection methods."""
    logger.info("Evaluating results...")
    
    # Get true communities (original subject labels)
    true_communities = {}
    for i, (paper_id, label) in enumerate(zip(data.paper_ids.numpy(), data.y.numpy())):
        true_communities[i] = int(label)
    
    # Get classical algorithm results
    louvain_communities = {}
    lpa_communities = {}
    for i, (louvain, lpa) in enumerate(zip(data.louvain_communities.numpy(), 
                                          data.lpa_communities.numpy())):
        louvain_communities[i] = int(louvain)
        lpa_communities[i] = int(lpa)
    
    # Evaluate all methods
    evaluation_results = {}
    
    # Evaluate classical methods
    trainer = CommunityDetectionTrainer()  # Just for evaluation functions
    
    eval_louvain = trainer.evaluate_communities(louvain_communities, true_communities)
    eval_lpa = trainer.evaluate_communities(lpa_communities, true_communities)
    
    evaluation_results['Louvain'] = eval_louvain
    evaluation_results['Label_Propagation'] = eval_lpa
    
    # Evaluate GNN methods
    for method_name, gnn_communities in gnn_results.items():
        eval_gnn = trainer.evaluate_communities(gnn_communities, true_communities)
        evaluation_results[method_name] = eval_gnn
    
    # Write best GNN results back to Neo4j
    if gnn_results:
        best_method = max(gnn_results.keys(), 
                         key=lambda x: evaluation_results[x]['adjusted_rand_index'])
        best_communities = gnn_results[best_method]
        
        # Convert indices back to paper IDs
        paper_id_communities = {}
        for idx, community in best_communities.items():
            paper_id = int(data.paper_ids[idx])
            paper_id_communities[paper_id] = community
        
        neo4j_manager.write_gnn_predictions(paper_id_communities)
        logger.info(f"Best GNN method ({best_method}) results written to Neo4j")
    
    return evaluation_results


def create_visualizations(neo4j_manager: Neo4jManager, evaluation_results: dict, 
                        class_info: dict):
    """Create comprehensive visualizations."""
    logger.info("Creating visualizations...")
    
    visualizer = CommunityVisualizer()
    
    # Plot dataset statistics
    visualizer.plot_dataset_statistics(class_info)
    
    # Get community statistics from Neo4j
    community_stats = neo4j_manager.get_community_statistics()
    
    # Plot community comparison
    visualizer.plot_community_comparison(community_stats)
    
    # Plot evaluation metrics
    visualizer.plot_evaluation_metrics(evaluation_results)
    
    # Get graph data for network visualization
    nodes_df, edges_df = neo4j_manager.get_graph_data()
    
    # Create interactive network plots
    visualizer.create_interactive_network_plot(nodes_df, edges_df, 'louvain_community', 'louvain_network')
    visualizer.create_interactive_network_plot(nodes_df, edges_df, 'gnn_community', 'gnn_network')
    
    # Create comparison dashboard
    visualizer.create_comparison_dashboard(community_stats, evaluation_results)
    
    # Save results summary
    summary = visualizer.save_results_summary(evaluation_results, community_stats)
    
    return summary


def main():
    """Main execution function."""
    logger.info("Starting Community Detection project...")
    
    try:
        # Step 1: Setup data
        data_loader, class_info = setup_data()
        
        # Step 2: Run classical algorithms
        with Neo4jManager() as neo4j_manager:
            run_classical_algorithms(neo4j_manager)
            
            # Step 3: Get data for PyTorch
            nodes_df, edges_df = neo4j_manager.get_graph_data()
            data = prepare_pytorch_data(data_loader, nodes_df)
            
            # Step 4: Run GNN experiments
            gnn_results = run_gnn_experiments(data)
            
            # Step 5: Evaluate results
            evaluation_results = evaluate_results(gnn_results, data, neo4j_manager)
            
            # Step 6: Create visualizations
            summary = create_visualizations(neo4j_manager, evaluation_results, class_info)
            
            # Cleanup
            neo4j_manager.cleanup_graph_projection()
        
        # Print final results
        logger.info("=== FINAL RESULTS ===")
        for method, metrics in evaluation_results.items():
            logger.info(f"{method}:")
            logger.info(f"  ARI: {metrics['adjusted_rand_index']:.4f}")
            logger.info(f"  NMI: {metrics['normalized_mutual_info']:.4f}")
            logger.info(f"  Communities: {metrics['num_predicted_communities']}")
        
        logger.info("Community Detection project completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main() 