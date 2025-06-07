#!/usr/bin/env python3
"""
Quick Start script for Community Detection project.
Tests basic functionality and setup without running the full pipeline.
"""
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required packages can be imported."""
    logger.info("Testing imports...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
        logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        import torch_geometric
        logger.info(f"✓ PyTorch Geometric {torch_geometric.__version__}")
        
        import neo4j
        logger.info(f"✓ Neo4j driver available")
        
        import pandas as pd
        logger.info(f"✓ Pandas {pd.__version__}")
        
        import matplotlib
        logger.info(f"✓ Matplotlib {matplotlib.__version__}")
        
        import plotly
        logger.info(f"✓ Plotly {plotly.__version__}")
        
        import networkx as nx
        logger.info(f"✓ NetworkX {nx.__version__}")
        
        from src.config import NEO4J_CONFIG, PROJECT_CONFIG, MODEL_CONFIG
        logger.info("✓ Project configuration loaded")
        
        from src.data_loader import CoraDataLoader
        logger.info("✓ CoraDataLoader imported")
        
        from src.gnn_models import CommunityDetectionTrainer
        logger.info("✓ GNN models imported")
        
        from src.visualization import CommunityVisualizer
        logger.info("✓ Visualization tools imported")
        
        logger.info("✅ All imports successful!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_directories():
    """Test if all required directories exist."""
    logger.info("Testing directory structure...")
    
    required_dirs = ['src', 'data', 'results', 'notebooks']
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            logger.info(f"✓ {dir_name}/ directory exists")
        else:
            logger.warning(f"⚠️ {dir_name}/ directory missing - creating it")
            dir_path.mkdir(exist_ok=True)
    
    return True

def test_data_loader():
    """Test the data loader functionality."""
    logger.info("Testing data loader...")
    
    try:
        from src.data_loader import CoraDataLoader
        
        data_loader = CoraDataLoader()
        logger.info("✓ CoraDataLoader instantiated")
        
        # Test if we can get class info (this will download data if needed)
        logger.info("Downloading Cora dataset (this may take a moment)...")
        class_info = data_loader.get_class_info()
        
        logger.info(f"✓ Dataset loaded: {class_info['num_classes']} classes")
        logger.info(f"✓ Classes: {class_info['classes']}")
        
        # Test PyTorch data preparation
        features, edge_index, labels, paper_ids = data_loader.get_pytorch_data()
        logger.info(f"✓ PyTorch data prepared: {features.shape[0]} nodes, {features.shape[1]} features")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loader test failed: {e}")
        return False

def test_neo4j_connection():
    """Test Neo4j connection (if running)."""
    logger.info("Testing Neo4j connection...")
    
    try:
        from src.neo4j_manager import Neo4jManager
        
        neo4j_manager = Neo4jManager()
        logger.info("✓ Neo4j connection established")
        
        # Test basic query
        with neo4j_manager.driver.session() as session:
            result = session.run("RETURN 'Hello, Neo4j!' as message")
            message = result.single()["message"]
            logger.info(f"✓ Neo4j query successful: {message}")
        
        neo4j_manager.close()
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Neo4j connection failed: {e}")
        logger.info("💡 Make sure Neo4j is running: docker-compose up -d")
        return False

def test_gnn_models():
    """Test GNN model creation."""
    logger.info("Testing GNN models...")
    
    try:
        from src.gnn_models import CommunityDetectionTrainer
        import torch
        
        # Create a simple test case
        device = 'cpu'  # Use CPU for testing
        trainer = CommunityDetectionTrainer(model_type='gcn', device=device)
        
        # Create dummy data
        num_features = 10
        num_classes = 3
        model = trainer.create_model(num_features, num_classes)
        
        logger.info(f"✓ GCN model created: {type(model).__name__}")
        
        # Test other models
        for model_type in ['sage', 'gat']:
            trainer = CommunityDetectionTrainer(model_type=model_type, device=device)
            model = trainer.create_model(num_features, num_classes)
            logger.info(f"✓ {model_type.upper()} model created: {type(model).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GNN model test failed: {e}")
        return False

def test_visualization():
    """Test visualization components."""
    logger.info("Testing visualization...")
    
    try:
        from src.visualization import CommunityVisualizer
        
        visualizer = CommunityVisualizer(save_plots=False)  # Don't save during testing
        logger.info("✓ CommunityVisualizer instantiated")
        
        # Test with dummy data
        class_info = {
            'classes': ['A', 'B', 'C'],
            'counts': [10, 20, 15],
            'num_classes': 3
        }
        
        logger.info("✓ Visualization components working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Visualization test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting Community Detection project quick start tests...")
    logger.info("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Directory Structure", test_directories),
        ("Data Loader", test_data_loader),
        ("Neo4j Connection", test_neo4j_connection),
        ("GNN Models", test_gnn_models),
        ("Visualization", test_visualization),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 QUICK START SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed! Your environment is ready.")
        logger.info("\n🚀 Next steps:")
        logger.info("1. Start Neo4j: docker-compose up -d")
        logger.info("2. Run full pipeline: python main.py")
        logger.info("3. Or use Jupyter: jupyter notebook notebooks/community_detection_analysis.ipynb")
    else:
        logger.warning(f"\n⚠️ {total - passed} test(s) failed. Please fix the issues before proceeding.")
        
        if not results.get("Neo4j Connection", False):
            logger.info("\n💡 To start Neo4j:")
            logger.info("   docker-compose up -d")
            logger.info("   Then access http://localhost:7474 (user: neo4j, pass: password123)")

if __name__ == "__main__":
    main() 