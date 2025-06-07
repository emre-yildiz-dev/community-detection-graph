"""
Configuration settings for the Community Detection project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
NOTEBOOKS_DIR.mkdir(exist_ok=True)

# Neo4j Configuration
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "username": os.getenv("NEO4J_USERNAME", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "password123"),
    "database": os.getenv("NEO4J_DATABASE", "neo4j")
}

# Project Configuration
PROJECT_CONFIG = {
    "name": os.getenv("PROJECT_NAME", "community-detection"),
    "random_seed": 42,
    "test_size": 0.2,
    "validation_size": 0.1
}

# Model Configuration
MODEL_CONFIG = {
    "hidden_channels": 64,
    "num_layers": 2,
    "dropout": 0.5,
    "learning_rate": 0.01,
    "epochs": 200,
    "patience": 20
} 