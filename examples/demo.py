#!/usr/bin/env python3
"""
Ciyaar Intelligence Platform Demo
================================

Comprehensive demonstration of the quantum-neural AI platform capabilities
across different industries and use cases.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from loguru import logger
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ciyaar_intelligence import CiyaarPlatform, NetworkConfig, PlatformConfig


def generate_sample_data(
    n_samples: int = 1000,
    input_dim: int = 20,
    output_dim: int = 5,
    complexity: str = "medium"
) -> tuple:
    """Generate synthetic data for demonstration."""
    
    logger.info(f"Generating {n_samples} samples with complexity: {complexity}")
    
    if complexity == "simple":
        # Linear relationship with noise
        X = torch.randn(n_samples, input_dim)
        weights = torch.randn(input_dim, output_dim) * 0.5
        y = X @ weights + torch.randn(n_samples, output_dim) * 0.1
        
    elif complexity == "medium":
        # Nonlinear relationship
        X = torch.randn(n_samples, input_dim)
        hidden = torch.tanh(X @ torch.randn(input_dim, input_dim // 2))
        y = hidden @ torch.randn(input_dim // 2, output_dim) + torch.randn(n_samples, output_dim) * 0.2
        
    elif complexity == "high":
        # Complex nonlinear with temporal dependencies
        X = torch.randn(n_samples, input_dim)
        # Add temporal correlations
        for i in range(1, n_samples):
            X[i] = 0.7 * X[i] + 0.3 * X[i-1]
        
        # Complex transformation
        hidden1 = torch.sigmoid(X @ torch.randn(input_dim, input_dim) * 0.5)
        hidden2 = torch.tanh(hidden1 @ torch.randn(input_dim, input_dim // 2))
        y = hidden2 @ torch.randn(input_dim // 2, output_dim) + torch.randn(n_samples, output_dim) * 0.3
    
    return X, y


def demo_basic_functionality():
    """Demonstrate basic platform functionality."""
    
    logger.info("=" * 60)
    logger.info("DEMO: Basic Platform Functionality")
    logger.info("=" * 60)
    
    # Initialize platform
    config = PlatformConfig(
        model_dir="./demo_models",
        auto_optimize=True,
        gpu_acceleration=torch.cuda.is_available()
    )
    
    platform = CiyaarPlatform(config)
    
    # Generate sample data
    X_train, y_train = generate_sample_data(800, 20, 5, "medium")
    X_val, y_val = generate_sample_data(200, 20, 5, "medium")
    
    # Create a quantum-neural model
    model = platform.create_model(
        model_name="demo_basic",
        input_dim=20,
        output_dim=5,
        industry_mode="research"
    )
    
    logger.info(f"Model architecture summary:\n{model.get_architecture_summary()}")
    
    # Train the model
    training_results = platform.train_model(
        model_name="demo_basic",
        train_data=X_train,
        train_targets=y_train,
        validation_data=X_val,
        validation_targets=y_val,
        epochs=50
    )
    
    logger.info(f"Training completed. Best validation loss: {training_results.get('best_validation_loss', 'N/A')}")
    
    # Make predictions
    X_test, y_test = generate_sample_data(100, 20, 5, "medium")
    predictions, uncertainty = platform.predict(
        model_name="demo_basic",
        input_data=X_test,
        return_uncertainty=True
    )
    
    # Evaluate predictions
    mse = torch.nn.functional.mse_loss(predictions, y_test)
    logger.info(f"Test MSE: {mse.item():.4f}")
    logger.info(f"Average uncertainty: {uncertainty.mean().item():.4f}")
    
    # Get performance metrics
    performance = platform.get_model_performance("demo_basic")
    logger.info(f"Quantum advantage score: {performance.get('quantum_advantage_score', 0):.4f}")
    
    return platform


def demo_industry_specialization():
    """Demonstrate industry-specific optimizations."""
    
    logger.info("=" * 60)
    logger.info("DEMO: Industry Specialization")
    logger.info("=" * 60)
    
    config = PlatformConfig(model_dir="./demo_models")
    platform = CiyaarPlatform(config)
    
    industries = ["finance", "healthcare", "manufacturing", "research"]
    
    for industry in industries:
        logger.info(f"\n--- Testing {industry.upper()} specialization ---")
        
        # Generate industry-specific data
        if industry == "finance":
            # Time series financial data
            X, y = generate_sample_data(500, 15, 3, "high")
        elif industry == "healthcare":
            # Medical signal data
            X, y = generate_sample_data(400, 25, 2, "medium")
        elif industry == "manufacturing":
            # Process monitoring data
            X, y = generate_sample_data(600, 12, 4, "medium")
        else:  # research
            # Complex research data
            X, y = generate_sample_data(300, 30, 6, "high")
        
        # Create industry-specific model
        model_name = f"demo_{industry}"
        model = platform.create_model(
            model_name=model_name,
            input_dim=X.shape[1],
            output_dim=y.shape[1],
            industry_mode=industry
        )
        
        # Quick training
        training_results = platform.train_model(
            model_name=model_name,
            train_data=X[:300],
            train_targets=y[:300],
            validation_data=X[300:],
            validation_targets=y[300:],
            epochs=30
        )
        
        # Industry optimization
        optimization_results = platform.optimize_for_industry(
            model_name=model_name,
            industry=industry,
            sample_data=X[:100]
        )
        
        logger.info(f"Industry optimization completed for {industry}")
        logger.info(f"Optimization results: {list(optimization_results.keys())}")
        
        # Performance evaluation
        performance = platform.get_model_performance(model_name)
        logger.info(f"Final performance score: {performance.get('quantum_advantage_score', 0):.4f}")
    
    return platform


def demo_quantum_advantage():
    """Demonstrate quantum advantage benchmarking."""
    
    logger.info("=" * 60)
    logger.info("DEMO: Quantum Advantage Benchmarking")
    logger.info("=" * 60)
    
    config = PlatformConfig(model_dir="./demo_models")
    platform = CiyaarPlatform(config)
    
    # Create quantum model
    X, y = generate_sample_data(500, 16, 4, "high")
    
    quantum_model = platform.create_model(
        model_name="quantum_benchmark",
        input_dim=16,
        output_dim=4
    )
    
    # Train quantum model
    platform.train_model(
        model_name="quantum_benchmark",
        train_data=X[:400],
        train_targets=y[:400],
        validation_data=X[400:],
        validation_targets=y[400:],
        epochs=40
    )
    
    # Create classical baseline
    class ClassicalBaseline(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, output_dim)
            )
        
        def forward(self, x):
            return self.network(x)
    
    classical_model = ClassicalBaseline(16, 4)
    
    # Benchmark quantum advantage
    benchmark_data = X[:100]
    benchmark_results = platform.benchmark_quantum_advantage(
        model_name="quantum_benchmark",
        benchmark_data=benchmark_data,
        classical_baseline=classical_model
    )
    
    logger.info("Quantum Advantage Benchmark Results:")
    for metric, value in benchmark_results.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return platform


def demo_adaptive_architecture():
    """Demonstrate adaptive architecture capabilities."""
    
    logger.info("=" * 60)
    logger.info("DEMO: Adaptive Architecture")
    logger.info("=" * 60)
    
    config = PlatformConfig(model_dir="./demo_models")
    platform = CiyaarPlatform(config)
    
    # Test with different complexity levels
    complexities = ["simple", "medium", "high"]
    
    for complexity in complexities:
        logger.info(f"\n--- Testing {complexity} complexity adaptation ---")
        
        X, y = generate_sample_data(600, 20, 5, complexity)
        
        model_name = f"adaptive_{complexity}"
        model = platform.create_model(
            model_name=model_name,
            input_dim=20,
            output_dim=5
        )
        
        # Get initial architecture stats
        initial_stats = model.adaptive_arch.get_adaptation_stats() if hasattr(model, 'adaptive_arch') else {}
        logger.info(f"Initial blocks: {initial_stats.get('num_blocks', 'N/A')}")
        
        # Train with adaptation enabled
        training_results = platform.train_model(
            model_name=model_name,
            train_data=X[:400],
            train_targets=y[:400],
            validation_data=X[400:500],
            validation_targets=y[400:500],
            epochs=40
        )
        
        # Get final architecture stats
        final_stats = model.adaptive_arch.get_adaptation_stats() if hasattr(model, 'adaptive_arch') else {}
        logger.info(f"Final blocks: {final_stats.get('num_blocks', 'N/A')}")
        logger.info(f"Adaptation history: {len(final_stats.get('adaptation_history', []))}")
        
        # Test task-specific optimization
        if hasattr(model, 'adaptive_arch'):
            task_optimization = model.adaptive_arch.optimize_for_task(X[500:], target_complexity=0.5)
            logger.info(f"Task optimization ratio: {task_optimization.get('optimization_ratio', 1.0):.3f}")
    
    return platform


def demo_comprehensive_workflow():
    """Demonstrate a comprehensive AI workflow."""
    
    logger.info("=" * 60)
    logger.info("DEMO: Comprehensive AI Workflow")
    logger.info("=" * 60)
    
    config = PlatformConfig(
        model_dir="./demo_models",
        auto_optimize=True,
        industry_mode="research"
    )
    platform = CiyaarPlatform(config)
    
    # Step 1: Data preparation
    logger.info("Step 1: Data Preparation")
    X_train, y_train = generate_sample_data(1000, 25, 8, "high")
    X_val, y_val = generate_sample_data(200, 25, 8, "high")
    X_test, y_test = generate_sample_data(200, 25, 8, "high")
    
    # Step 2: Model creation and configuration
    logger.info("Step 2: Model Creation")
    network_config = NetworkConfig(
        n_qubits=6,
        n_layers=3,
        quantum_depth=4,
        use_gpu=torch.cuda.is_available(),
        mixed_precision=True
    )
    
    model = platform.create_model(
        model_name="comprehensive_demo",
        input_dim=25,
        output_dim=8,
        industry_mode="research",
        network_config=network_config
    )
    
    # Step 3: Training with monitoring
    logger.info("Step 3: Model Training")
    training_results = platform.train_model(
        model_name="comprehensive_demo",
        train_data=X_train,
        train_targets=y_train,
        validation_data=X_val,
        validation_targets=y_val,
        epochs=60
    )
    
    # Step 4: Industry optimization
    logger.info("Step 4: Industry Optimization")
    optimization_results = platform.optimize_for_industry(
        model_name="comprehensive_demo",
        industry="research",
        sample_data=X_val
    )
    
    # Step 5: Performance evaluation
    logger.info("Step 5: Performance Evaluation")
    predictions, uncertainty = platform.predict(
        model_name="comprehensive_demo",
        input_data=X_test,
        return_uncertainty=True
    )
    
    test_mse = torch.nn.functional.mse_loss(predictions, y_test)
    logger.info(f"Final test MSE: {test_mse.item():.6f}")
    
    # Step 6: Model analysis
    logger.info("Step 6: Model Analysis")
    performance = platform.get_model_performance("comprehensive_demo")
    platform_stats = platform.get_platform_stats()
    
    logger.info("Performance Summary:")
    logger.info(f"  Quantum advantage score: {performance.get('quantum_advantage_score', 0):.4f}")
    logger.info(f"  Total parameters: {performance['training_stats'].get('total_parameters', 'N/A')}")
    logger.info(f"  Average forward time: {performance['training_stats'].get('avg_forward_time', 0):.4f}s")
    
    logger.info("Platform Summary:")
    logger.info(f"  Active models: {platform_stats['active_models']}")
    logger.info(f"  Total registered models: {platform_stats['total_registered_models']}")
    
    # Step 7: Model persistence
    logger.info("Step 7: Model Persistence")
    saved_path = platform.save_model("comprehensive_demo")
    logger.info(f"Model saved to: {saved_path}")
    
    return platform


def visualize_results(platform: CiyaarPlatform, model_name: str, test_data: tuple):
    """Visualize model predictions and performance."""
    
    logger.info("Generating visualizations...")
    
    X_test, y_test = test_data
    predictions, uncertainty = platform.predict(
        model_name=model_name,
        input_data=X_test,
        return_uncertainty=True
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Predictions vs True values
    axes[0, 0].scatter(y_test[:, 0].numpy(), predictions[:, 0].detach().numpy(), alpha=0.6)
    axes[0, 0].plot([y_test[:, 0].min(), y_test[:, 0].max()], 
                    [y_test[:, 0].min(), y_test[:, 0].max()], 'r--')
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predictions')
    axes[0, 0].set_title('Predictions vs True Values')
    
    # Plot 2: Prediction uncertainty
    axes[0, 1].hist(uncertainty[:, 0].detach().numpy(), bins=20, alpha=0.7)
    axes[0, 1].set_xlabel('Uncertainty')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Prediction Uncertainty Distribution')
    
    # Plot 3: Error vs Uncertainty
    error = torch.abs(predictions[:, 0] - y_test[:, 0])
    axes[1, 0].scatter(uncertainty[:, 0].detach().numpy(), error.numpy(), alpha=0.6)
    axes[1, 0].set_xlabel('Uncertainty')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Error vs Uncertainty')
    
    # Plot 4: Performance metrics
    performance = platform.get_model_performance(model_name)
    training_stats = performance.get('training_stats', {})
    
    if 'forward_pass_time' in training_stats:
        axes[1, 1].plot(training_stats['forward_pass_time'][-100:])  # Last 100 forward passes
        axes[1, 1].set_xlabel('Forward Pass')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].set_title('Forward Pass Time')
    
    plt.tight_layout()
    plt.savefig('ciyaar_demo_results.png', dpi=300, bbox_inches='tight')
    logger.info("Visualizations saved to 'ciyaar_demo_results.png'")


def main():
    """Run the complete Ciyaar Intelligence Platform demonstration."""
    
    logger.info("🚀 Starting Ciyaar Intelligence Platform Demo")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Run all demonstrations
        demos = [
            ("Basic Functionality", demo_basic_functionality),
            ("Industry Specialization", demo_industry_specialization),
            ("Quantum Advantage", demo_quantum_advantage),
            ("Adaptive Architecture", demo_adaptive_architecture),
            ("Comprehensive Workflow", demo_comprehensive_workflow)
        ]
        
        platforms = {}
        
        for demo_name, demo_func in demos:
            logger.info(f"\n🔬 Running {demo_name} Demo...")
            demo_start = time.time()
            
            try:
                platform = demo_func()
                platforms[demo_name] = platform
                
                demo_time = time.time() - demo_start
                logger.info(f"✅ {demo_name} completed in {demo_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ {demo_name} failed: {e}")
                continue
        
        # Generate final summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 DEMO SUMMARY")
        logger.info("=" * 80)
        
        total_time = time.time() - start_time
        logger.info(f"Total demo time: {total_time:.2f}s")
        logger.info(f"Demos completed: {len(platforms)}")
        
        # Visualize results from comprehensive demo
        if "Comprehensive Workflow" in platforms:
            platform = platforms["Comprehensive Workflow"]
            X_test, y_test = generate_sample_data(100, 25, 8, "high")
            try:
                visualize_results(platform, "comprehensive_demo", (X_test, y_test))
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")
        
        # Platform statistics
        if platforms:
            platform = list(platforms.values())[0]
            final_stats = platform.get_platform_stats()
            
            logger.info("\n📈 Final Platform Statistics:")
            for key, value in final_stats.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}: {len(value)} items")
                else:
                    logger.info(f"  {key}: {value}")
        
        logger.info("\n🎉 Ciyaar Intelligence Platform Demo completed successfully!")
        logger.info("🔗 Visit https://github.com/ciyaar-intelligence for more information")
        
    except Exception as e:
        logger.error(f"💥 Demo failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        for platform in platforms.values():
            try:
                platform.shutdown()
            except:
                pass


if __name__ == "__main__":
    main()