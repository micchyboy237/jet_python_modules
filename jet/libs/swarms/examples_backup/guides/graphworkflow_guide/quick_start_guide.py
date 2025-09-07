#!/usr/bin/env python3
"""
GraphWorkflow Quick Start Guide
==============================

This script provides a step-by-step introduction to Swarms' GraphWorkflow system.
Perfect for developers who want to get started quickly with multi-agent workflows.

Installation:
    uv pip install swarms

Usage:
    python quick_start_guide.py
"""

from swarms import Agent
from swarms.structs.graph_workflow import GraphWorkflow


def step_1_basic_setup():
    """Step 1: Create your first GraphWorkflow with two agents."""

    print("🚀 STEP 1: Basic GraphWorkflow Setup")
    print("=" * 50)

    # Create two simple agents
    print("📝 Creating agents...")

    researcher = Agent(
        agent_name="Researcher",
        model_name="ollama/llama3.2",  # Use cost-effective model for demo
        max_loops=1,
        system_prompt="You are a research specialist. Gather and analyze information on the given topic.",
        verbose=False,
    )

    writer = Agent(
        agent_name="Writer",
        model_name="ollama/llama3.2",
        max_loops=1,
        system_prompt="You are a content writer. Create engaging content based on research findings.",
        verbose=False,
    )

    print(
        f"✅ Created agents: {researcher.agent_name}, {writer.agent_name}"
    )

    # Create workflow
    print("\n🔧 Creating workflow...")

    workflow = GraphWorkflow(
        name="MyFirstWorkflow",
        description="A simple research and writing workflow",
        verbose=True,  # Enable detailed logging
        auto_compile=True,  # Automatically optimize the workflow
    )

    print(f"✅ Created workflow: {workflow.name}")

    # Add agents to workflow
    print("\n➕ Adding agents to workflow...")

    workflow.add_node(researcher)
    workflow.add_node(writer)

    print(f"✅ Added {len(workflow.nodes)} agents to workflow")

    # Connect agents
    print("\n🔗 Connecting agents...")

    workflow.add_edge(
        "Researcher", "Writer"
    )  # Researcher feeds into Writer

    print(f"✅ Added {len(workflow.edges)} connections")

    # Set entry and exit points
    print("\n🎯 Setting entry and exit points...")

    workflow.set_entry_points(["Researcher"])  # Start with Researcher
    workflow.set_end_points(["Writer"])  # End with Writer

    print("✅ Entry point: Researcher")
    print("✅ Exit point: Writer")

    return workflow


def step_2_run_workflow(workflow):
    """Step 2: Execute the workflow with a task."""

    print("\n🚀 STEP 2: Running Your First Workflow")
    print("=" * 50)

    # Define a task
    task = "Research the benefits of electric vehicles and write a compelling article about why consumers should consider making the switch."

    print(f"📋 Task: {task}")

    # Execute workflow
    print("\n⚡ Executing workflow...")

    results = workflow.run(task=task)

    print(
        f"✅ Workflow completed! Got results from {len(results)} agents."
    )

    # Display results
    print("\n📊 Results:")
    print("-" * 30)

    for agent_name, result in results.items():
        print(f"\n🤖 {agent_name}:")
        print(
            f"📝 {result[:300]}{'...' if len(result) > 300 else ''}"
        )

    return results


def step_3_parallel_processing():
    """Step 3: Create a workflow with parallel processing."""

    print("\n🚀 STEP 3: Parallel Processing")
    print("=" * 50)

    # Create multiple specialist agents
    print("👥 Creating specialist agents...")

    tech_analyst = Agent(
        agent_name="TechAnalyst",
        model_name="ollama/llama3.2",
        max_loops=1,
        system_prompt="You are a technology analyst. Focus on technical specifications, performance, and innovation.",
        verbose=False,
    )

    market_analyst = Agent(
        agent_name="MarketAnalyst",
        model_name="ollama/llama3.2",
        max_loops=1,
        system_prompt="You are a market analyst. Focus on market trends, pricing, and consumer adoption.",
        verbose=False,
    )

    environmental_analyst = Agent(
        agent_name="EnvironmentalAnalyst",
        model_name="ollama/llama3.2",
        max_loops=1,
        system_prompt="You are an environmental analyst. Focus on sustainability, emissions, and environmental impact.",
        verbose=False,
    )

    synthesizer = Agent(
        agent_name="Synthesizer",
        model_name="ollama/llama3.2",
        max_loops=1,
        system_prompt="You are a synthesis expert. Combine insights from multiple analysts into a comprehensive conclusion.",
        verbose=False,
    )

    print(f"✅ Created {4} specialist agents")

    # Create parallel workflow
    print("\n🔧 Creating parallel workflow...")

    parallel_workflow = GraphWorkflow(
        name="ParallelAnalysisWorkflow",
        description="Multi-specialist analysis with parallel processing",
        verbose=True,
        auto_compile=True,
    )

    # Add all agents
    agents = [
        tech_analyst,
        market_analyst,
        environmental_analyst,
        synthesizer,
    ]
    for agent in agents:
        parallel_workflow.add_node(agent)

    print(f"✅ Added {len(agents)} agents to parallel workflow")

    # Create parallel pattern: Multiple analysts feed into synthesizer
    print("\n🔀 Setting up parallel processing pattern...")

    # All analysts run in parallel, then feed into synthesizer
    parallel_workflow.add_edges_to_target(
        ["TechAnalyst", "MarketAnalyst", "EnvironmentalAnalyst"],
        "Synthesizer",
    )

    # Set multiple entry points (parallel execution)
    parallel_workflow.set_entry_points(
        ["TechAnalyst", "MarketAnalyst", "EnvironmentalAnalyst"]
    )
    parallel_workflow.set_end_points(["Synthesizer"])

    print("✅ Parallel pattern configured:")
    print("   📤 3 analysts run in parallel")
    print("   📥 Results feed into synthesizer")

    # Execute parallel workflow
    task = "Analyze the future of renewable energy technology from technical, market, and environmental perspectives."

    print("\n⚡ Executing parallel workflow...")
    print(f"📋 Task: {task}")

    results = parallel_workflow.run(task=task)

    print(
        f"✅ Parallel execution completed! {len(results)} agents processed."
    )

    # Display results
    print("\n📊 Parallel Analysis Results:")
    print("-" * 40)

    for agent_name, result in results.items():
        print(f"\n🤖 {agent_name}:")
        print(
            f"📝 {result[:250]}{'...' if len(result) > 250 else ''}"
        )

    return parallel_workflow, results


def step_4_advanced_patterns():
    """Step 4: Demonstrate advanced workflow patterns."""

    print("\n🚀 STEP 4: Advanced Workflow Patterns")
    print("=" * 50)

    # Create agents for different patterns
    data_collector = Agent(
        agent_name="DataCollector",
        model_name="ollama/llama3.2",
        max_loops=1,
        system_prompt="You collect and organize data from various sources.",
        verbose=False,
    )

    processor_a = Agent(
        agent_name="ProcessorA",
        model_name="ollama/llama3.2",
        max_loops=1,
        system_prompt="You are processor A specializing in quantitative analysis.",
        verbose=False,
    )

    processor_b = Agent(
        agent_name="ProcessorB",
        model_name="ollama/llama3.2",
        max_loops=1,
        system_prompt="You are processor B specializing in qualitative analysis.",
        verbose=False,
    )

    validator_x = Agent(
        agent_name="ValidatorX",
        model_name="ollama/llama3.2",
        max_loops=1,
        system_prompt="You are validator X focusing on accuracy and consistency.",
        verbose=False,
    )

    validator_y = Agent(
        agent_name="ValidatorY",
        model_name="ollama/llama3.2",
        max_loops=1,
        system_prompt="You are validator Y focusing on completeness and quality.",
        verbose=False,
    )

    final_reporter = Agent(
        agent_name="FinalReporter",
        model_name="ollama/llama3.2",
        max_loops=1,
        system_prompt="You create final comprehensive reports from all validated analyses.",
        verbose=False,
    )

    # Create advanced workflow
    advanced_workflow = GraphWorkflow(
        name="AdvancedPatternsWorkflow",
        description="Demonstrates fan-out, parallel chains, and fan-in patterns",
        verbose=True,
        auto_compile=True,
    )

    # Add all agents
    agents = [
        data_collector,
        processor_a,
        processor_b,
        validator_x,
        validator_y,
        final_reporter,
    ]
    for agent in agents:
        advanced_workflow.add_node(agent)

    print(f"✅ Created advanced workflow with {len(agents)} agents")

    # Demonstrate different patterns
    print("\n🎯 Setting up advanced patterns...")

    # Pattern 1: Fan-out (one-to-many)
    print("   📤 Fan-out: DataCollector → Multiple Processors")
    advanced_workflow.add_edges_from_source(
        "DataCollector", ["ProcessorA", "ProcessorB"]
    )

    # Pattern 2: Parallel chain (many-to-many)
    print("   🔗 Parallel chain: Processors → Validators")
    advanced_workflow.add_parallel_chain(
        ["ProcessorA", "ProcessorB"], ["ValidatorX", "ValidatorY"]
    )

    # Pattern 3: Fan-in (many-to-one)
    print("   📥 Fan-in: Validators → Final Reporter")
    advanced_workflow.add_edges_to_target(
        ["ValidatorX", "ValidatorY"], "FinalReporter"
    )

    # Set workflow boundaries
    advanced_workflow.set_entry_points(["DataCollector"])
    advanced_workflow.set_end_points(["FinalReporter"])

    print("✅ Advanced patterns configured")

    # Show workflow structure
    print("\n📊 Workflow structure:")
    try:
        advanced_workflow.visualize_simple()
    except:
        print("   (Text visualization not available)")

    # Execute advanced workflow
    task = "Analyze the impact of artificial intelligence on job markets, including both opportunities and challenges."

    print("\n⚡ Executing advanced workflow...")

    results = advanced_workflow.run(task=task)

    print(
        f"✅ Advanced execution completed! {len(results)} agents processed."
    )

    return advanced_workflow, results


def step_5_workflow_features():
    """Step 5: Explore additional workflow features."""

    print("\n🚀 STEP 5: Additional Workflow Features")
    print("=" * 50)

    # Create a simple workflow for feature demonstration
    agent1 = Agent(
        agent_name="FeatureTestAgent1",
        model_name="ollama/llama3.2",
        max_loops=1,
        system_prompt="You are a feature testing agent.",
        verbose=False,
    )

    agent2 = Agent(
        agent_name="FeatureTestAgent2",
        model_name="ollama/llama3.2",
        max_loops=1,
        system_prompt="You are another feature testing agent.",
        verbose=False,
    )

    workflow = GraphWorkflow(
        name="FeatureTestWorkflow",
        description="Workflow for testing additional features",
        verbose=True,
        auto_compile=True,
    )

    workflow.add_node(agent1)
    workflow.add_node(agent2)
    workflow.add_edge("FeatureTestAgent1", "FeatureTestAgent2")

    # Feature 1: Compilation status
    print("🔍 Feature 1: Compilation Status")
    status = workflow.get_compilation_status()
    print(f"   ✅ Compiled: {status['is_compiled']}")
    print(f"   📊 Layers: {status.get('cached_layers_count', 'N/A')}")
    print(f"   ⚡ Workers: {status.get('max_workers', 'N/A')}")

    # Feature 2: Workflow validation
    print("\n🔍 Feature 2: Workflow Validation")
    validation = workflow.validate(auto_fix=True)
    print(f"   ✅ Valid: {validation['is_valid']}")
    print(f"   ⚠️ Warnings: {len(validation['warnings'])}")
    print(f"   ❌ Errors: {len(validation['errors'])}")

    # Feature 3: JSON serialization
    print("\n🔍 Feature 3: JSON Serialization")
    try:
        json_data = workflow.to_json()
        print(
            f"   ✅ JSON export successful ({len(json_data)} characters)"
        )

        # Test deserialization
        restored = GraphWorkflow.from_json(json_data)
        print(
            f"   ✅ JSON import successful ({len(restored.nodes)} nodes)"
        )
    except Exception as e:
        print(f"   ❌ JSON serialization failed: {e}")

    # Feature 4: Workflow summary
    print("\n🔍 Feature 4: Workflow Summary")
    try:
        summary = workflow.export_summary()
        print(
            f"   📊 Workflow info: {summary['workflow_info']['name']}"
        )
        print(f"   📈 Structure: {summary['structure']}")
        print(f"   ⚙️ Configuration: {summary['configuration']}")
    except Exception as e:
        print(f"   ❌ Summary generation failed: {e}")

    # Feature 5: Performance monitoring
    print("\n🔍 Feature 5: Performance Monitoring")
    import time

    task = "Perform a simple test task for feature demonstration."

    start_time = time.time()
    results = workflow.run(task=task)
    execution_time = time.time() - start_time

    print(f"   ⏱️ Execution time: {execution_time:.3f} seconds")
    print(
        f"   🚀 Throughput: {len(results)/execution_time:.1f} agents/second"
    )
    print(f"   📊 Results: {len(results)} agents completed")

    return workflow


def main():
    """Main quick start guide function."""

    print("🌟 GRAPHWORKFLOW QUICK START GUIDE")
    print("=" * 60)
    print("Learn GraphWorkflow in 5 easy steps!")
    print("=" * 60)

    try:
        # Step 1: Basic setup
        workflow = step_1_basic_setup()

        # Step 2: Run workflow
        step_2_run_workflow(workflow)

        # Step 3: Parallel processing
        step_3_parallel_processing()

        # Step 4: Advanced patterns
        step_4_advanced_patterns()

        # Step 5: Additional features
        step_5_workflow_features()

        # Conclusion
        print("\n🎉 QUICK START GUIDE COMPLETED!")
        print("=" * 50)
        print("You've learned how to:")
        print("✅ Create basic workflows with agents")
        print("✅ Execute workflows with tasks")
        print("✅ Set up parallel processing")
        print("✅ Use advanced workflow patterns")
        print("✅ Monitor and optimize performance")

        print("\n🚀 Next Steps:")
        print(
            "1. Try the comprehensive demo: python comprehensive_demo.py"
        )
        print("2. Read the full technical guide")
        print("3. Implement workflows for your specific use case")
        print("4. Explore healthcare and finance examples")
        print("5. Deploy to production with monitoring")

    except Exception as e:
        print(f"\n❌ Quick start guide failed: {e}")
        print("Please check your installation and try again.")


if __name__ == "__main__":
    main()
