# ruff: noqa: PLR0912, PLR0915, PLR2004

"""Simple demonstration of robust copy functions.

This script shows practical examples of:
- Copying files and directories to containers
- Extracting results from containers
- Consistent behavior across backends
- Error handling and robustness features

Usage:
    python examples/copy_demo.py [backend]

    backend: docker, podman, kubernetes, tenki (default: docker)
"""

import logging
import tempfile
from pathlib import Path

from llm_sandbox import SandboxBackend, SandboxSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


def create_demo_content(base_dir: Path, root: str = "/sandbox") -> dict[str, Path]:
    """Create demonstration files and directories."""
    content = {}

    # Create a data processing script
    content["processor"] = base_dir / "data_processor.py"
    content["processor"].write_text(f'''
import json
import csv
from pathlib import Path

def process_data():
    """Process input data and generate outputs."""
    print("🔄 Processing data...")

    # Read input data
    with open("{root}/input/data.json", "r") as f:
        data = json.load(f)

    # Process and create output
    output_dir = Path("{root}/output")
    output_dir.mkdir(exist_ok=True)

    # Create summary report
    summary = {{
        "total_items": len(data["items"]),
        "processed_timestamp": "2024-01-01",
        "status": "completed"
    }}

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Create CSV export
    with open(output_dir / "export.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "value"])
        for item in data["items"]:
            writer.writerow([item["id"], item["name"], item["value"]])

    print("✅ Processing complete! Check {root}/output/")

if __name__ == "__main__":
    process_data()
''')

    # Create input data directory
    input_dir = base_dir / "input_data"
    input_dir.mkdir(exist_ok=True)

    # Create sample JSON data
    sample_data = {
        "items": [
            {"id": 1, "name": "Widget A", "value": 100},
            {"id": 2, "name": "Widget B", "value": 250},
            {"id": 3, "name": "Widget C", "value": 175},
        ]
    }

    with (input_dir / "data.json").open("w") as f:
        import json

        json.dump(sample_data, f, indent=2)

    content["input_dir"] = input_dir

    # Create a configuration file
    content["config"] = base_dir / "config.txt"
    content["config"].write_text("# Configuration\nverbose=true\noutput_format=json,csv\n")

    return content


def run_demo(backend_name: str = "docker") -> None:
    """Run the copy functions demonstration."""
    logger.info("🚀 Copy Functions Demo - %s Backend", backend_name.upper())
    logger.info("=" * 50)

    # Map backend names to enums
    backend_map = {
        "docker": SandboxBackend.DOCKER,
        "podman": SandboxBackend.PODMAN,
        "kubernetes": SandboxBackend.KUBERNETES,
        "tenki": SandboxBackend.TENKI,
    }

    if backend_name not in backend_map:
        logger.error("❌ Unknown backend: %s", backend_name)
        logger.error("Available: %s", list(backend_map.keys()))
        return

    root = "/home/tenki" if backend_name == "tenki" else "/sandbox"

    client = None
    if backend_name == "docker":
        import docker

        # Use Docker Desktop's actual socket path
        client = docker.DockerClient.from_env()
    elif backend_name == "podman":
        from podman import PodmanClient

        client = PodmanClient.from_env()

    backend_enum = backend_map[backend_name]

    # Create temporary directories for demo content
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        logger.info("📝 Creating demo content...")
        demo_content = create_demo_content(temp_path, root)

        logger.info("📦 Creating %s session...", backend_name)

        try:
            with SandboxSession(
                backend=backend_enum,
                lang="python",
                verbose=False,
                keep_template=True,
                client=client,
                workdir=root,
            ) as session:
                logger.info("✅ %s session ready", backend_name)

                # Demo 1: Copy Python script to container
                logger.info("\n📁 Step 1: Copying Python script to container")
                session.copy_to_runtime(str(demo_content["processor"]), f"{root}/data_processor.py")
                logger.info("✅ Script copied successfully")

                # Demo 2: Copy input directory to container
                logger.info("\n📁 Step 2: Copying input data directory to container")
                session.copy_to_runtime(str(demo_content["input_dir"]), f"{root}/input")
                logger.info("✅ Input data copied successfully")

                # Demo 3: Copy single config file
                logger.info("\n📁 Step 3: Copying configuration file")
                session.copy_to_runtime(str(demo_content["config"]), f"{root}/config.txt")
                logger.info("✅ Config file copied successfully")

                # Demo 4: Verify files are in container
                logger.info("\n🔍 Step 4: Verifying files in container")
                result = session.execute_command(f"find {root} -type f | head -10")
                logger.info("📋 Files in container:")
                for line in result.stdout.strip().split("\n"):
                    if line:
                        logger.info("   %s", line)

                # Demo 5: Execute the processing script
                logger.info("\n🏃 Step 5: Running data processing script")
                result = session.execute_command(f"python {root}/data_processor.py")
                logger.info("📤 Script output:")
                logger.info(result.stdout)

                # Demo 6: Copy results back from container
                logger.info("\n📁 Step 6: Copying results back to host")
                output_dir = temp_path / "results"

                # Copy the entire output directory
                session.copy_from_runtime(f"{root}/output", str(output_dir))
                logger.info("✅ Results copied back successfully")

                # Demo 7: Show what we got back
                logger.info("\n📋 Step 7: Examining extracted results")
                if output_dir.exists():
                    for file_path in output_dir.rglob("*"):
                        if file_path.is_file():
                            rel_path = file_path.relative_to(output_dir)
                            logger.info("   📄 %s (%s bytes)", rel_path, file_path.stat().st_size)

                            # Show content of small files
                            if file_path.suffix in [".json", ".csv", ".txt"] and file_path.stat().st_size < 500:
                                logger.info("      Content preview:")
                                content = file_path.read_text()
                                for line in content.split("\n")[:5]:
                                    if line.strip():
                                        logger.info("      %s", line)
                                if len(content.split("\n")) > 5:
                                    logger.info("      ...")
                else:
                    logger.info("   ❌ No results directory found")

                # Demo 8: Error handling demonstration
                logger.info("\n🛡️  Step 8: Demonstrating error handling")
                try:
                    session.copy_to_runtime("/nonexistent/file.txt", f"{root}/dummy.txt")
                    logger.info("   ❌ Expected this to fail!")
                except FileNotFoundError:
                    logger.info("   ✅ Correctly handled error")

                logger.info("\n🎉 Demo completed successfully with %s backend!", backend_name)
                logger.info("   All copy operations worked robustly and consistently.")

        except Exception:
            logger.exception("❌ Demo failed")
            raise


def main() -> None:
    """Execute main function."""
    import sys

    backend = sys.argv[1] if len(sys.argv) > 1 else "docker"

    logger.info("🧪 LLM Sandbox Copy Functions Demo")
    logger.info("==================================")
    logger.info("This demo shows how to reliably copy files and directories")
    logger.info("between your host system and sandbox containers.\n")

    try:
        run_demo(backend)

        logger.info("\n💡 Key Features Demonstrated:")
        logger.info("   • File and directory copying (both directions)")
        logger.info("   • Robust error handling")
        logger.info("   • Consistent behavior across backends")
        logger.info("   • Real-world data processing workflow")
        logger.info("   • Safe path handling")

    except Exception:
        logger.exception("❌ Demo failed")
        logger.info("💡 Make sure the selected backend is available and running.")
        sys.exit(1)


if __name__ == "__main__":
    main()
