import subprocess
import sys

from hatchling.builders.hooks.plugin.interface import BuildHookInterface  # type: ignore


class CustomBuildHook(BuildHookInterface):
    """Custom build hook to generate schemas before building."""

    def initialize(self, version, build_data):
        """Initialize hook - run schema generation before build."""
        print("Running pre-build schema generation...")

        # Run the schema generation script
        try:
            result = subprocess.run(
                [sys.executable, "scripts/generate_schemas.py"],
                check=True,
                capture_output=True,
                text=True,
            )

            print("Schema generation completed successfully:")
            if result.stdout.strip():
                # Print the schema generation output
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        print(f"   {line}")

        except subprocess.CalledProcessError as e:
            print(f"Schema generation failed: {e}")
            print("⚠️  Continuing build with existing schemas")
        except Exception as e:
            print(f"Unexpected error during schema generation: {e}")
            print("⚠️  Continuing build with existing schemas")
