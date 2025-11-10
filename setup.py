"""Setup script for ProtoScore package."""

from pathlib import Path
from setuptools import setup, find_packages

# Read README

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements from docker folder

def read_requirements(filename):
    """Read requirements from docker/ folder."""
    req_file = Path(__file__).parent / "docker" / filename
    if req_file.exists():
        with open(req_file, encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip() 
                and not line.startswith("#") 
                and not line.startswith("-r")
            ]
    return []

setup(
    name="ProtoScore",
    version="0.1.0",
    author="Helena Monke, Benjamin Sae-Chew",
    author_email="helena.monke@ipa.fraunhofer.de",
    description="Train and evaluate prototype-based explainable AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    install_requires=read_requirements("requirements.txt"),
    
    python_requires=">=3.12",
    classifiers=[
        "Programming Language :: Python :: 3.12",
    ],
)