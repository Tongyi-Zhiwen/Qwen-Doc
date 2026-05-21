from setuptools import setup, find_packages
import os

def read_requirements():
    # Read and parse requirements from requirements.txt file.
    requirements = []
    # Construct path to requirements.txt relative to this setup.py file
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    
    # Read requirements file if it exists
    if os.path.exists(req_file):
        with open(req_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    # Skip URL-based dependencies (handled separately)
                    if not line.startswith('http'):
                        requirements.append(line)
    
    return requirements

setup(
    # Basic package information
    name="youtu-parsing-utils",
    version="0.1.0",
    author="Youtu Team",
    url="https://github.com/TencentCloudADP/youtu-parsing",
    
    # Package structure - single utility package
    packages=['youtu_parsing_utils'],
    
    # Map package name to its directory location (current directory)
    package_dir={'youtu_parsing_utils': '.'},
    
    # PyPI classifiers for package categorization and discovery
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    
    # Minimum Python version requirement
    python_requires=">=3.10",
    
    # Install dependencies from requirements.txt
    install_requires=read_requirements(),
    
    # Include additional non-Python files in the package
    include_package_data=True,
    
    # Specify which file types to include in the package distribution
    package_data={
        'youtu_parsing_utils': ['*.py', '*.txt', '*.md'],
    },
)
