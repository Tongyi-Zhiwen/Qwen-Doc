from setuptools import setup, find_packages

# Define package dependencies with specific versions for reproducibility
requirements = [
    "torch==2.6.0",
    "torchvision==0.21.0", 
    "transformers==4.51.3",
    "PyMuPDF==1.26.5",
    "numpy==2.0.2",
    "accelerate==1.12.0",
    "opencv-python-headless==4.12.0.88",
    "numba==0.60.0",
    "Pillow>=8.0.0",
    "tqdm",
    # "youtu-parsing-utils @ git+https://github.com/TencentCloudADP/youtu-parsing.git#subdirectory=youtu_parsing_utils",
]


setup(
    # Basic package information
    name="youtu-hf-parser",
    version="0.1.0",
    author="Youtu Team",
    url="https://github.com/TencentCloudADP/youtu-parsing",
    
    # Package structure definition
    # Explicitly list all packages to be included in the distribution
    packages=[
        'youtu_hf_parser',
        'youtu_hf_parser.preprocessing',
        'youtu_hf_parser.preprocessing.angle_predictor',
        'youtu_hf_parser.preprocessing.angle_predictor.model_structure'
    ],
    
    # Map package names to their actual directory locations
    package_dir={
        'youtu_hf_parser': '.',
        'youtu_hf_parser.preprocessing': 'preprocessing',
        'youtu_hf_parser.preprocessing.angle_predictor': 'preprocessing/angle_predictor',
        'youtu_hf_parser.preprocessing.angle_predictor.model_structure': 
            'preprocessing/angle_predictor/model_structure',
    },
    
    # PyPI classifiers for package categorization and metadata
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
    
    # Install the defined requirements
    install_requires=requirements,
    
    # Include additional package data files
    include_package_data=True,
    
    # Specify which files to include for each package
    package_data={
        'youtu_hf_parser': ['*.py', '*.txt', '*.md'],
        'youtu_hf_parser.preprocessing': ['*.py', 'model_weight/*.pth'],
        'youtu_hf_parser.preprocessing.angle_predictor': ['*.py'],
        'youtu_hf_parser.preprocessing.angle_predictor.model_structure': ['*.py'],
    },
)
