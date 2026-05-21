from setuptools import setup, find_packages

# 从requirements.txt文件读取依赖
def parse_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().splitlines()
        
setup(
    name='dots_ocr',  
    version='1.0', 
    packages=find_packages(),  
    include_package_data=True,  
    install_requires=parse_requirements('requirements.txt'),  
    description='dots.ocr: Multilingual Document Layout Parsing in one Vision-Language Model',
    url="https://github.com/rednote-hilab/dots.ocr",
    python_requires=">=3.10",
)