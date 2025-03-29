from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llamaethics",
    version="0.1.0",
    author="LlamaSearch AI Team",
    author_email="team@llamasearch.ai",
    description="A toolkit for ethical AI alignment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llamasearch/LlamaEthics",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "shap>=0.40.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "Flask>=2.0.0",
        "plotly>=5.3.0", 
        "dash>=2.0.0",
        "fairlearn>=0.7.0",
        "lime>=0.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.0",
            "sphinx>=4.2.0",
            "black>=21.8b0",
            "isort>=5.9.3",
            "flake8>=3.9.2",
        ],
    },
) 