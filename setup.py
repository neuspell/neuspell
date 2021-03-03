from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="neuspell",
    version="0.8.7",
    author="Sai Muralidhar Jayanthi, Danish Pruthi, and Graham Neubig",
    author_email="jsaimurali001@gmail.com",
    description="NeuSpell: A Neural Spelling Correction Toolkit",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/neuspell/neuspell",
    packages=find_packages(),
    classifiers=[
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">3.5",
    install_requires=[
        'numpy',
        'jsonlines',
        'sentencepiece',
        'spacy',
        'allennlp',
        'torch==1.4.0',
        'transformers==2.6.0'
    ],
    keywords="transformer networks neuspell neural spelling correction embedding PyTorch NLP deep learning"
)
