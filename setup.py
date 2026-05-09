from setuptools import setup, find_packages

setup(
    name="haqoa",
    version="0.2.0",
    description="Hybrid AI-Assisted Quantum-Inspired Optimization Architecture",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
)
