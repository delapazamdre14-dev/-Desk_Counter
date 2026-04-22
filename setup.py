from setuptools import setup, find_packages

setup(
    name="desk-counter",
    version="0.1.0",
    description="Object counting and segmentation library",
    author="Equipo 2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "matplotlib"
    ],
    python_requires=">=3.8",
)