from setuptools import setup, find_packages

setup(
    name="flask_tft_app",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "pytorch_forecasting",
        "torch",
        "pandas",
        "numpy",
    ],
    python_requires=">=3.8",
)
