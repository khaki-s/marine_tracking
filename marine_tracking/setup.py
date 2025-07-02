from setuptools import setup, find_packages

setup(
    name="marine_tracking",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "ultralytics==8.3.27",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "opencv-python>=4.5.0",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "Pillow>=8.0.0",
    ],
) 