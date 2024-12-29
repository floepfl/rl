from setuptools import setup, find_packages

setup(
    name="YahtzeeRL",  # Replace with your project name
    version="0.1.0",           # Initial version
    author="Florian",
    author_email="florianravasi@gmail.com",
    description="Yahtzee RL",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your_project",  # Replace with your repo URL
    license="MIT",  # Replace with your license
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # Specify the src directory
    # install_requires=[
    #     "numpy",
    #     "gymnasium",
    #     "ray[rllib]",
    # ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="=3.9",
)