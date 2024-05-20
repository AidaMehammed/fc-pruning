import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name="fc_pruning",
                 version="0.0.17",
                 author="Aida Mehammed",
                 author_email="aida.mehammed@studium.uni-hamburg.de",
                 description="FC Pruning",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://github.com/AidaMehammed/fc-pruning",
                 project_urls={
                     "Bug Tracker": "https://github.com/AidaMehammed/fc-pruning/issues",
                 },
                 packages=setuptools.find_packages(
                     include=['fc_pruning', 'fc_pruning.Compress', 'fc_pruning.Compress.*']),
                classifiers=[  # Classifiers to categorize the project
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 python_requires=">=3.8",
                 install_requires=['featurecloud', 'torch_pruning']
                 )
