import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name="fc-pruning",
                 version="0.0.1",
                 author="Aida Mehammed",
                 author_email="aida.mehammed@studium.uni-hamburg.de",
                 description="FC-Pruning",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://github.com/FeatureCloud/fc-pruning",
                 project_urls={
                     "Bug Tracker": "https://github.com/AidaMehammed/fc-pruning",
                 },
                 packages=setuptools.find_packages(include=['Compress','Compress.*']),

                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "Operating System :: OS Independent",
                 ],
                 python_requires=">=3.7",
                 
                 install_requires=['featurecloud','torch_pruning']

                 )
