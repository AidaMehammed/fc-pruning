import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name="fc-pruning",
                 version="0.0.1",
                 author="Aida Mehammed",
                 author_email="aida.mehammed@studium.uni-hamburg.de",
                 description="Image Classification Pruning",
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url="https://github.com/AidaMehammed/Image-Classification-Pruning",
                 project_urls={
                     "Bug Tracker": "https://github.com/FeatureCloud/app-template/issues",
                 },
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "Operating System :: OS Independent",
                 ],
                 packages=setuptools.find_packages(include=['FeatureCloud', 'FeatureCloud.*']),
                 python_requires=">=3.7",
                 entry_points={'console_scripts': ['FeatureCloud = FeatureCloud.api.cli.__main__:fc_cli',
                                                   'featurecloud = FeatureCloud.api.cli.__main__:fc_cli',
                                                   ]
                               },
                 install_requires=['featurecloud','torch_pruning','bios','torchvision']

                 )
