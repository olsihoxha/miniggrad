import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="miniggrad",
    version="0.1.0",
    author="Olsi Hoxha",
    author_email="olsihoxha824@gmail.com",
    description="A compact autograd engine that operates with scalar values,"
                " featuring a minimalistic PyTorch-like neural network library built on top.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/olsihoxha/miniggrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'graphviz>=0.20.1',
    ],
)
