import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lungmask-jhofmanninger", # Replace with your own username
    version="0.1",
    author="Johannes Hofmanninger",
    author_email="j.hofmanninger@gmail.com",
    description="Package for automated lung mask generation in CT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JoHof/lungmask",
    packages=setuptools.find_packages(),
    install_requires=[
        'pydicom',
        'numpy',
        'scikit_image',
        'torch',
        'torchvision',
        'scipy',
        'SimpleITK',
        'scikit-image',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)