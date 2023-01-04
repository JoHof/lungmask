import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="lungmask",
    version="0.2.12",
    author="Johannes Hofmanninger",
    author_email="johannes.hofmanninger@meduniwien.ac.at",
    description="Package for automated lung segmentation in CT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JoHof/lungmask",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'lungmask = lungmask.__main__:main'
        ]
    },
    install_requires=[
        'pydicom',
        'numpy',
        'torch',
        'scipy',
        'SimpleITK',
        'tqdm',
        'scikit-image',
        'fill_voids'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
)
