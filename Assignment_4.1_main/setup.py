import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="housing_packaged",
    version="0.3",
    author="Purva Patidar",
    author_email="purva.patidar@tigeranalytics.com",
    description="Package made for assignment 4.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PurvaPatidar-Tiger/mle-training",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "housing_packaged"},
    packages=setuptools.find_packages(where="housing_packaged"),
    python_requires=">=3.6",
)
