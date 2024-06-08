import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Chest-Disease-Classification-from-Chest-CT-Scan-Image"
AUTHOR_USER_NAME = "ajayghimire9"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "ajayghimire9@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/Ajayghimire9/Chest-Diesease-Classification-from-chest-CT-Scan-Image",
    project_urls={
        "Bug Tracker": f"https://github.com/Ajayghimire9/Chest-Diesease-Classification-from-chest-CT-Scan-Image/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)