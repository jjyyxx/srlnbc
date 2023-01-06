from setuptools import setup, find_packages

setup(
    name='srlnbc',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=False,
    package_data={
        "srlnbc": ["env/xmls/*.xml"],
    },
)

