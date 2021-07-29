from setuptools import setup, find_packages


setup(
    name="jespipe",
    version="0.0.1",
    description="Build and use manipulation, machine learning, attack, and plotting plguins with the Jespipe system",
    url="https://github.com/NucciTheBoss/jespipe",
    author="Jason C. Nucciarone",
    author_email="nucci.programming@gmail.com",
    license="GNU General Public License version 3",
    packages=find_packages(),
    install_requires=[
        "joblib",
        "tqdm",
        "matplotlib",
        "tensorflow",
        "Keras",
        "numpy",
        "pandas",
        "xgboost",
        "dataclasses",
        "scikit_learn"
    ],

    keywords=['machine-learning', 'adversarial-machine-learning', 'automation', 'plugin'],
    classifiers=[
        'Development Status :: 1 - Experimental',
        'Intended Audience :: Scientific/Research',
        'License :: OSI Approved :: GNU GPLv3 License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.9.5',
        'Programming Language :: Python :: 3.9.6'
    ],
)
