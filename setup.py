from setuptools import setup

setup(
    name='nrobust',
    version='0.0.1.dev4',
    description='Multiversal estimation for robust inference.',
    long_description='Multiversal estimation for robust inference.',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
    ],
    url='https://github.com/centre-for-care/nrobust',
    author='Charles Rahal, Daniel Valdenegro',
    author_email='charles.rahal@sociology.ox.ac.uk, daniel.valdenegro@sociology.ox.ac.uk',
    license='GPLv3',
    packages=['nrobust'],
    package_dir={'': 'src'},
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.23.2',
        'pandas>=1.4.3',
        'scipy>=1.9.0',
        'statsmodels>=0.13.2',
        'tqdm>=4.64.0',
        'joblib>=1.2.0',
        'joypy>=0.2.6',
        'linearmodels>=4.27',
        'matplotlib>=3.6.1',
    ],
    zip_safe=False
)
