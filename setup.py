from setuptools import setup

setup(
    name='robustipy',
    version='0.0.1.dev5',
    description='Multiversal estimation for robust inference.',
    long_description='Multiversal estimation for robust inference.',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
    ],
    url='https://github.com/dhvalden/robustify.git',
    author='Charles Rahal, Daniel Valdenegro',
    author_email='charles.rahal@demography.ox.ac.uk, daniel.valdenegro@demography.ox.ac.uk',
    license='GPLv3',
    packages=['robustipy'],
    package_dir={'': 'src'},
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.23.2',
        'pandas>=1.4.3',
        'scipy>=1.9.0',
        'statsmodels>=0.13.2',
        'joblib>=1.2.0',
        'joypy>=0.2.6',
        'linearmodels>=4.27',
        'matplotlib>=3.6.1',
        'seaborn',
        'rich',
        'scikit-learn'
    ],
    zip_safe=False
)
