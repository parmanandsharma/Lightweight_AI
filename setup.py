from setuptools import setup, find_packages

setup(
    name='Lightweight_AI',
    version='0.1.0',
    author='Parmanand Sharma',
    author_email='sharmap.imr@gmail.com',
    description='Lightweight deep learning model for biological image analysis.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/parmanandsharma/Lightweight_AI',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',  # For OpenCV functionality
        'pandas',
        # 'glob' and 'tkinter' are part of the standard library and not required here
        'tensorflow-gpu>=2.5',  # Ensure this matches the version you used
        # Add other dependencies as needed
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    project_urls={
        'Bug Reports': 'https://github.com/parmanandsharma/Lightweight_AI/issues',
        'Source': 'https://github.com/parmanandsharma/Lightweight_AI',
    },
)

