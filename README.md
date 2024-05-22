# Software-Engineering

Streamlit Application using Python

## Usage

## Table of Contents
<!-- vim-markdown-toc Marked -->

* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Basic Usage](#basic-usage)
* [Reporting Bugs](#reporting-bugs)
* [License](#license)
* [Authors](#authors)

<!-- vim-markdown-toc -->

## Features

- User-friendly interface for data analysis and visualization.
- 2D Visualization: 
- Classification Algorithms:
- Clustering Algorithms:
- Information tab about the contributions of the team.

## Requirements

- Python
- Docker
- Streamlit
- Pandas
- Numpy
- Matplotlib
- Pillow
- Scikit-learn
and others. You can find the complete list in the requirements.txt file.

## Installation

### Option 1: Pull the Docker Image

1. Pull the Docker image from Docker Hub:

    `docker pull spyridonkokotos/brigade-01-sw:latest`

2. Run the Docker container:

    `docker run -d --name brigade-01-sw -p 8501:8501 brigade-01-sw:latest`

3. Open your web browser and go to `http://localhost:8501` to view the application.

### Option 2: Build the Dockerfile Locally

1. Clone the repository:

    `git clone https://github.com/Brigade-01/Software-Engineering.git`

2. Navigate to the project directory:
 
    `cd Software-Engineering/src/`

3. Build the Docker image:
  
    docker build -t brigade-01-sw:latest .
  
4. Run the Docker container:

    `docker run -d --name brigade-01-sw -p 8501:8501 brigade-01-sw:latest`

5. Open your web browser and go to `http://localhost:8501` to view the application.


## Basic Usage

## Reporting Bugs

When a bug is found, please do report it by [opening an issue at github](https://github.com/Brigade-01/Software-Engineering/issues), as already stated above.

## License

MIT License

Copyright (c) 2024 Brigade-01

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Authors

The above program is a creation of:

<center>

| Name                  | Email                        |
|:---------------------:|:----------------------------:|
| Florian Dima          | inf2021044@ionio.gr          |
| Nikolaos Balatos      | inf2021151@ionio.gr          |
| Spyridon Eftychios Kokotos | skokotos@ionio.gr / inf2021098@ionio.gr |

</center>


and you can use it for your personal projects or further develop it as long as you always give credit to the creators.
