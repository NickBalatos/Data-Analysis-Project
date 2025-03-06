# Software-Engineering: 💯

Streamlit Application using Python:

This project focuses on building a web-based application for data mining and analysis, leveraging either Streamlit or RShiny. Key features include tabular data loading, structured representation, 2D visualization, machine learning algorithm comparison, comprehensive result analysis, and project information. Also, this project aims to provide a user-friendly interface for in-depth data exploration and algorithm evaluation, catering to a wide audience.

Key words: Data Mining, Data Analysis, 2D Visualization, Machine Learning

## Usage: 📈

![GUI](https://github.com/Greekforce1821/Software-Engineering/assets/33377581/7925b3c6-13f4-4032-8748-120d9277eb6c)


<p align="center">Welcome Screen of the Application 🌐</p>


## Table of Contents: 📖
<!-- vim-markdown-toc Marked -->

* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Basic Usage](#basic-usage)
* [Reporting Bugs](#reporting-bugs)
* [License](#license)
* [Authors](#authors)

<!-- vim-markdown-toc -->

## Features: ✨

- User-friendly interface for data analysis and visualization.
- 2D Visualization: You can visualize the data using a variety of algorithms such as PCA and t-SNE.
- Classification Algorithms: You can perform classification with the Random Forest and SVC algorithms, as well as compare the accuracy between the two algorithms.
- Clustering Algorithms: You can perform clustering using the K-Means and Hierarchical Clustering algorithms.
- Information tab about the contributions of the team.

## Requirements: ⚙️

- Python
- Docker
- Streamlit
- Pandas
- Numpy
- Matplotlib
- Pillow
- Scikit-learn
and others. You can find the complete list in the requirements.txt file.

## Installation: 👩🏻‍💻

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
  
    `docker build -t brigade-01-sw:latest .`
  
4. Run the Docker container:

    `docker run -d --name brigade-01-sw -p 8501:8501 brigade-01-sw:latest`

5. Open your web browser and go to `http://localhost:8501` to view the application.

## Basic Usage: 📈

1. Upload your file: Select the .csv or .xls file you want to use by clicking the <i>"Browse Files"</i> button.

2. Wait for confirmation: Wait until a message confirming the <b>successful</b> upload of your file appears.

3. Start exploring: Once the file is loaded and there are no issues with it's content, you can start exploring the application.

4. Navigate through the app: Use the menu on the left to switch between tabs and utilize the included algorithms to test their performance.

By following these steps, you can efficiently use the application for data analysis and algorithm comparison.

## Reporting Bugs 🐞

When a bug is found, please do report it by [opening an issue at github](https://github.com/Brigade-01/Software-Engineering/issues), as already stated above.

## License ✒️

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

## Authors 👨🏻‍⚖️

The above program is a creation of:

<center>

| Name                  | Registry Number              | Semester              | Email                        |
|:---------------------:|:----------------------------:|:---------------------:|:----------------------------:|
| Florian Dima          | inf2021044                   | 6th                   | inf2021044@ionio.gr          |
| Nikolaos Balatos      | inf2021151                   | 6th                   | nbalatos@gmail.com          |
| Spyridon Eftychios Kokotos | inf2021098              | 6th                   | skokotos@ionio.gr / inf2021098@ionio.gr |

</center>


and you can use it for your personal projects or further develop it as long as you always give credit to the creators.
