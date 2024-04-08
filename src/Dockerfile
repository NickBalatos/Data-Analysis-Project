FROM python:3.10
# set a directory for the app
WORKDIR /usr/src/app
# copy all the files to the container
COPY . .
# install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# define the port number the container should expose
EXPOSE 8501
# run the command
CMD ["streamlit", "run","./main.py"]