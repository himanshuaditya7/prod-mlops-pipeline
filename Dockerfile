FROM python:3.6

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install the dependencies
RUN pip install -r requirements.txt
# Run
CMD ["python", "scripts/x2_process_data.py" ,"&&", "python", "scripts/x3_train.py" ]

