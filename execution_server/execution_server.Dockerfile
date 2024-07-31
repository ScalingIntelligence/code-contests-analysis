# from this directory:
# docker build -t execution-server -f execution_server.Dockerfile .

# or from parent directory:
# docker build -t simple-server -f sampling/execution_server.Dockerfile sampling

# docker run -p 8004:8004 execution-server

# Use an official Python runtime as the parent image
FROM python:3.11-slim
RUN apt update && apt-get install -y curl tmux && apt-get clean

# Install the required dependencies
RUN pip install fastapi uvicorn typer

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY execution_server/execution_server.py /app
COPY code_contests_analysis/schema.py /app

# Make port 8004 available to the world outside this container
EXPOSE 8004

# Run the FastAPI server when the container launches

CMD ["python", "execution_server.py"]
