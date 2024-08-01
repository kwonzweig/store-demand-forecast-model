FROM python:3.6.4

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install python dependencies with no cache
# to keep the image small and allow re-installing the dependencies with updated requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local directory app, data, model, test
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
