FROM python:3.12-slim
WORKDIR /work
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY experiments ./experiments
COPY tests ./tests
RUN mkdir -p results
CMD ["python", "experiments/run_all.py"]
