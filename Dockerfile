FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY app.py ./app.py
COPY model ./model
COPY model/best_random_forest_model.pkl ./model/best_random_forest_model.pkl

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501"]
CMD ["app.py"]
