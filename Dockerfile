FROM python:3.8

WORKDIR /app

COPY requirements.txt ./
COPY coachneuro-dev-firebase-adminsdk.json ./

RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install -y ffmpeg

COPY ./app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "$PORT"]