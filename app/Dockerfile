FROM python:3.8
LABEL maintainer="Liam Barstad"

WORKDIR /opt

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["python", "app.py"]