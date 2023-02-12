FROM ghcr.io/mlflow/mlflow:v2.1.1

EXPOSE 5000

CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri mysql+pymysql://${USERNAME}:${PASSWORD}@${HOST}:${PORT}/${DATABASE} \
    --default-artifact-root ${BUCKET} 
