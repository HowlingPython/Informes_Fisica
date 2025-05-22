FROM python:3.11-slim 
RUN apt-get update && apt-get install -y ffmpeg 
COPY mov2mp4_gui.py .env requirements.txt ./
RUN pip install -r requirements.txt
CMD ["python", "mov2mp4_gui.py"]
