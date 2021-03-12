FROM python:3.8.6-buster
COPY SmArtTorch /SmArtTorch
RUN pip install --upgrade pip
RUN pip install -r SmArtTorch/requirements.txt
CMD streamlit run SmArtTorch/app_sidebar.py --server.port $PORT