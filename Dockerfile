FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home atlas
COPY --chown=atlas:atlas . .
RUN mkdir -p /app/data && chown atlas:atlas /app/data
ENV ATLAS_DB=/app/data/atlas.sqlite3
VOLUME ["/app/data"]
USER atlas
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=4s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=3)"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
