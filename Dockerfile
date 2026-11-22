FROM python:3.12-slim
WORKDIR /app
ARG FIELD_OBJECTS=0
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home atlas
COPY --chown=atlas:atlas . .
RUN mkdir -p /app/data && chown atlas:atlas /app/data
ENV ATLAS_DB=/app/data/atlas.sqlite3
VOLUME ["/app/data"]
USER atlas
RUN if [ "$FIELD_OBJECTS" = 1 ]; then python -m pip install --user --no-cache-dir -r requirements-field.txt && python -m vision.field --prepare; fi
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=4s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=3)"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
