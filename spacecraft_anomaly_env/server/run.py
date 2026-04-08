import uvicorn

def main():
    uvicorn.run(
        "spacecraft_anomaly_env.server.app:app",
        host="0.0.0.0",
        port=7860
