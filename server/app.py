from spacecraft_anomaly_env.server.app import create_app
import uvicorn

# create app instance
app = create_app()

def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860
    )

if __name__ == "__main__":
    main()
