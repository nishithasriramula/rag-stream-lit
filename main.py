import os
import sys
from streamlit.web import cli as stcli


def main() -> int:
    port = os.environ.get("PORT", "8501")
    sys.argv = [
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        port,
        "--server.address",
        "0.0.0.0",
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
