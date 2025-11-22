import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()
TIMEOUT = 60 * 30  # 30 min

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv('PORT')),
        timeout_keep_alive=TIMEOUT
    )
