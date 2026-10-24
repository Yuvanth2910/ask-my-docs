import os
import weaviate
from dotenv import load_dotenv

load_dotenv()

client = weaviate.connect_to_local()
print("Weaviate ready:", client.is_ready())
client.close()
print("Anthropic key set:", bool(os.getenv("ANTHROPIC_API_KEY")))