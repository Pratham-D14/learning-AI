from dotenv import load_dotenv
from google import genai
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


load_dotenv()

client = genai.Client(
    api_key="AIzaSyAeuwv3_VJ2YhRfwJNvneacn9UMyICGI6c",
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)
