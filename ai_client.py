from google import genai
from IPython.display import Markdown


client = genai.Client(api_key="AIzaSyDcTvsNocVzRQKp8-5L00b6XNRY2S1ZnGc")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in a few words",
)

print(response.text)
