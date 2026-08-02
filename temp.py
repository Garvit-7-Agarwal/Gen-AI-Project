from google import genai

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.6-flash",
    input="Write a three.js script that renders an interactive 3D robot."
)

print(interaction.output_text)