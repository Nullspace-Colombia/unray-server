import google.generativeai as genai

genai.configure(api_key="AIzaSyAu2P-K46LiNwg7BnJv0NDF80BZaLWHUt4")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain how AI works")
print(response.text)