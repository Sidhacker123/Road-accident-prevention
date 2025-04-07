
import openai

openai.api_key = "your-openai-api-key"

# Replace with your fine-tuned model name
model_name = "davinci:ft-yourname-2025-accident-risk"

def predict_accident_severity(weather, light, road_surface, junction, cause):
    prompt = (
        f"Weather: {weather}. "
        f"Light: {light}. "
        f"Road surface: {road_surface}. "
        f"Junction: {junction}. "
        f"Cause: {cause}. "
        f"What is the accident severity?"
    )

    response = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0
    )
    return response.choices[0].text.strip()

# Example usage
print(predict_accident_severity("Rainy", "Night", "Asphalt roads", "Y Shape", "Overtaking"))
