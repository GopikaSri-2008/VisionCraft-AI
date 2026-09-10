from flask import Flask, render_template, request, jsonify
import os
import base64
import io
from huggingface_hub import InferenceClient

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({
                "error": "Please enter an image description."
            }), 400

        api_key = os.environ.get("HF_TOKEN")

        if not api_key:
            return jsonify({
                "error": "Hugging Face API key is not configured."
            }), 500

        client = InferenceClient(
            api_key=api_key
        )

        image = client.text_to_image(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell"
        )

        buffer = io.BytesIO()

        image.convert("RGB").save(
            buffer,
            format="PNG"
        )

        encoded_image = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return jsonify({
            "image": "data:image/png;base64," + encoded_image
        })

    except Exception as e:
        print("ERROR:", str(e))

        return jsonify({
            "error": "Image generation failed. Please try again."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)