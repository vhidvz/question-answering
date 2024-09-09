import os
import fasttext
import requests

from tqdm import tqdm

# Model location
MODEL_PATH = (".data", "lid.176.bin")

# URL for downloading FastText language detection model
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


class LanguageIdentification():
    def __init__(self):
        self.model = None

    def load_model(self):
        path = '/'.join(MODEL_PATH)
        try:
            if not os.path.exists(path):
                if not os.path.exists(MODEL_PATH[0]):
                    os.makedirs(MODEL_PATH[0])

                print(f"Downloading model to {MODEL_PATH[0]}...")

                response = requests.get(MODEL_URL, stream=True)
                total_size_in_bytes = int(
                    response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kilobyte

                progress_bar = tqdm(total=total_size_in_bytes,
                                    unit='iB', unit_scale=True)

                with open(path, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)

                progress_bar.close()

                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("Error: Download failed or incomplete.")
                    raise Exception(f"Download failed")
                else:
                    print("Model downloaded successfully.")
            self.model = fasttext.load_model(path)
        except BaseException as e:
            if os.path.exists(path):
                os.remove(path)
            raise e
        return self

    def detect(self, text: str):
        if self.model is None:
            raise Exception(f"Model not loaded")

        prediction = self.model.predict(text)
        # Extract language code
        code = prediction[0][0].replace("__label__", "")
        # Extract confidence percentage
        confidence: float = round(prediction[1][0] * 100, 2)
        return {"language": code, "confidence": confidence}


if __name__ == "__main__":
    language_identification = LanguageIdentification()
    language_identification.load_model()
    print(language_identification.detect("Hello, world!"))
