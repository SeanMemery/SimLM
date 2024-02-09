import requests, json, re

from openai import OpenAI
from vertexai.language_models import ChatModel


PARAMS = {
    "SPEED": "V0",
    "THETA": "theta",
    "PHI": "phi",
    "X": "a",
    "Z": "b"
}

class LLM():
    def __init__(self, model_name, config) -> None:
        self.model_name = model_name

        self.CTX_N = config["ctx_n"]
        self.MODEL_OPTIONS = {
            "temperature": config["temperature"],
            "top_k": config["top_k"],
            "top_p": config["top_p"],
            "num_predict": config["num_predict"],
        }
        self.system_prompt = ""
        self.context = []
        self.session = requests.Session()
        self.url = ""
        self.openai = None

        if "gpt" in model_name:
            self.setup_gpt()
        elif "bison" in model_name:
            self.setup_bison()
        else:
            self.setup_ollama()

    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        self.context = [system_prompt]

    def get_context(self):
        return "\n".join(self.context)

    def reset(self):
        self.context = []
        self.session = requests.Session()

    def setup_ollama(self):
        url = "http://localhost:11434/api/tags"
        response = requests.get(url)
        models = json.loads(response.text)["models"]
        models = [x["name"] for x in models]
        if self.model_name not in models:
            print(f"Model {self.model_name} not found in Ollama pool, attempting download...")
            url = "http://localhost:11434/api/pull"
            myobj = {
                "name": self.model_name,
            }
            with self.session.post(url, json = myobj, stream=True) as resp:
                if resp.status_code != 200:
                    raise Exception(f"Request failed with status {resp.status_code} and message {resp.text}")
                for line in resp.iter_lines():
                    j = json.loads(line)
                    if "completed" in j.keys():
                        ratio = j["completed"] / j["total"]
                        print(f"{j['status']} - {ratio:.2f}")
                # TODO: Correctly parse response and check for success
                # print(response.text)
                # response = json.loads(response.text)
                # if response["status"] == "success":
                #     print("Download successful! Continuing...")
                # else: 
                #     print(f"Download failed with status {response["status"]}, exiting...")
                #     exit()
        else:
            print(f"Model {self.model_name} found in Ollama pool, continuing...")

        self.url = "http://localhost:11434/api/generate"

    def setup_gpt(self):
        self.url = " https://api.openai.com/v1/chat/completions"

        import dotenv, os
        dotenv.load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    
        assert api_key is not None, "OpenAI API key not found in .env file"

        self.openai = OpenAI(api_key=api_key)

    def setup_bison(self):
        import dotenv
        dotenv.load_dotenv()

    def embedding(self, prompt):
        url = "http://localhost:11434/api/embeddings"
        message = {
            "model": self.model_name,
            "prompt": prompt,
            "options": self.MODEL_OPTIONS,
        }
        response = requests.post(url, json = message)
        return json.loads(response.text)["embedding"]

    def generate(self, prompt):

        #TODO: make a better fix for context limit
        full_prompt = "\n".join(self.context[1:] + [prompt])
        system_prompt = self.system_prompt if len((self.system_prompt + full_prompt).split(" "))*1.55 < self.CTX_N else ""

        message = {
            "model": self.model_name,
            "json": True,
            "system": system_prompt,
            "prompt": full_prompt,
            "stream": True,
            "options": self.MODEL_OPTIONS,
            "num_ctx": self.CTX_N,
        }

        ### TODO: think about this
        if not "### Prediction" in prompt:
            self.context.append(prompt)

        response = self.handle_response(message)
        
        self.context.append(response)

        return response
    
    def parse_prediction(self, prediction):
        pattern = r"^([A-Z]+)\s*=\s*(-?\d+(?:\.\d{1,2})?)(?=\s|$)"
        p_clean = prediction.strip()
        if "\n" in p_clean:
            p_clean = p_clean.split("\n")
        elif "," in p_clean:
            p_clean = p_clean.split(",")
        p_clean = [x.strip() for x in p_clean if x != ""]
        p_clean = "\n".join(p_clean)
        matches = re.findall(pattern, p_clean, re.MULTILINE)
        return {PARAMS[key]: float(value) for key, value in matches if key in PARAMS.keys()}

    def handle_response(self, message):
        if "gpt" in self.model_name:
            return self.handle_gpt(message)
        elif "bison" in self.model_name:
            return self.handle_bison(message)
        else:
            response = ""
            with self.session.post(self.url, json = message, stream=True) as resp:
                if resp.status_code != 200:
                    print(f"Request failed with status {resp.status_code} and message {resp.text}")
                    return ""
                for line in resp.iter_lines():
                    j = json.loads(line)
                    response += j["response"]
            return response
        
    def handle_gpt(self, message):
        messages = [
            {
                "role": "system",
                "content": message["system"],
            },
            {
                "role": "user",
                "content": message["prompt"],
            },
        ]
        data = {
            "model": self.model_name, 
            "max_tokens": self.MODEL_OPTIONS["num_predict"], 
            "temperature":self.MODEL_OPTIONS["temperature"],
            "top_p": self.MODEL_OPTIONS["top_p"],
        }

        try:
            response = self.openai.chat.completions.create(**data, messages=messages)
            response = response.choices[0].message.content
        except Exception as e:
            print(e)
            response = ""

        return response

    def handle_bison(self, message):
        chat_model = ChatModel.from_pretrained(self.model_name)

        parameters = {
            "temperature": self.MODEL_OPTIONS["temperature"],  
            "max_output_tokens": self.MODEL_OPTIONS["num_predict"],
            "top_p": self.MODEL_OPTIONS["top_p"],
            "top_k": self.MODEL_OPTIONS["top_k"],
        }

        try:
            chat = chat_model.start_chat(
                context=message["system"],
            )

            response = chat.send_message(
                message["prompt"], **parameters
            )
        except Exception as e:
            print(e)
            response = ""

        return response.text
