# llm_agent.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re

class LLMAgent:
    def __init__(self):
        model_name = "microsoft/phi-3-mini-4k-instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def _generate(self, messages, max_tokens=300):
        # Apply chat template 
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Only decode new tokens (not the full prompt)
        generated = outputs[0][inputs["input_ids"].shape[-1]:]

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def decide_query(self, history):
        messages = [
            {"role": "system", "content": "You monitor illegal fishing in Brazil's EEZ."},
            {"role": "user", "content": f"""
                Return STRICT JSON only:
                {{
                "start_date": "YYYY-MM-DD",
                "end_date": "YYYY-MM-DD",
                "focus": "string"
                }}

                History:
                {history}
                """}
        ]

        output = self._generate(messages)
        return self.safe_json_parse(output)

    def explain_results(self, anomalies):
        if anomalies.empty:
            return "No anomalies detected."

        
        cols = [c for c in ["ssvid", "lat", "lon", "speed", "course", "anomaly"] if c in anomalies.columns]

        sample = anomalies[cols].head(15).to_dict(orient="records")

        messages = [
            {"role": "system", "content": "You are a maritime intelligence analyst."},
            {"role": "user", "content": f"""
                Explain why these vessels are suspicious.

                Use:
                - clustering
                - anomalies
                - repeated presence

                Be concise.

                Data:
                {sample}
                """}
                    ]

        return self._generate(messages)

    def safe_json_parse(self, text):
        try:
            return json.loads(text)
        except:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
        return None