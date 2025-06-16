from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import Dict, List


class LlamaIntegration:
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "models/Llama-3.1-8B-Instruct",
            quantization_config=quantization_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("models/Llama-3.1-8B-Instruct")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("")
        ]

    def label_to_text(self, label):
        # label_map = {
        #     0: "angry",
        #     1: "bored",
        #     2: "calm",
        #     3: "fear",
        #     4: "happy",
        #     5: "love",
        #     6: "sad"
        # }
        label_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive",
        }
        return label_map.get(label, "unknown")

    def generate_prompt(self, fusion_output, reference_samples=None, similarities=None):
        if reference_samples is not None and similarities is not None and len(reference_samples) > 0 and len(
                similarities) > 0:
            content = (f"Analyze the sentiment based on the following fusion model output:\n\n"
                       f"Multimodal Model Output: {fusion_output}\n\n"
                       f"Here are the most similar reference cases:")

            all_samples = list(zip(reference_samples, similarities))
            all_samples.sort(key=lambda x: x[1], reverse=True)

            for sample, sim in all_samples[:3]:
                if isinstance(sample, dict) and 'label' in sample and 'text' in sample:
                    sentiment = self.label_to_text(sample['label'])
                    content += (f"\n- [{sentiment.upper()}] "
                                f"Text: \"{sample['text'][:100]}...\" "
                                f"Image: \"{sample['image']}\" "
                                f"(Similarity: {sim:.3f})")

            content += ("\n\nBased on the fusion model output and reference examples, analyze:\n"
                        "1. The model's prediction and confidence\n"
                        "2. The explanation provided\n"
                        "3. Similar patterns with reference cases\n"
                        "4. The emotional indicators in the text\n\n"
                        "Please respond with exactly one word from: Positive or Negative.")

            return [
                {"role": "system",
                 "content": "You are a sentiment analysis assistant. Analyze the sentiment based on the fusion model output, explanation, and the text content. Consider the model's confidence and explanation in your analysis. Respond with one of following sentiments: Positive , Neutral or Negative"},
                {"role": "user",
                 "content": content}
            ]
        else:
            return [
                {"role": "system",
                 "content": "You are a sentiment analysis assistant. Analyze the sentiment based on the fusion model output, explanation, and the text content. Consider the model's confidence and explanation in your analysis. Respond with one of following sentiments: Positive Neutral or Negative"},
                {"role": "user",
                 "content": f"Analyze the sentiment of this fusion model output, explanation, and text: {fusion_output}"}
            ]

    def get_llama_response(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)

    def extract_sentiment(self, response):
        try:
            response = response.lower().strip()

            # if response.startswith('a'):
            #     return 0
            # elif response.startswith('b'):
            #     return 1
            # elif response.startswith('c'):
            #     return 2
            # elif response.startswith('f'):
            #     return 3
            # elif response.startswith('h'):
            #     return 4
            # elif response.startswith('l'):
            #     return 5
            # elif response.startswith('s'):
            #     return 6

            if response.startswith('neg'):
                return 0
            elif response.startswith('neu'):
                return 1
            elif response.startswith('pos'):
                return 2

            print(f"Warning: Could not match response '{response}' to any sentiment, defaulting to Neutral")
            return 1

        except Exception as e:
            print(f"Error in extract_sentiment: {str(e)}, response: {response}")
            return 1

    def process_fusion_output(self, fusion_output, reference_samples=None, similarities=None):
        prompt = self.generate_prompt(fusion_output, reference_samples, similarities)
        # print(prompt)
        response = self.get_llama_response(prompt)
        label = self.extract_sentiment(response)
        return {"sentiment": label}

    def get_predictions(self, texts):
        predictions = []
        for text in texts:
            fusion_output = {
                'text': text,
                'prediction': None,
                'confidence': None,
                'explanation': None
            }
            result = self.process_fusion_output(fusion_output)
            predictions.append(result['sentiment'])

        return predictions
