from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from typing import Dict, List, Union


class BertIntegration:
    def __init__(self, classnames: List[str] = None):
        self.model = AutoModelForMaskedLM.from_pretrained("models/bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("models/bert-base-uncased")
        self.model.eval()

        self.classnames = [name.lower() for name in
                           (classnames or ['Angry', 'Bored', 'Calm', 'Fear', 'Happy', 'Love', 'Sad'])]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.classnames)}

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

        self.max_length = 512

    def get_bert_predictions(self, prompt: Union[str, List[str]]) -> Dict[str, float]:
        if isinstance(prompt, str):
            prompt = [prompt]

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        print(prompt)
        mask_positions = torch.where(inputs['input_ids'] == self.mask_token_id)[1]

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits[:, mask_positions, :]
            probs = torch.softmax(predictions, dim=-1)

        batch_emotion_probs = []
        for batch_idx in range(len(prompt)):
            emotion_probs = {}
            for emotion in self.classnames:
                token_id = self.tokenizer.convert_tokens_to_ids(emotion)
                if batch_idx < probs.size(0):
                    emotion_probs[emotion] = probs[batch_idx, 0, token_id].item()
            batch_emotion_probs.append(emotion_probs)

        return batch_emotion_probs[0] if len(prompt) == 1 else batch_emotion_probs

    def extract_sentiment(self, emotion_probs: Dict[str, float]) -> int:
        try:
            max_emotion = max(emotion_probs.items(), key=lambda x: x[1])[0]
            return self.label_to_idx[max_emotion]
        except Exception as e:
            print(f"Error in extract_sentiment: {str(e)}, probs: {emotion_probs}")
            return self.label_to_idx.get('calm', 2)

    def generate_prompt(self, fusion_output: Dict, reference_samples: List = None, similarities: List = None) -> str:
        text = fusion_output['text']['content'] if isinstance(fusion_output['text'], dict) else fusion_output['text']

        options = ", ".join(self.classnames)
        base_prompt = f"Choose from [{options}]. The text '{text}' expresses a [MASK] emotion."

        if reference_samples is not None and similarities is not None and len(reference_samples) > 0:
            content = base_prompt + " Similar cases show: "
            all_samples = list(zip(reference_samples, similarities))
            all_samples.sort(key=lambda x: x[1], reverse=True)

            for sample, sim in all_samples[:3]:
                if isinstance(sample, dict) and 'label' in sample:
                    emotion = self.classnames[sample['label']]
                    content += f"{emotion}, "

            content = content.rstrip(", ") + f" therefore this text is [MASK]."
            return content

        return base_prompt

    def process_fusion_output(self, fusion_output: Dict, reference_samples: List = None,
                              similarities: List = None) -> Dict:
        prompt = self.generate_prompt(fusion_output, reference_samples, similarities)
        emotion_probs = self.get_bert_predictions(prompt)
        label = self.extract_sentiment(emotion_probs)

        return {
            "sentiment": label,
            "probabilities": emotion_probs
        }

    def get_predictions(self, texts: List[str]) -> List[int]:
        predictions = []
        for text in texts:
            options = ", ".join(self.classnames)
            prompt = f"Choose from [{options}]. The text '{text}' expresses a [MASK] emotion."

            emotion_probs = self.get_bert_predictions(prompt)
            label = self.extract_sentiment(emotion_probs)
            predictions.append(label)

        return predictions


def test_bert_integration():
    classnames = ['Angry', 'Bored', 'Calm', 'Fear', 'Happy', 'Love', 'Sad']
    bert_integration = BertIntegration(classnames=classnames)

    test_text = "I'm feeling really happy today because the sun is shining!"
    fusion_output = {
        'text': {'content': test_text},
        'prediction': None,
        'confidence': None,
        'explanation': None
    }

    result = bert_integration.process_fusion_output(fusion_output)
    print(f"Single prediction test:")
    print(f"Text: {test_text}")
    print(f"Predicted sentiment: {classnames[result['sentiment']]}")
    print(f"Probabilities: {result['probabilities']}\n")

    # 测试批量预测
    test_texts = [
        "I'm so angry right now!",
        "Just feeling calm and peaceful",
        "This makes me really sad..."
    ]
    predictions = bert_integration.get_predictions(test_texts)
    print("Batch prediction test:")
    for text, pred in zip(test_texts, predictions):
        print(f"Text: {text}")
        print(f"Prediction: {classnames[pred]}\n")


if __name__ == "__main__":
    test_bert_integration()