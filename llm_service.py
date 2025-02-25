import re
import os
import ollama
import torch
from langchain_groq import ChatGroq
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["GROQ_API_KEY"] = "your_api_key"

class LLMService:
    def __init__(self, model_name, chat_history, question, context):
        self.model_name = model_name
        self.history_prompt = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history])
        self.prompt = (
            "You are a Retrieval-Augmented Generation (RAG)-based AI legal assistant specialized in answering "
            "questions about the laws of Pakistan. You should strictly prioritize the provided context for "
            "your responses. If the context does not contain enough information to answer, or if your knowledge "
            "base does not have the required details, respond by stating that the information is not available in your "
            "knowledge base or that you are not aware of this information. Use these phrases instead of referring "
            "to the context.\n\n"
            "Do not answer questions unrelated to Pakistani laws or provide information outside the context "
            "and domain of Pakistani legal matters.\n\n"
            f"Chat History:\n{self.history_prompt}\n\n"
            f"User's Question: {question}\n\n"
            f"Provided Context: {context}\n\n"
            "Your Response:"
        )

    def groq_execution(self):

        llm = ChatGroq(model=self.model_name)
        response = llm.invoke(self.prompt)

        return response.content

    def ollama_execution(self):

        response = ollama.chat(
        model=self.model_name,
        messages=[{"role": "user", "content": self.prompt}],
        )

        response_content = response["message"]["content"]

        # Remove content between <think> and </think> tags to remove thinking output
        final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()

        return final_answer
    
    def hugging_face_execution(self):

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model.to("cuda")

        inputs = tokenizer(self.prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return summary


        
