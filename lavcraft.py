import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# === Загрузка эмбеддингов и текста ===
embeddings = torch.load("lembeddings.pt")
paragraphs = torch.load("lparagraphs.pt")

print(f"\nЗагружено {len(paragraphs)} абзацев и эмбеддингов.")

# === Загрузка модели для поиска похожих текстов ===
search_model_name = "distilbert-base-uncased"
search_tokenizer = AutoTokenizer.from_pretrained(search_model_name)
search_model = AutoModel.from_pretrained(search_model_name).eval()

# === Загрузка модели LLaMA ===
llama_model_name = "D:/anaconda3/envs/Lama5/Lama5/Llama-3.2-3B-Instruct"  # Убедитесь, что модель скачана
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name).eval()
llama_tokenizer.pad_token = llama_tokenizer.eos_token
# Перемещаем на GPU, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llama_model.to(device)
search_model.to(device)

print("\nМодели DistilBERT и LLaMA успешно загружены.")


# === Функция для извлечения эмбеддингов запроса ===
def get_query_embedding(query):
    inputs = search_tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = search_model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :]  # Используем CLS токен
    return query_embedding


# === Функция для поиска похожих абзацев ===
def find_most_similar_paragraph(query, embeddings, paragraphs, similarity_threshold=0.7):
    query_embedding = get_query_embedding(query)

    # Перемещаем embeddings на то же устройство, что и query_embedding
    embeddings = embeddings.to(query_embedding.device)

    similarities = torch.nn.functional.cosine_similarity(query_embedding, embeddings)
    most_similar_idx = torch.argmax(similarities).item()

    # Применяем порог для схожести
    if similarities[most_similar_idx] < similarity_threshold:
        return None, None  # Если сходство слишком низкое, возвращаем None

    return paragraphs[most_similar_idx], similarities[most_similar_idx].item()


# === Функция для генерации ответа с помощью LLaMA ===
def generate_llama_response(context, query, language="ru"):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer (in Russian):"
    inputs = llama_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Ensure the attention_mask is included and that pad_token_id is set to eos_token_id
    inputs = inputs.to(device)
    pad_token_id = llama_tokenizer.eos_token_id  # Set pad_token_id to eos_token_id

    # Generate the response
    with torch.no_grad():
        outputs = llama_model.generate(
            inputs["input_ids"],
            max_length=700,
            num_beams=5,
            attention_mask=inputs["attention_mask"],  # Pass the attention mask
            pad_token_id=pad_token_id,  # Explicitly set pad_token_id
            repetition_penalty=2.0,  # Штраф за повторения
            early_stopping=True,
            do_sample=True,
            top_p=0.5,
            temperature=0.2,
            top_k=60,
        )

    # Декодируем и выбираем лучший ответ
    response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# === Основной цикл ожидания запросов ===
def interactive_qa():
    print("\nСистема готова к работе. Задайте свой вопрос (введите 'exit' для выхода).\n")
    while True:
        query = input("Ваш вопрос: ")
        if query.lower() in ["exit", "quit"]:
            print("\nЗавершение работы. Спасибо!")
            break

        # Найти самый похожий абзац
        most_similar_paragraph, similarity_score = find_most_similar_paragraph(query, embeddings, paragraphs)
        if not most_similar_paragraph:
            print("\nНе удалось найти релевантный абзац.\n")
            continue

        print(f"\nНаиболее похожий абзац (сходство {similarity_score:.2f}):\n{most_similar_paragraph}\n")

        # Сгенерировать ответ
        response = generate_llama_response(most_similar_paragraph, query)
        print(f"Ответ модели:\n{response}\n")


# === Запуск программы ===
if __name__ == "__main__":
    interactive_qa()
