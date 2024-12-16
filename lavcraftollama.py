import torch
import ollama
from transformers import AutoTokenizer, AutoModel

# === Загрузка эмбеддингов и текста ===
embeddings = torch.load("lembeddings.pt")
paragraphs = torch.load("lparagraphs.pt")

print(f"\nЗагружено {len(paragraphs)} абзацев и эмбеддингов.")

# === Загрузка модели для поиска похожих текстов ===
search_model_name = "distilbert-base-uncased"
search_tokenizer = AutoTokenizer.from_pretrained(search_model_name)
search_model = AutoModel.from_pretrained(search_model_name).eval()

# === Загрузка модели LLaMA через ollama ===
llama_model_name = "llama3.2-vision"  # Используем модель LLaMA через ollama

# Перемещаем на GPU, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Перемещаем модель на нужное устройство
search_model = search_model.to(device)

print("\nМодель DistilBERT успешно загружена.")

# === Функция для извлечения эмбеддингов запроса ===
def get_query_embedding(query):
    # Перемещаем запрос на устройство, где находится модель
    inputs = search_tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = search_model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :]  # Используем CLS токен
    return query_embedding

def find_most_similar_paragraph(query, embeddings, paragraphs, similarity_threshold=0.7):
    query_embedding = get_query_embedding(query)

    # Перемещаем embeddings на то же устройство, что и query_embedding
    embeddings = embeddings.to(device)

    similarities = torch.nn.functional.cosine_similarity(query_embedding, embeddings)
    most_similar_idx = torch.argmax(similarities).item()

    # Применяем порог для схожести
    if similarities[most_similar_idx] < similarity_threshold:
        return None, None  # Если сходство слишком низкое, возвращаем None

    return paragraphs[most_similar_idx], similarities[most_similar_idx].item()

# === Функция для генерации ответа с помощью LLaMA через ollama ===
def generate_llama_response(context, query, language="ru"):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer (in Russian):"

    # Генерация ответа через ollama
    response = ollama.chat(model=llama_model_name, messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ])

    print(response)  # Для диагностики
    return response.get('text', 'Ответ не найден')

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
