from transformers import AutoTokenizer, AutoModel
import torch

# Проверяем наличие GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

# Загрузка модели и токенизатора
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)  # Перемещаем модель на GPU

# Загрузка книги
with open("Лавкрафт Говард Филлипс. Ужас в музее - royallib.com.txt", "r", encoding="utf-8") as file:
    book_text = file.read()

# Разделение текста на фрагменты фиксированной длины (например, ~200 слов)
def split_into_chunks(text, chunk_size=200):
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

paragraphs = split_into_chunks(book_text, chunk_size=200)

# Создание эмбеддингов для каждого фрагмента в батчах
batch_size = 16  # Настройте в зависимости от объема памяти
embeddings = []

for i in range(0, len(paragraphs), batch_size):
    batch = paragraphs[i:i + batch_size]
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)  # Данные на GPU
    with torch.no_grad():  # Выключаем автоград для экономии памяти
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS токен
        embeddings.append(cls_embeddings.cpu())  # Перемещаем результаты на CPU для сохранения

    print(f"Обработано {i + len(batch)} фрагментов из {len(paragraphs)}")

# Объединяем все эмбеддинги
embeddings = torch.cat(embeddings)

# Сохраняем эмбеддинги и текстовые фрагменты
torch.save(embeddings, "lembeddings.pt")
torch.save(paragraphs, "lparagraphs.pt")

print("Обработка завершена, данные сохранены.")
